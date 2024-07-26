# -*- coding: utf-8 -*-
import os
import gc
import argparse
import json
import random
import math
from functools import reduce
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, ShuffleSplit, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_recall_fscore_support, classification_report,precision_score, recall_score, roc_auc_score
from torch import nn
from torch.optim import Adam, SGD, AdamW
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CyclicLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from performer_pytorch import PerformerLM
from utils import *
import pickle as pkl
from collections import OrderedDict
from sklearn.metrics import roc_curve, auc
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--bin_num", type=int, default=2, help='Number of bins.')
parser.add_argument("--epoch", type=int, default=100, help='Number of epochs.')
parser.add_argument("--seed", type=int, default=2021, help='Random seed.')
parser.add_argument("--batch_size", type=int, default=7, help='Number of batch size.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--grad_acc", type=int, default=5, help='Number of gradient accumulation.')
parser.add_argument("--valid_every", type=int, default=1, help='Number of training epochs between twice validation.')
parser.add_argument("--pos_embed", type=bool, default=True, help='Using Gene2vec encoding or not.')
parser.add_argument("--data_path", type=str, default='./data/Zheng68K.h5ad', help='Path of data for finetune.')
parser.add_argument("--model_path", type=str, default='/home/pretrain_model.pkl', help='Path of pretrained model.')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='finetune', help='Finetuned model name.')
args = parser.parse_args()

rank = int(os.environ["RANK"])
local_rank = args.local_rank
is_master = local_rank == 0
SEED = args.seed
EPOCHS = args.epoch
BATCH_SIZE = args.batch_size
GRADIENT_ACCUMULATION = args.grad_acc
LEARNING_RATE = args.learning_rate
SEQ_LEN = 24447
VALIDATE_EVERY = args.valid_every

PATIENCE = 10
UNASSIGN_THRES = 0.0

model_name = args.model_name
ckpt_dir = args.ckpt_dir

dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)
world_size = torch.distributed.get_world_size()

seed_all(SEED + torch.distributed.get_rank())


class GeneDataset(Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data = data
        self.label = label

    def __getitem__(self, index):
        rand_start = random.randint(0, self.data.shape[0]-1)
        full_seq = self.data[rand_start]
        full_seq = torch.from_numpy(full_seq).long()
        seq_label = self.label[rand_start]
        return full_seq, seq_label

    def __len__(self):
        return self.data.shape[0]



class Identity(torch.nn.Module):
    def __init__(self, dropout = 0., h_dim = 128, out_dim = 2):
        super(Identity, self).__init__()
        self.conv1 = nn.Conv2d(1, 1, (1, 200))
        self.act = nn.ReLU()
        self.fc1 = nn.Linear(in_features=SEQ_LEN, out_features=512, bias=True)
        self.act1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout) 
        self.fc2 = nn.Linear(in_features=512, out_features=h_dim, bias=True)
        self.act2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout) 
        self.fc3 = nn.Linear(in_features=h_dim, out_features=out_dim, bias=True)

    def forward(self, x):

        x = x[:,None,:,:]
        x = self.conv1(x)
        x = self.act(x)
        x = x.view(x.shape[0],-1)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)

        return x

df = pd.read_csv('/home/ Depression-20_bin_test.csv')

for column in df.columns[:-1]:
    df[column] = df[column].apply(lambda x: x + 1 if x != 0 else x)

train_set, test_set = train_test_split(df, test_size=0.1, stratify=df.iloc[:, -1], random_state=42)

data = train_set.iloc[:, :-1].values 
label = train_set.iloc[:, -1].values  

label = torch.from_numpy(label).long()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


def exists(val):
    return val is not None

class FineTunedPerformerLM(PerformerLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if hasattr(self, 'mlm'):
            delattr(self, 'mlm')

    def forward(self, x, return_encodings = False, output_attentions = False, **kwargs):

        # token and positional embedding
        x_token = self.token_emb(x)

        if output_attentions:
            x.requires_grad_()    # used for attn_map output

     
        x =x_token+self.pos_emb(x)

        x = self.dropout(x)

        if output_attentions:
            x, attn_weights = self.performer(x, output_attentions = output_attentions, **kwargs)
            # norm and to logits
            x = self.norm(x)
            if exists(self.to_out):
                return self.to_out(x), attn_weights
            if return_encodings:
                return x, attn_weights        
            return (x @ self.token_emb.weight.t()), attn_weights
        else:

            x = self.performer(x, output_attentions = output_attentions, **kwargs)
            # norm and to logits
            x = self.norm(x)

            if exists(self.to_out):
                x = self.to_out(x)
                return x
            if return_encodings:
                return x
            return x @ self.token_emb.weight.t()

def load_pretrained_model(model_path):
    # Load model weights
    ckpt = torch.load(model_path, map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        # Exclude parameters related to the MLP layer for mask task
        if "mlp" not in k:
            new_state_dict[k] = v
    keys = new_state_dict.keys()
    # Create a new model instance and load modified state dict
    model = FineTunedPerformerLM(num_tokens=20002, dim=200, depth=6, max_seq_len=24447, heads=10, vocab_size=20002)
    model.load_state_dict(new_state_dict)
    return model

dist.barrier()


all_accs = []
all_precisions = []
all_recalls = []
all_f1s = []
all_aucs = []



for fold_idx, (index_train, index_val) in enumerate(skf.split(data, label), 1):

    data_train, label_train = data[index_train], label[index_train]
    data_val, label_val = data[index_val], label[index_val]

    train_dataset = GeneDataset(data_train, label_train)
    val_dataset = GeneDataset(data_val, label_val)
    
    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)


    path = args.model_path
    model = load_pretrained_model(path)

    for param in model.parameters():
        param.requires_grad = False
    for param in model.norm.parameters():
        param.requires_grad = True
    for param in model.performer.net.layers[5].parameters():
        param.requires_grad = True     
                                                                           
    model.to_out = Identity(dropout=0., h_dim=128, out_dim=2)

    model = model.to(device)

    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# optimizer
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=15,
        cycle_mult=2,
        max_lr=LEARNING_RATE,
        min_lr=1e-6,
        warmup_steps=5,
        gamma=0.9
)
    loss_fn = nn.CrossEntropyLoss(weight=None).to(local_rank)
    dist.barrier()    
    trigger_times = 0
    max_acc = 0.0

    for i in range(1, EPOCHS+1):
        train_loader.sampler.set_epoch(i)
        model.train()
        dist.barrier()
        running_loss = 0.0
        cum_acc = 0.0
        for index, (datas, labels) in enumerate(train_loader):
            index += 1
            datas, labels = datas.to(device), labels.to(device)
            if index % GRADIENT_ACCUMULATION != 0:
                with model.no_sync():
                    logits = model(datas)
                    loss = loss_fn(logits, labels)
                    loss.backward()
            if index % GRADIENT_ACCUMULATION == 0:
                logits = model(datas)
                loss = loss_fn(logits, labels)
                loss.backward()                 
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item()
            softmax = nn.Softmax(dim=-1)
            final = softmax(logits)
            final = final.argmax(dim=-1)
            pred_num = labels.size(0)
            correct_num = torch.eq(final, labels).sum(dim=-1)
            cum_acc += torch.true_divide(correct_num, pred_num).mean().item()
        epoch_loss = running_loss / index
        epoch_acc = 100 * cum_acc / index
        epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
        epoch_acc = get_reduced(epoch_acc, local_rank, 0, world_size)
        
        if is_master:
            print(f'    ==  Epoch: {i} | Training Loss: {epoch_loss:.6f} | Accuracy: {epoch_acc:6.4f}%  ==')
        dist.barrier()
        scheduler.step()

        if i % VALIDATE_EVERY == 0:
            model.eval()
            dist.barrier()
            running_loss = 0.0
            predictions = []
            truths = []
            predict_proba = []
            with torch.no_grad():
                for index, (data_v, labels_v) in enumerate(val_loader):
                    index += 1
                    data_v, labels_v = data_v.to(device), labels_v.to(device)
                    logits = model(data_v)
                    loss = loss_fn(logits, labels_v)
                    running_loss += loss.item()
                    softmax = nn.Softmax(dim=-1)
                    final_prob = softmax(logits)
                    final_prob_class1 = final_prob[:, 1]
                    final = final_prob.argmax(dim=-1)
                    final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
                    predictions.append(final)
                    truths.append(labels_v)
                    predict_proba.append(final_prob_class1)
                del data_v, labels_v, logits, final_prob, final,final_prob_class1
                predictions = distributed_concat(torch.cat(predictions, dim=0), len(val_sampler.dataset), world_size)
                truths = distributed_concat(torch.cat(truths, dim=0), len(val_sampler.dataset), world_size)
                predict_proba = distributed_concat(torch.cat(predict_proba, dim=0), len(val_sampler.dataset), world_size)
                no_drop = predictions != -1
                predictions = np.array((predictions[no_drop]).cpu())
                truths = np.array((truths[no_drop]).cpu())
                predict_proba = np.array((predict_proba[no_drop]).cpu())
                cur_acc = accuracy_score(truths, predictions)
                f1 = f1_score(truths, predictions, average='macro')
                precision = precision_score(truths, predictions)
                recall = recall_score(truths, predictions)
                roc_auc = roc_auc_score(truths, predict_proba)
                val_loss = running_loss / index
                val_loss = get_reduced(val_loss, local_rank, 0, world_size)
                if is_master:
                    print(f'    ==  Epoch: {i} | Validation Loss: {val_loss:.6f} | ACC: {cur_acc:.6f} | Precision: {precision:.6f} | Recall: {recall:.6f} | F1-score: {f1:.6f} | AUC: {roc_auc:.6f}  ==')
                if cur_acc > max_acc:
                    max_acc = cur_acc
                    trigger_times = 0
                    torch.save(model.state_dict(), '/home/best_model_finetune.pth')
                else:
                    trigger_times += 1
                    if trigger_times > PATIENCE:
                        break
    


    all_accs.append(cur_acc)
    all_precisions.append(precision)
    all_recalls.append(recall)
    all_f1s.append(f1)
    all_aucs.append(auc)

avg_acc = np.mean(all_accs)
avg_precision = np.mean(all_precisions)
avg_recall = np.mean(all_recalls)
avg_f1 = np.mean(all_f1s)
avg_auc = np.mean(all_aucs)


if is_master:
    print(f'Average Accuracy: {avg_acc:.6f}')
    print(f'Average Precision: {avg_precision:.6f}')
    print(f'Average Recall: {avg_recall:.6f}')
    print(f'Average F1-score: {avg_f1:.6f}')
    print(f'Average AUC: {avg_auc:.6f}')

def load_pretrained_model(model_path):
    # Load model weights
    ckpt = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k, v in ckpt.items():
        # Exclude parameters related to the MLP layer for mask task
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    keys = new_state_dict.keys()

    # Create a new model instance and load modified state dict
    model = FineTunedPerformerLM(num_tokens=20002, dim=200, depth=6, max_seq_len=24447, heads=10, vocab_size=20002)
    model.to_out = Identity(dropout=0., h_dim=128, out_dim=2)
    model.load_state_dict(new_state_dict)
    return model
model = load_pretrained_model('/home/best_model_finetune.pth')
model = model.to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)
model.eval()

data_test = test_set.iloc[:, :-1].values  
label_test = test_set.iloc[:, -1].values  
test_dataset = GeneDataset(data_test, label_test) 
test_sampler = DistributedSampler(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=test_sampler) 

model.eval()
running_loss = 0.0
predictions = []
truths = []
predict_proba=[]
loss_fn = nn.CrossEntropyLoss(weight=None).to(local_rank)
with torch.no_grad():
    for index, (data_t, labels_t) in enumerate(test_loader):
        data_t, labels_t = data_t.to(device), labels_t.to(device)
        logits = model(data_t)
        loss = loss_fn(logits, labels_t)
        running_loss += loss.item()
        softmax = nn.Softmax(dim=-1)
        final_prob = softmax(logits)
        final_prob_class1 = final_prob[:, 1]
        final = final_prob.argmax(dim=-1)
        final[np.amax(np.array(final_prob.cpu()), axis=-1) < UNASSIGN_THRES] = -1
        predictions.append(final)
        truths.append(labels_t)
        predict_proba.append(final_prob_class1)
    del data_t, labels_t, logits, final_prob, final,final_prob_class1
    # gather
    predictions = distributed_concat(torch.cat(predictions, dim=0), len(test_dataset), world_size)
    truths = distributed_concat(torch.cat(truths, dim=0), len(test_dataset), world_size)
    predict_proba = distributed_concat(torch.cat(predict_proba, dim=0), len(test_dataset), world_size)
    no_drop = predictions != -1
    predictions = np.array((predictions[no_drop]).cpu())
    truths = np.array((truths[no_drop]).cpu())
    predict_proba = np.array((truths[no_drop]).cpu())

    test_loss = running_loss / index
    test_loss = get_reduced(test_loss, local_rank, 0, world_size)
    test_acc = accuracy_score(truths, predictions)
    f1 = f1_score(truths, predictions, average='macro')
    precision = precision_score(truths, predictions)
    recall = recall_score(truths, predictions)
    roc_auc = roc_auc_score(truths, predict_proba) 

    print(f'Test Loss: {test_loss:.6f} | Accuracy: {test_acc:.6f} | Precision: {precision:.6f} | Recall: {recall:.6f} | F1-score: {f1:.6f} | AUC: {roc_auc:.6f}')

