import torch
from torch import nn
import performer_pytorch as performer
import preTrainData as getData
from torch.utils.data import DataLoader
from preTrainData import geneDataset 
import torch.distributed as dist
import argparse
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", type=int, default=-1, help='Local process rank.')
parser.add_argument("--grad_acc", type=int, default=10, help='Number of gradient accumulation.')
parser.add_argument("--learning_rate", type=float, default=1e-4, help='Learning rate.')
parser.add_argument("--ckpt_dir", type=str, default='./ckpts/', help='Directory of checkpoint to save.')
parser.add_argument("--model_name", type=str, default='Pretrain_best', help='Finetuned model name.')
args = parser.parse_args()

model_name = args.model_name
ckpt_dir = args.ckpt_dir

BATCH_SIZE=5
PATIENCE = 10
rank = int(os.environ["RANK"])
local_rank = args.local_rank
LEARNING_RATE = args.learning_rate
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl') 
device = torch.device("cuda", local_rank) 
world_size = torch.distributed.get_world_size()

GRADIENT_ACCUMULATION = args.grad_acc



def _get_batch_loss_bert(net, loss, vocab_size, tokens_X,
                         pred_positions_X, mlm_weights_X,
                         mlm_Y):      

    _, mlm_Y_hat= net(tokens_X, pred_positions_X)
    mlm_l = loss(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *mlm_weights_X.reshape(-1, 1)
    mlm_l = mlm_l.sum() / (mlm_weights_X.sum() + 1e-8)

    return mlm_l


def train_bert(train_iter,max_len,net, loss, vocab, device, num_epochs):

    net=net.to(device)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank)
    trainer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)
  

    train_dataset = geneDataset(train_iter, max_len, vocab)
    train_sampler = DistributedSampler(train_dataset)
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    scheduler = CosineAnnealingWarmupRestarts(
    trainer,
    first_cycle_steps=15,
    cycle_mult=2,
    max_lr=LEARNING_RATE,
    min_lr=1e-6,
    warmup_steps=5,
    gamma=0.9
)

    dist.barrier()

    trigger_times = 0
    min_loss = float('inf')
    
    for epoch in range(num_epochs):
        dataloader.sampler.set_epoch(epoch)
        dist.barrier()
        running_loss = 0.0
        for i, (tokens_X, valid_lens_x, pred_positions_X, mlm_weights_X, mlm_Y) in enumerate(dataloader):               
                i += 1
                tokens_X = tokens_X.to(local_rank)
                valid_lens_x = valid_lens_x.to(local_rank)
                pred_positions_X = pred_positions_X.to(local_rank)
                mlm_weights_X = mlm_weights_X.to(local_rank)
                mlm_Y = mlm_Y.to(local_rank)

                if i % GRADIENT_ACCUMULATION != 0:

                    with net.no_sync():
                        
                        l = _get_batch_loss_bert(net, loss, len(vocab), tokens_X,
                                          pred_positions_X, mlm_weights_X, mlm_Y)
                        l.backward()  
                if i % GRADIENT_ACCUMULATION == 0:
                    l = _get_batch_loss_bert(net, loss, len(vocab), tokens_X,
                                          pred_positions_X, mlm_weights_X, mlm_Y)
                    l.backward()  
                    torch.nn.utils.clip_grad_norm_(net.parameters(), int(1e6))
                    trainer.step()
                    trainer.zero_grad()
                running_loss += l.item()
        epoch_loss = running_loss / i
        epoch_loss = get_reduced(epoch_loss, local_rank, 0, world_size)
        if dist.get_rank() == 0:
            print(f'    ==  Epoch: {epoch} | Training Loss: {epoch_loss:.6f}')
        if epoch_loss < min_loss:
            min_loss = epoch_loss
            trigger_times = 0
        else:
            trigger_times += 1
            if trigger_times > PATIENCE:
                break        
        dist.barrier()
        scheduler.step()

if __name__ == '__main__':   
    max_len =  24447
    vocab=getData.getVocab(20001)
    dataset ='/home/Psychiatric disorder-64_bin_test'
    net = performer.PerformerLM(
        num_tokens = 20002,        #The size of the token embedding
        dim = 200,                 #embedding dim
        depth = 6,                 #performer layers
        max_seq_len = 24447,       #max sequence length
        heads = 10,                #The number of heads in the multi-head attention 
        vocab_size=20002,          #The vocabulary size is the number of bins plus 2, with the additional two tokens representing 'padding' and 'mask'.
        num_hiddens=2048,          #The dimension of the hidden neurons in the MLP for the MLM pretraining task.
    )


    loss = nn.CrossEntropyLoss().to(local_rank)

    train_bert(dataset,max_len, net, loss, vocab, device, 10)

