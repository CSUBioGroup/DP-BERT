import pandas as pd
import random
import torch
import numpy as np 

def getVocab(number):
    max_number = number
    vocab = {'0': 0, '<mask>': 1}
    for i in range(max_number):
        if str(i) != '0':
            vocab[str(i)] = i+1

    return vocab

def _replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds,
                        vocab):

    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []
    random.shuffle(candidate_pred_positions)
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        if random.random() < 0.8:
            masked_token = '<mask>'
        else:
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            else:
                masked_token=random.choice(list(vocab)) 
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position]))
    return mlm_input_tokens, pred_positions_and_labels

def _get_mlm_data_from_tokens(tokens, vocab): 

    candidate_pred_positions = []
    for i, token in enumerate(tokens):
        if token in ['0']:
            continue
        candidate_pred_positions.append(i)
    num_mlm_preds = max(1, round(len(candidate_pred_positions) * 0.15))
    
    mlm_input_tokens, pred_positions_and_labels = _replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab)


    pred_positions_and_labels = sorted(pred_positions_and_labels,
                                       key=lambda x: x[0])
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]

    vocab_mlm_input_tokens = [vocab[key] for key in mlm_input_tokens]
    vocab_mlm_pred_labels = [vocab[key] for key in mlm_pred_labels]

    return vocab_mlm_input_tokens, pred_positions, vocab_mlm_pred_labels,len(candidate_pred_positions)

def _pad_bert_inputs(examples, max_len):
    max_num_mlm_preds = round(max_len * 0.15)
    all_token_ids, valid_lens,  = [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    for (token_ids, pred_positions, mlm_pred_label_ids,real_len) in examples:
        all_token_ids.append(torch.tensor(token_ids, dtype=torch.long))
        valid_lens.append(torch.tensor(real_len, dtype=torch.float32))
        print(max_num_mlm_preds)
        print(len(pred_positions))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)),
                dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))

    return (all_token_ids, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels)

class geneDataset(torch.utils.data.Dataset):
    def __init__(self,path,max_len,vocab):

        with open(path, 'r') as file:
            paragraphs = []
            while True:
                line = file.readline()
                if not line:
                   break
                line = line.strip().split(',')
                paragraphs.append(line)

        examples = []
        for paragraph in paragraphs:
            mlm_input_tokens, pred_positions, mlm_pred_labels,real_len=_get_mlm_data_from_tokens(paragraph,vocab)
            examples.append((mlm_input_tokens, pred_positions, mlm_pred_labels,real_len))  
        (self.all_token_ids,self.valid_lens,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels) = _pad_bert_inputs(
            examples, max_len)

    def __getitem__(self, idx):
        rand_start = random.randint(0, len(self.all_token_ids)-1)

        return (self.all_token_ids[rand_start],
                self.valid_lens[rand_start], self.all_pred_positions[rand_start],
                self.all_mlm_weights[rand_start], self.all_mlm_labels[rand_start])

    def __len__(self):
        return len(self.all_token_ids)

def load_data(batch_size,path,max_len,vocab):
    train_set = geneDataset(path,max_len,vocab)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size,
                                        shuffle=True, num_workers=4)
    return train_iter,train_set
