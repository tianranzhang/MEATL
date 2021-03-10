import pdb
import numpy as np
import torch
from torch.nn import ZeroPad2d
#from torch.utils.serialization import load_lua
from options_med import opt

def freeze_net(net):
    for p in net.parameters():
        p.requires_grad = False


def unfreeze_net(net):
    for p in net.parameters():
        p.requires_grad = True


def sorted_collate(batch):
    return my_collate(batch, sort=True)


def unsorted_collate(batch):
    return my_collate(batch, sort=False)


def my_collate(batch, sort):
    x, y = zip(*batch)
    #print(y)
    x, y = pad(x, y, opt.eos_idx, sort)
    x = (x[0].to(opt.device), x[1].to(opt.device),x[2].to(opt.device) )#(padded_inputs, lengths)
    y = y.to(opt.device)
    return (x, y)


def pad(x, y, eos_idx, sort):
    inputs, lengths = zip(*x)
    max_len = max(lengths)
    # pad sequences
    padded_inputs = torch.full((len(inputs), max_len), eos_idx, dtype=torch.long)
    for i, row in enumerate(inputs):
        assert eos_idx not in row, f'EOS in sequence {row}'
        padded_inputs[i][-len(row):] = torch.tensor(row, dtype=torch.long)
    lengths = torch.tensor(lengths, dtype=torch.long)
    
    attention_mask = torch.full((len(inputs), max_len), 0, dtype=torch.long)
    for i, row in enumerate(inputs):
        attention_mask[i][-len(row):] = torch.ones([1,len(row)], dtype=torch.long)
    #print(y)
    y = torch.tensor(y, dtype=torch.long).view(-1)
    #print(y)
    if sort:
        # sort by length
        sort_len, sort_idx = lengths.sort(0, descending=True)
        padded_inputs = padded_inputs.index_select(0, sort_idx)
        y = y.index_select(0, sort_idx)
        return (padded_inputs, sort_len), y
    else:
        #print(padded_inputs.size())
        #exit(0)
        return (padded_inputs, lengths, attention_mask), y


def zero_eos(emb, eos_idx):
    emb.weight.data[eos_idx].zero_()
