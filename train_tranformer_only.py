import argparse
import logging
import os
import random
import sys
from tqdm import tqdm
import numpy as np


from transformers import BertConfig
from transformers import  AdamW
from torch.optim.lr_scheduler import LambdaLR

import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from torch.utils.data import DataLoader
from torchnet.meter import ConfusionMeter
from torchnet.meter import AUCMeter
from ignite.metrics import Precision
from sklearn import metrics


from data_prep.yelp_dataset import get_yelp_datasets
from data_prep.chn_hotel_dataset import get_chn_htl_datasets
from data_prep.event_seq_dataset import get_event_seq_datasets

from models import *
from options_med import opt
from vocab_med import Vocab
import utils

random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
np.random.seed(opt.random_seed)

torch.cuda.empty_cache()
import gc
gc.collect()

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# save logs
if not os.path.exists(opt.model_save_file):
    os.makedirs(opt.model_save_file)
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG if opt.debug else logging.INFO)
log = logging.getLogger(__name__)
fh = logging.FileHandler(os.path.join(opt.model_save_file, 'log.txt'))
log.addHandler(fh)
 
# output options
log.info('Training ADAN with options:')
log.info(opt)
 

def train(opt):
    # vocab
    log.info(f'Loading Embeddings...')
    vocab = Vocab(opt.emb_filename)
    # datasets
    log.info(f'Loading data...')
    yelp_X_train = os.path.join(opt.src_data_dir, 'X_all.aligned.txt')
    yelp_Y_train = os.path.join(opt.src_data_dir, 'Y_all.aligned.txt')
    yelp_X_val = os.path.join(opt.src_data_dir, 'X_val.aligned.txt')
    yelp_Y_val = os.path.join(opt.src_data_dir, 'Y_val.aligned.txt')
    yelp_X_test = os.path.join(opt.src_data_dir, 'X_test.aligned.txt')
    yelp_Y_test = os.path.join(opt.src_data_dir, 'Y_test.aligned.txt')
    yelp_train, yelp_valid, yelp_test = get_event_seq_datasets(vocab, yelp_X_train, yelp_Y_train,
            opt.en_train_lines, yelp_X_val, yelp_Y_val, yelp_X_test, yelp_Y_test, opt.max_seq_len)
    mimic_train_X_file = os.path.join(opt.tgt_data_dir, 'X_train_filtered.txt')
    mimic_train_Y_file = os.path.join(opt.tgt_data_dir, 'Y_train.txt')

    mimic_val_X_file = os.path.join(opt.tgt_data_dir, 'X_val_filtered.txt')
    mimic_val_Y_file = os.path.join(opt.tgt_data_dir, 'Y_val.txt')

    mimic_test_X_file = os.path.join(opt.tgt_data_dir, 'X_test_filtered.txt')
    mimic_test_Y_file = os.path.join(opt.tgt_data_dir, 'Y_test.txt')

    mimic_train, mimic_valid, mimic_test = get_event_seq_datasets(vocab, mimic_train_X_file, mimic_train_Y_file,
            opt.ch_train_lines, mimic_val_X_file, mimic_val_Y_file, mimic_test_X_file, mimic_test_Y_file, opt.max_seq_len)
    log.info('Done loading datasets.')
    opt.num_labels = yelp_train.num_labels

    if opt.max_seq_len <= 0:
        # set to true max_seq_len in the datasets
        opt.max_seq_len = max(yelp_train.get_max_seq_len(),
                              mimic_train.get_max_seq_len())
    # dataset loaders
    my_collate = utils.sorted_collate if opt.model=='lstm' else utils.unsorted_collate
    yelp_train_loader = DataLoader(yelp_train, opt.batch_size,
            shuffle=True, collate_fn = my_collate)
    yelp_train_loader_Q = DataLoader(yelp_train,
                                     opt.batch_size,
                                     shuffle=True, collate_fn=my_collate)
    mimic_train_loader = DataLoader(mimic_train, opt.batch_size,
            shuffle=True, collate_fn=my_collate)
    mimic_train_loader_Q = DataLoader(mimic_train,
                                    opt.batch_size,
                                    shuffle=True, collate_fn=my_collate)
    yelp_train_iter_Q = iter(yelp_train_loader_Q)
    mimic_train_iter = iter(mimic_train_loader)
    mimic_train_iter_Q = iter(mimic_train_loader_Q)

    yelp_valid_loader = DataLoader(yelp_valid, opt.batch_size,
            shuffle=False, collate_fn=my_collate)
    yelp_test_loader = DataLoader(yelp_test, opt.batch_size,
            shuffle=False, collate_fn=my_collate)
    mimic_valid_loader = DataLoader(mimic_valid, opt.batch_size,
            shuffle=False, collate_fn=my_collate)
    mimic_test_loader = DataLoader(mimic_test, opt.batch_size,
            shuffle=False, collate_fn=my_collate)

    # models
    if opt.model.lower() == 'dan':
        F = DANFeatureExtractor(vocab, opt.F_layers, opt.hidden_size, opt.dropout, opt.F_bn)
    elif opt.model.lower() == 'lstm':
        F = LSTMFeatureExtractor(vocab, opt.F_layers, opt.hidden_size, opt.dropout,
                opt.bdrnn, opt.attn)
    elif opt.model.lower() == 'cnn':
        F = CNNFeatureExtractor(vocab, opt.F_layers,
                opt.hidden_size, opt.kernel_num, opt.kernel_sizes, opt.dropout)
    elif opt.model.lower() == 'transformer':
        
        config = BertConfig(
        hidden_size = vocab.emb_size,
        max_position_embeddings = opt.max_pos_emb,
        num_attention_heads = opt.head_num,
        num_hidden_layers = opt.F_layers,
        vocab_size = opt.vocab_size)
        F = TransformerFeatureExtractor(vocab, opt.hidden_size, config = config)#TransformerModel(vocab, opt.emb_size, opt.head_num, opt.hidden_size, opt.F_layers,opt.dropout)
    else:
        raise Exception('Unknown model')
    P = SentimentClassifier(opt.P_layers, opt.hidden_size, 2,
            opt.dropout, opt.P_bn)
    #Q = LanguageDetector(opt.Q_layers, opt.hidden_size, opt.dropout, opt.Q_bn)
    F, P = F.to(opt.device), P.to(opt.device)
    
    optimizer = AdamW(list(F.parameters()) + list(P.parameters()),
                           lr=opt.learning_rate, weight_decay=0.01)
    #optimizerQ = optim.Adam(Q.parameters(), lr=opt.Q_learning_rate)
    num_training_steps = len(yelp_train_loader) * opt.max_epoch
    num_warmup_steps = opt.num_warmup_steps
    scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=num_warmup_steps, 
    num_training_steps=num_training_steps)

    # training
    best_acc = 0.0
    best_AUC = 0.0
    best_prec = 0.0
    for epoch in range(opt.max_epoch):
        F.train()
        P.train()
        #Q.train()
        yelp_train_iter = iter(yelp_train_loader)
        # training accuracy
        correct, total = 0, 0
        y_score = []
        y_true = []
        sum_en_q, sum_ch_q = (0, 0.0), (0, 0.0)
        grad_norm_p, grad_norm_q = (0, 0.0), (0, 0.0)
        #m = tf.keras.metrics.AUC()
        for i, (inputs_en, targets_en) in tqdm(enumerate(yelp_train_iter),
                                               total=len(yelp_train)//opt.batch_size):
            try:
                inputs_ch, _ = next(mimic_train_iter)  # MIMIC labels are not used
            except:
                # check if MIMIC data is exhausted
                mimic_train_iter = iter(mimic_train_loader)
                inputs_ch, _ = next(mimic_train_iter)

            # Q iterations

            # F&P iteration
            #utils.unfreeze_net(F)
            #utils.unfreeze_net(P)
            #utils.freeze_net(Q)
            if opt.fix_emb:
                utils.freeze_net(F.word_emb)
            # clip Q weights
            #for p in Q.parameters():
                #p.data.clamp_(opt.clip_lower, opt.clip_upper)
            F.zero_grad()
            P.zero_grad()
            
            features_en = F(inputs_en)
            o_en_sent = P(features_en)
            
            #targets_en = targets_en.unsqueeze(1)
            #targets_en = targets_en.to(torch.float32)
            pos_weight = torch.tensor([0.2,0.8]).to(opt.device)

            loss = nn.CrossEntropyLoss(weight = pos_weight)#nn.BCEWithLogitsLoss(pos_weight = pos_weight)
            l_en_sent = loss(o_en_sent, targets_en)
            l_en_sent.backward(retain_graph=True)
            

            # training accuracy
            total += targets_en.size(0)

            y_score +=[o_en_sent[:,1].cpu().detach().numpy()]
            y_true +=[targets_en.cpu().detach().numpy()]            

            optimizer.step()
            scheduler.step()
    
        # end of epoch
        

        log.info('Ending epoch {}'.format(epoch+1))
        
        fpr, tpr, _ = metrics.roc_curve(np.concatenate(y_true, axis = 0), np.concatenate(y_score, axis = 0))
        roc_auc = metrics.auc(fpr, tpr)

        log.info('Training AUC: {}%'.format(100.0*roc_auc))
        log.info('***Trained on all UCLA data available***')

        log.info('Evaluating MIMIC Validation set:')
        (AUC, prauc) = evaluate(opt, mimic_valid_loader, F, P)
        log.info('Evaluating MIMIC test set:')
        (AUC_t, prauc_t) = evaluate(opt, mimic_test_loader, F, P)
        #print('Validation AUC: ', AUC)
        #print('Validation prAUC: ', prauc)
        if AUC > best_AUC:
            log.info(f'New Best MIMIC validation accuracy: {AUC}')
            best_AUC = AUC
            torch.save(F.state_dict(),
                    '{}/netF_epoch_{}.pth'.format(opt.model_save_file, epoch))
            torch.save(P.state_dict(),
                    '{}/netP_epoch_{}.pth'.format(opt.model_save_file, epoch))
        #log.info('Evaluating UCLA test set:')
        #evaluate(opt, yelp_test_loader, F, P)
        log.info('Evaluating MIMIC test set:')
        evaluate(opt, mimic_test_loader, F, P)
    
    #log metrics/parameters   
    import csv    
    with open("../param_search/exp_logs_transformer_baseline.csv", 'a', newline='') as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=['val_AUC','val_prAUC','test_AUC', 'test_prAUC', 'random_emb',  'fix_emb', 'dropout', 'p_lr',    
                'F_layer','P_layer','max_seq_len','hidden_size','emb_size','emb_filename', 'fix_unk',  'max_pos_emb',
                 'vocab_num', 'num_head','optimizer', 'p_weight_decay',  'num_warmup', 'max_epoch','target_data' ])
        writer.writerow({'val_AUC': str(best_AUC),'val_prAUC': str(best_prAUC),'test_AUC': str(best_AUC_t),  'test_prAUC': str(best_prAUC_t),  
                'random_emb': opt.random_emb,  'fix_emb': opt.fix_emb, 'dropout': str(opt.dropout), 'p_lr': str(opt.learning_rate),    
                'F_layer':str(opt.F_layers), 'P_layer':str(opt.P_layers), 'max_seq_len':str(opt.max_seq_len), 
                'hidden_size': str(opt.hidden_size), 'emb_size':str(opt.emb_size), 'emb_filename': opt.emb_filename, 'fix_unk': opt.fix_unk,  
                'max_pos_emb':str(opt.max_pos_emb) , 'vocab_num': str(opt.vocab_size),   'num_head': str(opt.head_num),'optimizer': 'AdamW', 
                'p_weight_decay': str(0.01), 'num_warmup': str(opt.num_warmup_steps),
                'max_epoch': str(opt.max_epoch), 'target_data' : opt.tgt_data_dir})
    csvFile.close()

    log.info(f'Best UCLA validation Acc: {best_AUC}')

def evaluate(opt, loader, F, P):
    F.eval()
    P.eval()
    it = iter(loader)
    correct = 0
    total = 0
    y_score = []
    y_true = []
    auc = AUCMeter()
    with torch.no_grad():
        for inputs, targets in tqdm(it):
            #print(inputs[0])
            outputs = P(F(inputs))

            targets = targets.unsqueeze(1)
            targets = targets.to(torch.float32)

            y_score+=[outputs[:,1].cpu().detach().numpy()]
            y_true+=[targets.cpu().detach().numpy()]
            total += targets.size(0)
            #correct += (outputs == targets).sum().item()
    fpr, tpr, _ = metrics.roc_curve(np.concatenate(y_true, axis = 0), np.concatenate(y_score, axis = 0))
    roc_auc = metrics.auc(fpr, tpr)
    prauc = metrics.average_precision_score(np.concatenate(y_true, axis = 0), np.concatenate(y_score, axis = 0))
    #AUC = auc.value()[0]
    log.info('AUC on {} samples: {}%'.format(total, 100.00*roc_auc))
    log.info('prAUC on {} samples: {}%'.format(total, 100.00*prauc))
    return (roc_auc, prauc)

if __name__ == '__main__':
    train(opt)
