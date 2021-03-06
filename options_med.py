import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument('--max_epoch', type=int, default=30)
# parser.add_argument('--dataset', default='yelp')  # yelp, yelp-aren or amazon
# path to the datasets
parser.add_argument('--src_data_dir', default='../data/UCLA_labs/')
parser.add_argument('--tgt_data_dir', default='../data/mimic/')
parser.add_argument('--en_train_lines', type=int, default=0)  # set to 0 to use all
parser.add_argument('--ch_train_lines', type=int, default=0)  # set to 0 to use all
parser.add_argument('--max_seq_len', type=int, default=0) # set to 0 to not truncate
parser.add_argument('--random_seed', type=int, default=1)
parser.add_argument('--model_save_file', default='./save/adan')
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=0.00005)
parser.add_argument('--Q_learning_rate', type=float, default=0.0005)
# path to BWE
parser.add_argument('--emb_filename', default='../data/ucla_mimic_emb/emb_aligned_with_mimic_loinc.txt')
parser.add_argument('--fix_emb', default = False)
parser.add_argument('--random_emb', default= True)
# use a fixed <unk> token for all words without pretrained embeddings when building vocab
parser.add_argument('--fix_unk', default = True)
parser.add_argument('--emb_size', type=int, default=768)
parser.add_argument('--model', default='cnn')  # dan or lstm or cnn
# for LSTM model
parser.add_argument('--attn', default='dot')  # attention mechanism (for LSTM): avg, last, dot
parser.add_argument('--bdrnn', dest='bdrnn', action='store_true', default=True)  # bi-directional LSTM
# use deep averaging network or deep summing network (for DAN model)
parser.add_argument('--sum_pooling/', dest='sum_pooling', action='store_true')
parser.add_argument('--avg_pooling/', dest='sum_pooling', action='store_false')
# for CNN model
parser.add_argument('--kernel_num', type=int, default=400)
parser.add_argument('--kernel_sizes', type=int, nargs='+', default=[3,4,4])
parser.add_argument('--hidden_size', type=int, default=900)

#for transformer encoder
parser.add_argument('--head_num', type=int, default=12)
parser.add_argument('--t_hidden_size', type=int, default=768)
parser.add_argument('--num_warmup_steps', type=int, default=500)
parser.add_argument('--num_warmup_steps_Q', type=int, default=3122)
parser.add_argument('--max_pos_emb', type=int, default=1024)
parser.add_argument('--vocab_size', type=int, default=2370)


parser.add_argument('--F_layers', type=int, default=1)
parser.add_argument('--P_layers', type=int, default=2)
parser.add_argument('--Q_layers', type=int, default=2)
parser.add_argument('--n_critic', type=int, default=5)
parser.add_argument('--lambd', type=float, default=0.01)
parser.add_argument('--F_bn/', dest='F_bn', action='store_true')
parser.add_argument('--no_F_bn/', dest='F_bn', action='store_false')
parser.add_argument('--P_bn/', dest='P_bn', action='store_true', default=True)
parser.add_argument('--no_P_bn/', dest='P_bn', action='store_false')
parser.add_argument('--Q_bn/', dest='Q_bn', action='store_true', default=True)
parser.add_argument('--no_Q_bn/', dest='Q_bn', action='store_false')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--clip_lower', type=float, default=-0.01)
parser.add_argument('--clip_upper', type=float, default=0.01)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--debug/', dest='debug', action='store_true')
opt = parser.parse_args()

if not torch.cuda.is_available():
    opt.device = 'cpu'
