import torch
from torch import autograd, nn
import torch.nn.functional as functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from layers import *
from options_med import opt


import torch
import math
import torch.nn.functional as F

from transformers import BertPreTrainedModel,BertModel

class TransformerFeatureExtractor(BertPreTrainedModel):
    def __init__(self, vocab, hidden_size, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        print(self.num_labels)
        self.word_emb =  vocab.init_embed_layer()
        self.emb_size = config.hidden_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.hidden_size = hidden_size

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds= None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        output_dim = None,
    ):
        input_ids, lengths, attention_mask = input_ids
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        inputs_embeds = self.word_emb(input_ids)

        outputs = self.bert(
            input_ids = None,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds= inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        pooled_output = outputs[1]#outputs[1]#outputs[0][:,-1,:]#torch.sum(outputs[0], dim = 1)
        #m = nn.Linear(self.emb_size, self.hidden_size)
        #pooled_output = m(pooled_output)

        #print(pooled_output.size())

        #pooled_output = outputs[1]

        #pooled_output = self.dropout(pooled_output)
        #logits = self.classifier(pooled_output)

        #loss = None
        #if labels is not None:
            #if self.num_labels == 1:
                #  We are doing regression
                #loss_fct = MSELoss()
                #loss = loss_fct(logits.view(-1), labels.view(-1))
            #else:
                #loss_fct = CrossEntropyLoss()
                #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        #if not return_dict:
            #output = (logits,) + outputs[2:]
            #return ((loss,) + output) if loss is not None else output

        return pooled_output#outputs.hidden_states[-1],1
            #SequenceClassifierOutput(
            #loss=loss,
            #logits=logits,
            #hidden_states=outputs.hidden_states,
            #attentions=outputs.attentions,
        #)

class DANFeatureExtractor(nn.Module):
    def __init__(self,
                 vocab,
                 num_layers,
                 hidden_size,
                 sum_pooling,
                 dropout,
                 batch_norm=False):
        super(DANFeatureExtractor, self).__init__()
        self.word_emb = vocab.init_embed_layer()

        if sum_pooling:
            self.avg = SummingLayer(self.word_emb)
        else:
            self.avg = AveragingLayer(self.word_emb)
        
        assert num_layers >= 0, 'Invalid layer numbers'
        self.fcnet = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.fcnet.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(vocab.emb_size, hidden_size))
            else:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.fcnet.add_module('f-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.fcnet.add_module('f-relu-{}'.format(i), nn.ReLU())

    def forward(self, input):
        input = (input[0], input[1])
        return self.fcnet(self.avg(input))


class LSTMFeatureExtractor(nn.Module):
    def __init__(self,
                 vocab,
                 num_layers,
                 hidden_size,
                 dropout,
                 bdrnn,
                 attn_type):
        super(LSTMFeatureExtractor, self).__init__()
        self.num_layers = num_layers
        self.bdrnn = bdrnn
        self.attn_type = attn_type
        self.hidden_size = hidden_size//2 if bdrnn else hidden_size
        self.n_cells = self.num_layers*2 if bdrnn else self.num_layers
        
        self.word_emb = vocab.init_embed_layer()
        self.rnn = nn.LSTM(input_size=vocab.emb_size, hidden_size=self.hidden_size,
                num_layers=num_layers, dropout=dropout, bidirectional=bdrnn)
        if attn_type == 'dot':
            self.attn = DotAttentionLayer(hidden_size)

    def forward(self, input):
        data, lengths, attention_mask = input
        lengths_list = lengths.tolist()
        batch_size = len(data)
        embeds = self.word_emb(data)
        packed = pack_padded_sequence(embeds, lengths_list, batch_first=True)
        state_shape = self.n_cells, batch_size, self.hidden_size
        h0 = c0 = embeds.data.new(*state_shape)
        output, (ht, ct) = self.rnn(packed, (h0, c0))

        if self.attn_type == 'last':
            return ht[-1] if not self.bdrnn \
                          else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
        elif self.attn_type == 'avg':
            unpacked_output = pad_packed_sequence(output, batch_first=True)[0]
            return torch.sum(unpacked_output, 1) / lengths.float().view(-1, 1)
        elif self.attn_type == 'dot':
            unpacked_output = pad_packed_sequence(output, batch_first=True)[0]
            return self.attn((unpacked_output, lengths))
        else:
            raise Exception('Please specify valid attention (pooling) mechanism')


class CNNFeatureExtractor(nn.Module):
    def __init__(self,
                 vocab,
                 num_layers,
                 hidden_size,
                 kernel_num,
                 kernel_sizes,
                 dropout):
        super(CNNFeatureExtractor, self).__init__()
        self.word_emb = vocab.init_embed_layer()
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, vocab.emb_size)) for K in kernel_sizes])
        
        assert num_layers >= 0, 'Invalid layer numbers'
        self.fcnet = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.fcnet.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.fcnet.add_module('f-linear-{}'.format(i),
                        nn.Linear(len(kernel_sizes)*kernel_num, hidden_size))
            else:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            self.fcnet.add_module('f-relu-{}'.format(i), nn.ReLU())

    def forward(self, input):
        data, lengths, attention_mask = input
        batch_size = len(data)
        embeds = self.word_emb(data)
        # conv
        embeds = embeds.unsqueeze(1) # batch_size, 1, seq_len, emb_size
        x = [functional.relu(conv(embeds)).squeeze(3) for conv in self.convs]
        x = [functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        # fcnet
        return self.fcnet(x)


class SentimentClassifier(nn.Module):
    def __init__(self,
                 num_layers,
                 hidden_size,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(SentimentClassifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('p-relu-{}'.format(i), nn.ReLU())

        self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))
        self.net.add_module('p-softmax', nn.Softmax())
        #self.net.add_module('p-sigmoid', nn.Sigmoid())
        #self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        return self.net(input)


class LanguageDetector(nn.Module):
    def __init__(self,
                 num_layers,
                 hidden_size,
                 dropout,
                 batch_norm=False):
        super(LanguageDetector, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('q-dropout-{}'.format(i), nn.Dropout(p=dropout))
            self.net.add_module('q-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('q-relu-{}'.format(i), nn.ReLU())

        self.net.add_module('q-linear-final', nn.Linear(hidden_size, 1))

    def forward(self, input):
        #print(self.net())
        return self.net(input)
