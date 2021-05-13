import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel

class ZSLSTM(nn.Module):
    
    def __init__(self, opt, num_labels=2):
        super(ZSLSTM, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        self.embedding_sentence = nn.Embedding(len(self.tokenizer.vocab), 64)
        self.embedding_label = nn.Embedding(len(self.tokenizer.vocab), 64)
        self.rnn_sentence = nn.LSTM(input_size=64, 
                           hidden_size=64,
                           bidirectional=True,
                           batch_first=True, 
                           num_layers=1)
        self.rnn_label = nn.LSTM(input_size=64, 
                           hidden_size=64,
                           bidirectional=True,
                           batch_first=True, 
                           num_layers=1)
        
        #self.dropout = nn.Dropout(0.1)
        self.num_labels = num_labels
        
        self.opt = opt

    def forward(self, x_caps, x_masks, y_caps, y_masks, labels):
        
        # Encoder
        X = self.embedding_sentence(x_caps)
        rnn_out_x, _ = self.rnn_sentence(X)
        rnn_out_x = rnn_out_x[:,-1,:]

        Y = self.embedding_label(y_caps)
        rnn_out_y, _ = self.rnn_sentence(Y)
        rnn_out_y = rnn_out_y[:,-1,:]

        logits = torch.mm(rnn_out_x, rnn_out_y.transpose(1,0))

        return rnn_out_x, rnn_out_y, logits
     