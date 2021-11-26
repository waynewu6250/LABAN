# -*- coding: utf-8 -*-
# Copyright 2019 The Hong Kong Polytechnic University (Xuandi Fu)
#
"""CDSSM baseline"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

class CDSSM(nn.Module):
    def __init__(self, opt, device):
        super(CDSSM, self).__init__()
        
        self.emb_len = 300
        self.st_len = opt.maxlen
        self.K = 1000 # dimension of Convolutional Layer: lc
        self.L = 300 # dimension of semantic layer: y 
        self.batch_size = opt.batch_size
        self.kernal = 3
        self.conv = nn.Conv1d(self.emb_len, self.K, self.kernal)
        self.linear = nn.Linear(self.K, self.L, bias = False) 
        self.max = nn.MaxPool1d(opt.maxlen-2)
        self.cossim = nn.CosineSimilarity(eps=1e-6)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.5)
        
        self.in_conv = nn.Conv1d(self.emb_len, self.K, self.kernal)
        self.in_max = nn.MaxPool1d(8)
        self.in_linear = nn.Linear(self.K, self.L,bias = False) 

        # w2v = KeyedVectors.load_word2vec_format('data/wiki.en.vec', binary=False, limit=500000)
        # w2v.save('vectors.kv')
        w2v = KeyedVectors.load('vectors.kv')
        self.pre_embedding = torch.FloatTensor(w2v.get_normed_vectors()).to(device)
        self.pre_embedding = torch.cat([self.pre_embedding, torch.randn(1, 300).to(device)], dim=0)
        self.word_embedding = nn.Embedding.from_pretrained(self.pre_embedding)
        self.device = device
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        
    def forward(self, captions_t, masks, intent_tokens, mask_tokens, labels):
        #print("forward")
        
        # 1. embedding
        utter = self.word_embedding(captions_t)      # (b,t,h)
        intents = self.word_embedding(intent_tokens) # (n,ti,h)
        # intents = torch.sum(intents, dim=1)
        intents = intents.repeat(captions_t.shape[0],1,1,1) # (b,n,ti,h)

        # 2. utter conv
        utter = utter.transpose(1,2) # (b,h,t)
        utter_conv = torch.tanh(self.conv(utter))  # (b, nh, t-)
        utter_conv_max = self.max(utter_conv) # (b, nh, 1)
        utter_conv_max_linear = torch.tanh(self.linear(utter_conv_max.permute(0,2,1))) # (b, 1, h)
        utter_conv_max_linear = utter_conv_max_linear.transpose(1,2) # (b, h, 1)

        # 3. intent conv
        intents = intents.permute(0,3,2,1) # (b,h,ti,n)
        class_num = list(intents.shape)
        
        int_convs = [torch.tanh(self.in_conv(intents[:,:,:,i])) for i in range(class_num[3])]  # for every intent (b,nh,ti-)
        int_convs = [self.in_max(int_convs[i]) for i in range(class_num[3])]  # for every intent (b,nh,1)
        int_conv_linear = [torch.tanh(self.in_linear(int_conv.permute(0,2,1))) for int_conv in int_convs] # for every intent (b,1,h)
       
        # ==== compute cossim
        sim = [torch.bmm(yi, utter_conv_max_linear) for yi in int_conv_linear]
        sim = torch.stack(sim) # (n,b)
        y_pred = sim.transpose(0,1).squeeze(2).squeeze(2) # (b,n)
        
        return int_convs, sim, y_pred
  
    def loss(self, y_pred, y_true): #y_pred result y: target intent
        loss = self.criterion(y_pred, y_true)
        return loss