import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel

class BertZSL(nn.Module):
    
    def __init__(self, config, num_labels=2):
        super(BertZSL, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_labels = num_labels
        self.bertlabelencoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        # for param in self.bertlabelencoder.parameters():
        #     param.requires_grad = False

        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)

        self.mapping = nn.Linear(config.hidden_size, num_labels)
        self.relations1 = nn.Linear(2*config.hidden_size, 10)
        self.relations2 = nn.Linear(10, 1)

        """
        mode: 
        'max-pooling'
        'self-attentive'
        'self-attentive-mean'
        'h-max-pooling'
        'bissect'
        'normal'

        mode2:
        'gram'
        'dot'
        'dnn'
        'student'
        'zero-shot'
        'normal'

        """
        self.mode = 'normal'
        self.mode2 = 'zero-shot'
        self.pre = False

        print('Surface encoder mode: ', self.mode)
        print('Inference mode: ', self.mode2)
        print('Use pretrained: ', self.pre)

        # Self-attentive
        if self.mode == 'self-attentive':
            self.linear1 = nn.Linear(config.hidden_size, 256)
            self.linear2 = nn.Linear(4*256, config.hidden_size)
            self.tanh = nn.Tanh()
            self.context_vector = nn.Parameter(torch.randn(256, 4), requires_grad=True)

        # Hierarchical
        if self.mode == 'h-max-pooling':
            self.linear = nn.Linear(13*config.hidden_size, config.hidden_size)
        
        # Load pretrained weights
        if self.pre:
            print('Loading pretrained weights...')
            pre_model_dict = torch.load('checkpoints/best_e2e_pretrain.pth')
            model_dict = self.bert.state_dict()
            pre_model_dict = {k:v for k, v in pre_model_dict.items() if k in model_dict and v.size() == model_dict[k].size}
            model_dict.update(pre_model_dict)
            self.bert.load_state_dict(model_dict)

    def forward(self, x_caps, x_masks, y_caps, y_masks, labels):
        """
        BERT outputs:
        last_hidden_states: (b, t, h)
        pooled_output: (b, h), from output of a linear classifier + tanh
        hidden_states: 13 x (b, t, h), embed to last layer embedding
        attentions: 12 x (b, num_heads, t, t)
        """
        # label encoder:
        last_hidden, clusters, hidden, att = self.bertlabelencoder(y_caps, attention_mask=y_masks)
        last_hidden_states, pooled_output, hidden_states, attentions = self.bert(x_caps, attention_mask=x_masks)

        pooled_output = self.transform(last_hidden_states, pooled_output, hidden_states, attentions, x_masks) # (b, h)
        logits = self.multi_learn(pooled_output, clusters, labels)

        return last_hidden_states, pooled_output, logits, clusters
        # loss = self.bert(input_ids, attention_mask=mask, labels=labels)
        # return loss
    
    def transform(self, last_hidden_states, pooled_output, hidden_states, attentions, mask):
        
        if self.mode == 'max-pooling':
            # Method 1: max pooling
            pooled_output, indexes = torch.max(last_hidden_states * mask[:,:,None], dim=1)
        
        elif self.mode == 'self-attentive':
            # Method 2: self-attentive network
            b, _, _ = last_hidden_states.shape
            vectors = self.context_vector.unsqueeze(0).repeat(b, 1, 1)

            h = self.linear1(last_hidden_states) # (b, t, h)
            scores = torch.bmm(h, vectors) # (b, t, 4)
            scores = nn.Softmax(dim=1)(scores) # (b, t, 4)
            outputs = torch.bmm(scores.permute(0, 2, 1), h).view(b, -1) # (b, 4h)
            pooled_output = self.linear2(outputs)
        
        elif self.mode == 'self-attentive-mean':
            # Method 2: self-attentive network
            b, _, _ = last_hidden_states.shape
            vector = torch.mean(last_hidden_states, dim=1).unsqueeze(2)

            #h = self.tanh(self.linear1(last_hidden_states)) # (b, t, h)
            scores = torch.bmm(last_hidden_states, vector) # (b, t, 1)
            scores = nn.Softmax(dim=1)(scores) # (b, t, 1)
            pooled_output = torch.bmm(scores.permute(0, 2, 1), last_hidden_states).squeeze(1) # (b, h)

        elif self.mode == 'h-max-pooling':
            # Method 3: hierarchical max pooling
            b, t, h = last_hidden_states.shape
            N = len(hidden_states)
            final_vectors = torch.zeros(b, h, N).to(self.device)
            for i in range(len(hidden_states)):
                outs, _ = torch.max(hidden_states[i] * mask[:,:,None], dim=1)
                final_vectors[:, :, i] = outs
            final_vectors = final_vectors.view(b, -1)
            pooled_output = self.linear(final_vectors)
        
        elif self.mode == 'bissect':
            # Method 4: bissecting bert
            hidden_states = hidden_states[1:]
            b, t, h = hidden_states[0].shape
            N = len(hidden_states)

            h_states = torch.zeros(b, t, h, N).to(self.device)
            for i in range(N):
                h_states[:, :, :, i] = hidden_states[i]

            final_vectors = torch.zeros(b, t, h).to(self.device)
            # x = (1-torch.eye(N)).unsqueeze(0).repeat(b, 1, 1).to(self.device)

            for i in range(t):
                word_vector = h_states[:, i, :, :] # (b, h, N)
                vector = torch.mean(word_vector, dim=2).unsqueeze(1) # (b, 1, h)

                scores = torch.bmm(vector, word_vector) # (b, 1, N)
                scores = nn.Softmax(dim=2)(scores) # (b, 1, N)
                final_vectors[:, i, :] = torch.bmm(word_vector, scores.permute(0, 2, 1)).squeeze(2) # (b, h)

                # word_vector = word_vector / torch.norm(word_vector, dim=1).unsqueeze(1) # normalize each vector
                # scores = torch.bmm(word_vector.permute(0,2,1), word_vector) # (b, N, N)
                # scores = scores * x # set self attention to be 0s
                # scores = 1 / scores.sum(dim=2)
                # scores = scores / scores.sum(dim=1).unsqueeze(1) # (b, N)
                # scores = scores.unsqueeze(2) # (b, N, 1)
                # final_vectors[:, i, :] = torch.bmm(word_vector, scores).squeeze(2) # (b, h)
                # final_vectors[:, i, :] = hidden_states[-1][:, i, :]
            
            pooled_output, indexes = torch.max(final_vectors * mask[:,:,None], dim=1)
        
        else:
            pooled_output = pooled_output
            
        return pooled_output

    def multi_learn(self, pooled_output, clusters, labels):
    
        if self.mode2 == 'gram':
            gram = torch.mm(clusters, clusters.permute(1,0)) # (n, n)
            weights = torch.mm(pooled_output, clusters.permute(1,0))
            weights = torch.mm(weights, torch.inverse(gram))
            pooled_output = torch.mm(weights, clusters) # (b, h)
        
        elif self.mode2 == 'dot':
            weights = torch.mm(pooled_output, clusters.permute(1,0))
            pooled_output = torch.mm(weights, clusters) # (b, h)

        elif self.mode2 == 'dnn':
            weights = self.mapping(pooled_output) # (b, n)
            weights = nn.Tanh()(weights)
            pooled_output = torch.mm(weights, clusters) # (b, h)
        
        elif self.mode2 == 'student':
            self.alpha = 20.0
            q = 1.0 / (1.0 + (torch.sum(torch.square(pooled_output[:,None,:] - clusters), axis=2) / self.alpha))
            # q = torch.pow(q, (self.alpha + 1.0) / 2.0)
            q = nn.Softmax(dim=1)(q)
            # q = q.transpose(1,0) / torch.sum(q, dim=1)
            # q = q.transpose(1,0)
            pooled_output = torch.mm(q, clusters)
        
        elif self.mode2 == 'zero-shot':
            gram = torch.mm(clusters, clusters.permute(1,0)) # (n, n)
            weights = torch.mm(pooled_output, clusters.permute(1,0))
            weights = torch.mm(weights, torch.inverse(gram)) * np.sqrt(768)
            return weights

            b, h = pooled_output.shape
            query = pooled_output.unsqueeze(1).repeat(1, self.num_labels, 1) # b, n, h
            support = clusters.unsqueeze(0).repeat(b, 1, 1) # b, n, h

            logits = torch.cat([query, support], dim=2)
            logits = nn.ReLU()(self.relations1(logits))
            logits = self.relations2(logits)
            logits = logits.squeeze(2)
            return logits

        else:
            pooled_output = pooled_output
        
        pooled_output_d = self.dropout(pooled_output)
        logits = self.classifier(pooled_output_d)
        # logits = nn.Sigmoid()(logits)

        return logits



