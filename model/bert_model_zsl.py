"""Main Zero-shot model pipeline"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from transformers import BertTokenizer, BertModel, AutoModel, AlbertModel

class BertZSL(nn.Module):
    """Main LABAN model"""
    
    def __init__(self, config, opt, num_labels=2):
        super(BertZSL, self).__init__()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.num_labels = num_labels
        self.laban = True

        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        # You can share the utterance and label encoder by removing the following encoder
        self.bertlabelencoder = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, output_attentions=True)
        
        # Uncomment the following line to use TOD-BERT as bert encoder
        # self.tod_bert = AutoModel.from_pretrained("TODBERT/TOD-BERT-JNT-V1", output_hidden_states=True, output_attentions=True)
        # self.tod_bert_label = AutoModel.from_pretrained("TODBERT/TOD-BERT-JNT-V1", output_hidden_states=True, output_attentions=True)

        # Uncomment the following line to use AL-BERT as bert encoder
        # self.albert = AlbertModel.from_pretrained('albert-base-v2', output_hidden_states=True, output_attentions=True)
        # self.albert_label = AlbertModel.from_pretrained('albert-base-v2', output_hidden_states=True, output_attentions=True)

        self.dropout = nn.Dropout(0.1)
        if not self.laban:
            self.classifier = nn.Linear(config.hidden_size, num_labels)
            nn.init.xavier_normal_(self.classifier.weight)
            self.mapping = nn.Linear(config.hidden_size, num_labels)
        self.relations1 = nn.Linear(2*config.hidden_size, 10)
        self.relations2 = nn.Linear(10, 1)

        """
        surface_encoder_mode: 
        'max-pooling'
        'self-attentive'
        'self-attentive-mean'
        'h-max-pooling'
        'bissect'
        'normal'

        label_aware_mode:
        'gram'
        'dot'
        'dnn'
        'student'
        'zero-shot'
        'normal'
        """
        self.surface_encoder_mode = 'normal'
        self.label_aware_mode = 'zero-shot'

        print('Surface encoder mode: ', self.surface_encoder_mode)
        print('Label aware layer mode: ', self.label_aware_mode)

        # Self-attentive
        if self.surface_encoder_mode == 'self-attentive':
            self.linear1 = nn.Linear(config.hidden_size, 256)
            self.linear2 = nn.Linear(4*256, config.hidden_size)
            self.tanh = nn.Tanh()
            self.context_vector = nn.Parameter(torch.randn(256, 4), requires_grad=True)

        # Hierarchical
        if self.surface_encoder_mode == 'h-max-pooling':
            self.linear = nn.Linear(13*config.hidden_size, config.hidden_size)
        
        # Only for running CDSSM-BERT baseline
        if opt.run_baseline == 'cdssmbert':
            self.emb_len = 768
            self.st_len = opt.maxlen
            self.K = 1000 # dimension of Convolutional Layer: lc
            self.L = 768 # dimension of semantic layer: y 
            self.batch_size = opt.batch_size
            self.kernal = 3
            self.conv = nn.Conv1d(self.emb_len, self.K, self.kernal)
            self.linear = nn.Linear(self.K, self.L, bias = False) 
            self.max = nn.MaxPool1d(opt.maxlen-2)
            self.in_conv = nn.Conv1d(self.emb_len, self.K, self.kernal)
            self.in_max = nn.MaxPool1d(8)
            self.in_linear = nn.Linear(self.K, self.L,bias = False)
        
        self.run_baseline = opt.run_baseline

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
        
        if self.run_baseline == 'zsbert':
            # baseline1: dot product (zero-shot BERT)
            logits = torch.mm(pooled_output, clusters.transpose(1,0))
        
        elif self.run_baseline == 'cdssmbert':
            # baseline2: cdssm BERT
            utter = last_hidden_states
            intents = last_hidden
            intents = intents.repeat(x_caps.shape[0],1,1,1) # (b,n,ti,h)

            utter = utter.transpose(1,2) # (b,h,t)
            utter_conv = torch.tanh(self.conv(utter))  # (b, nh, t-)
            utter_conv_max = self.max(utter_conv) # (b, nh, 1)
            utter_conv_max_linear = torch.tanh(self.linear(utter_conv_max.permute(0,2,1))) # (b, 1, h)
            utter_conv_max_linear = utter_conv_max_linear.transpose(1,2) # (b, h, 1)

            intents = intents.permute(0,3,2,1) # (b,h,ti,n)
            class_num = list(intents.shape)
            
            int_convs = [torch.tanh(self.in_conv(intents[:,:,:,i])) for i in range(class_num[3])]  # for every intent (b,nh,ti-)
            int_convs = [self.in_max(int_convs[i]) for i in range(class_num[3])]  # for every intent (b,nh,1)
            int_conv_linear = [torch.tanh(self.in_linear(int_conv.permute(0,2,1))) for int_conv in int_convs] # for every intent (b,1,h)
        
            sim = [torch.bmm(yi, utter_conv_max_linear) for yi in int_conv_linear]
            sim = torch.stack(sim) # (n,b)
            logits = sim.transpose(0,1).squeeze(2).squeeze(2) # (b,n)
        
        else:
            # LABAN
            logits = self.multi_learn(pooled_output, clusters, labels)


        return last_hidden_states, pooled_output, logits, clusters
    
    def transform(self, last_hidden_states, pooled_output, hidden_states, attentions, mask):
        """Fuse the token-level hidden states into sentence representation."""
        
        if self.surface_encoder_mode == 'max-pooling':
            # Method 1: max pooling
            pooled_output, indexes = torch.max(last_hidden_states * mask[:,:,None], dim=1)
        
        elif self.surface_encoder_mode == 'self-attentive':
            # Method 2: self-attentive network
            b, _, _ = last_hidden_states.shape
            vectors = self.context_vector.unsqueeze(0).repeat(b, 1, 1)

            h = self.linear1(last_hidden_states) # (b, t, h)
            scores = torch.bmm(h, vectors) # (b, t, 4)
            scores = nn.Softmax(dim=1)(scores) # (b, t, 4)
            outputs = torch.bmm(scores.permute(0, 2, 1), h).view(b, -1) # (b, 4h)
            pooled_output = self.linear2(outputs)
        
        elif self.surface_encoder_mode == 'self-attentive-mean':
            # Method 3: self-attentive network
            b, _, _ = last_hidden_states.shape
            vector = torch.mean(last_hidden_states, dim=1).unsqueeze(2)

            #h = self.tanh(self.linear1(last_hidden_states)) # (b, t, h)
            scores = torch.bmm(last_hidden_states, vector) # (b, t, 1)
            scores = nn.Softmax(dim=1)(scores) # (b, t, 1)
            pooled_output = torch.bmm(scores.permute(0, 2, 1), last_hidden_states).squeeze(1) # (b, h)

        elif self.surface_encoder_mode == 'h-max-pooling':
            # Method 4: hierarchical max pooling
            b, t, h = last_hidden_states.shape
            N = len(hidden_states)
            final_vectors = torch.zeros(b, h, N).to(self.device)
            for i in range(len(hidden_states)):
                outs, _ = torch.max(hidden_states[i] * mask[:,:,None], dim=1)
                final_vectors[:, :, i] = outs
            final_vectors = final_vectors.view(b, -1)
            pooled_output = self.linear(final_vectors)
        
        elif self.surface_encoder_mode == 'bissect':
            # Method 5: bissecting bert
            hidden_states = hidden_states[1:]
            b, t, h = hidden_states[0].shape
            N = len(hidden_states)

            h_states = torch.zeros(b, t, h, N).to(self.device)
            for i in range(N):
                h_states[:, :, :, i] = hidden_states[i]

            final_vectors = torch.zeros(b, t, h).to(self.device)

            for i in range(t):
                word_vector = h_states[:, i, :, :] # (b, h, N)
                vector = torch.mean(word_vector, dim=2).unsqueeze(1) # (b, 1, h)

                scores = torch.bmm(vector, word_vector) # (b, 1, N)
                scores = nn.Softmax(dim=2)(scores) # (b, 1, N)
                final_vectors[:, i, :] = torch.bmm(word_vector, scores.permute(0, 2, 1)).squeeze(2) # (b, h)
            
            pooled_output, indexes = torch.max(final_vectors * mask[:,:,None], dim=1)
        
        else:
            # Baseline: Use [CLS] head
            pooled_output = pooled_output
            
        return pooled_output

    def multi_learn(self, pooled_output, clusters, labels):
        """Interact with the label embeddings."""
    
        if self.label_aware_mode == 'gram':
            gram = torch.mm(clusters, clusters.permute(1,0)) # (n, n)
            weights = torch.mm(pooled_output, clusters.permute(1,0))
            weights = torch.mm(weights, torch.inverse(gram))
            pooled_output = torch.mm(weights, clusters) # (b, h)
        
        elif self.label_aware_mode == 'dot':
            weights = torch.mm(pooled_output, clusters.permute(1,0))
            pooled_output = torch.mm(weights, clusters) # (b, h)

        elif self.label_aware_mode == 'dnn':
            weights = self.mapping(pooled_output) # (b, n)
            weights = nn.Tanh()(weights)
            pooled_output = torch.mm(weights, clusters) # (b, h)
        
        elif self.label_aware_mode == 'student':
            self.alpha = 20.0
            q = 1.0 / (1.0 + (torch.sum(torch.square(pooled_output[:,None,:] - clusters), axis=2) / self.alpha))
            # q = torch.pow(q, (self.alpha + 1.0) / 2.0)
            q = nn.Softmax(dim=1)(q)
            # q = q.transpose(1,0) / torch.sum(q, dim=1)
            # q = q.transpose(1,0)
            pooled_output = torch.mm(q, clusters)
        
        elif self.label_aware_mode == 'zero-shot':
            gram = torch.mm(clusters, clusters.permute(1,0)) # (n, n)
            weights = torch.mm(pooled_output, clusters.permute(1,0))
            weights = torch.mm(weights, torch.inverse(gram)) * np.sqrt(768)
            return weights

        else:
            pooled_output = pooled_output
        
        pooled_output_d = self.dropout(pooled_output)
        logits = self.classifier(pooled_output_d)

        return logits



