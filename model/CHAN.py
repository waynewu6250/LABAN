import os.path
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss
from torch.nn import CosineEmbeddingLoss

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        self.scores = None

    def attention(self, q, k, v, d_k, mask=None, dropout=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        self.scores = scores
        output = torch.matmul(scores, v)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = self.attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out(concat)
        return output

    def get_scores(self):
        return self.scores

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class Encoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class ContextAttention(nn.Module):
    def __init__(self, device):
        super(ContextAttention, self).__init__()

        self.hidden_dim = 100
        self.attn_head = 4
        self.device = device

        ### Utterance Encoder
        # self.utterance_encoder = BertForUtteranceEncoding.from_pretrained(
        #     os.path.join(args.bert_dir, 'bert-base-uncased.model')
        # )
        # self.bert_output_dim = self.utterance_encoder.config.hidden_size
        # self.hidden_dropout_prob = self.utterance_encoder.config.hidden_dropout_prob
        # if args.fix_utterance_encoder:
        #     for p in self.utterance_encoder.bert.pooler.parameters():
        #         p.requires_grad = False

        ### slot, slot-value Encoder (not trainable)
        # self.sv_encoder = BertForUtteranceEncoding.from_pretrained(
        #         os.path.join(args.bert_dir, 'bert-base-uncased.model'))
        # for p in self.sv_encoder.bert.parameters():
        #     p.requires_grad = False

        # self.slot_lookup = nn.Embedding(self.num_slots, self.bert_output_dim)
        # self.value_lookup = nn.ModuleList([nn.Embedding(num_label, self.bert_output_dim) for num_label in num_labels])

        ### Attention layer
        self.attn = MultiHeadAttention(self.attn_head, 768, dropout=0.)
        self.attn2 = MultiHeadAttention(self.attn_head, 768, dropout=0.)
        self.add_pe = PositionalEncoding(768, 0.)

        ### Belief Tracker
        self.nbt = Encoder(EncoderLayer(768,
                                        MultiHeadAttention(self.attn_head, 768, dropout=0.),
                                        PositionwiseFeedForward(768, self.hidden_dim, 0.),
                                        0.1),
                                        N=6)

    def _make_aux_tensors(self, ids, len):
        token_type_ids = torch.zeros(ids.size(), dtype=torch.long).to(self.device)
        for i in range(len.size(0)):
            for j in range(len.size(1)):
                if len[i,j,0] == 0: # padding
                    break
                elif len[i,j,1] > 0: # escape only text_a case
                    start = len[i,j,0]
                    ending = len[i,j,0] + len[i,j,1]
                    token_type_ids[i, j, start:ending] = 1
        attention_mask = ids > 0
        return token_type_ids, attention_mask

    def forward(self, input_ids, result_masks):

        ds = input_ids.size(0) # dialog size
        ts = input_ids.size(1) # turn size

        hidden = self.add_pe(input_ids)

        # NBT
        turn_mask = torch.Tensor(ds, ts, ts).byte().to(self.device)
        for d in range(ds):
            padding_utter = (result_masks[d,:].sum(-1) != 0)
            turn_mask[d] = padding_utter.unsqueeze(0).repeat(ts,1) & subsequent_mask(ts).to(self.device)
        print(turn_mask[1])
        dafe

        hidden = self.nbt(hidden, turn_mask)
        return hidden