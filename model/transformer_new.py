"""Transformer module with masks"""
import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism.
    """

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, cross1, cross2, scale=None, attn_mask=None):
        """
        Args:
        	q: Queries [B, L_q, D_q]
        	k: Keys [B, L_k, D_k]
        	v: Values [B, L_v, D_v]
        	scale: 1/sqrt(dk)
        	attn_mask: [B, L_q, L_k]
        Returns:
        	context, attention
        """
        attention = torch.bmm(q, k.transpose(1, 2))

        if scale:
        	attention = attention * scale

        attention = attention.masked_fill_(attn_mask, -np.inf)
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        context = torch.bmm(attention, v)

        attention_score = torch.bmm(cross1, cross2.transpose(1,2))

        return context, attention_score

class Transformer(nn.Module):
    """Transformer module.
    """

    def __init__(self, hidden_dim, model_dim=512, num_heads=8, dropout=0.0):
        super(Transformer, self).__init__()

        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(hidden_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(hidden_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(hidden_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        self.layer_norm = nn.LayerNorm(model_dim)

        # cross attention mechanism
        self.embed_k = nn.Linear(hidden_dim, 200)
        self.embed_q = nn.Linear(hidden_dim, 200)

    def forward(self, key, value, query, masks=None):
        
        # Padding mask: Input size: (B, T)
        len_q = masks.size(1)
        pad_mask = masks.eq(0)
        attn_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)
        
        attn_mask1 = masks.unsqueeze(1).expand(-1, len_q, -1)
        attn_mask2 = masks.unsqueeze(2).expand(-1, -1, len_q)
        attn_mask3 = (attn_mask1*attn_mask2).eq(0)

        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # cross attention
        cross1 = self.embed_k(key)
        cross2 = self.embed_q(query)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
          query, key, value, cross1, cross2, scale, attn_mask)

        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        output = torch.cat([residual, context], dim=2)

        # average attention over head
        # attention = attention.view(batch_size, num_heads, len_q, len_q)
        # attention = torch.mean(attention, dim=1)
        attention = attention.masked_fill(attn_mask3, 0.)
        attention = nn.Softmax(dim=2)(attention)
        #print(attn_mask3[0])
        # attention = attention*attn_mask3

        return output, attention