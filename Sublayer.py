import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k 

    def forward(self, Q, K, V, mask):

        d_k = self.d_k
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / \
                     math.sqrt(d_k) 
        if mask is None:
            mask = torch.zeros_like(attn_score).bool() 
        else:
            mask = mask.unsqueeze(1).repeat(1, Q.size(1), 1, 1)
        attn_score[mask] = -1e9 

        attn_dist = F.softmax(attn_score, dim=-1) 
        output = torch.matmul(attn_dist, V) 
        return output, attn_dist

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, d_k, d_v, is_encoder=True):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads 
        self.d_k = d_k 
        self.d_v = d_v 
        self.multihead_combine = nn.Linear(d_model, d_model, bias = False)
        self.attention = ScaledDotProductAttention(d_k)

    def forward(self, Q, K, V, mask):
        batchSize, seqLen_Q, seqLen_K = Q.size(0), Q.size(1), K.size(1)

        Q = Q.view(batchSize, seqLen_Q, self.n_heads, self.d_k)
        K = K.view(batchSize, seqLen_K, self.n_heads, self.d_k)
        V = V.view(batchSize, seqLen_K, self.n_heads, self.d_v)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)  
        output, attn_dist = self.attention(Q, K, V, mask) 

        output = output.transpose(1, 2).contiguous() 
        output = output.view(batchSize, seqLen_Q, -1)  
        output = self.multihead_combine(output)
        return output, attn_dist