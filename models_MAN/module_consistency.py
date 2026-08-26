import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F

class LearnableGlobalLocalMultiheadAttention(nn.Module):
    NUM_WEIGHTS = 3
    def __init__(
            self, embed_dim, num_heads, dropout=0.):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(self.NUM_WEIGHTS * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(self.NUM_WEIGHTS * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.bias_k = self.bias_v = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)



    def in_proj_global_q(self, query):
        return self._in_proj(query, start=0, end=self.embed_dim)

    def in_proj_global_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_global_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim, end=3 * self.embed_dim)

    # local left
    def in_proj_local_left_q(self, query):
        return self._in_proj(query, start=3 * self.embed_dim, end=4 * self.embed_dim)

    def in_proj_local_left_k(self, key):
        return self._in_proj(key, start=4 * self.embed_dim, end=5 * self.embed_dim)

    # local right
    def in_proj_local_right_q(self, query):
        return self._in_proj(query, start=5 * self.embed_dim, end=6 * self.embed_dim)

    def in_proj_local_right_k(self, key):
        return self._in_proj(key, start=6 * self.embed_dim, end=7 * self.embed_dim)

    # local right
    def in_proj_local_q(self, query):
        return self._in_proj(query, start=7 * self.embed_dim, end=8 * self.embed_dim)

    def in_proj_local_k(self, key):
        return self._in_proj(key, start=8 * self.embed_dim, end=9 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)




    def forward(self, query, key, shape, value):

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        q = self.in_proj_global_q(query)
        k = self.in_proj_global_k(key)
        v = self.in_proj_global_v(value)


        q = q*self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        global_attn_weights = torch.bmm(q, k.transpose(1, 2))

        attn_weights = global_attn_weights

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)



        return attn

