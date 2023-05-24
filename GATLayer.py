from torch import nn
from labml_helpers.module import Module
import torch
from gelu import GELU
from SubLayerConnection import SublayerConnection
class GAT(nn.Module):
    def __init__(self, dmodel, dropout: float = 0.6, is_concat: bool = True,
                 leaky_relu_negative_slope: float = 0.2):
        super().__init__()

        self.is_concat = is_concat
        # self.n_heads = 32

        # Calculate the number of dimensions per head
        self.hiddensize = dmodel

        # Linear layer for initial transformation;
        # i.e. to transform the node embeddings before self-attention
        self.linear = nn.Linear(dmodel, dmodel)
        # self.linearSecond = nn.Linear(dmodel, dmodel)
        # Linear layer to compute attention score $e_{ij}$
        self.attn = nn.Linear(dmodel * 2, 1, bias=False)
        # The activation for attention score $e_{ij}$
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        # Softmax to compute attention $\alpha_{ij}$
        self.softmax = nn.Softmax(dim=1)
        # Dropout layer to be applied for attention
        self.dropout = nn.Dropout(dropout)
        # self.subconnect = SublayerConnection(dmodel, 0.1)
        self.lstm = nn.LSTMCell(dmodel, dmodel)
    
    def forward(self, state, left, inputad):
        if left is not None:
            state = torch.cat([left, state], dim=1)
        #state = torch.cat([left, state], dim=1)
        state = self.linear(state)
        degree2 = inputad
        s = state.size(1)
        # state = self.subconnect(state, lambda _x: self.lstm(torch.bmm(degree2, state).reshape(-1, self.hiddensize), (torch.zeros(_x.reshape(-1, self.hiddensize).size()).cuda(), _x.reshape(-1, self.hiddensize)))[1].reshape(-1, s, self.hiddensize)) #state + torch.matmul(degree2, state)
        # # state = self.linearSecond(state)
        # if left is not None:
        #     state = state[:,left.size(1):,:]

        
        # GAT New Code

        n_nodes = state.shape[0]
        self.n_heads = state.shape[1]
        self.hiddensize = state.shape[2]
        
        g = self.linear(state).view(n_nodes, self.n_heads, self.hiddensize)

        
        g_repeat = g.repeat(n_nodes, 1, 1)
        
        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        
        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        
        g_concat = g_concat.view(n_nodes, n_nodes, self.n_heads, 2 * self.hiddensize)

        e = self.activation(self.attn(g_concat))
        # Remove the last dimension of size `1`
        e = e.squeeze(-1)

        a = self.softmax(e)

        # Apply dropout regularization
        a = self.dropout(a)

        attn_res = torch.einsum('ijh,jhf->ihf', a, g)

        return attn_res.reshape(state.shape)

