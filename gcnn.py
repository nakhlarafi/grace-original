
from torch import nn
import torch
from gelu import GELU
from SubLayerConnection import SublayerConnection
class GCNN(nn.Module):
    def __init__(self, dmodel, liner_fourth=2000):
        super(GCNN ,self).__init__()
        self.hiddensize = dmodel
        self.linear = nn.Linear(dmodel, dmodel)
        self.linearSecond = nn.Linear(dmodel, dmodel)
        self.linearThird = nn.Linear(dmodel, dmodel)

        self.activate = GELU()
        self.dropout = nn.Dropout(p=0.1)
        self.subconnect = SublayerConnection(dmodel, 0.1)
        self.lstm = nn.LSTMCell(dmodel, dmodel)


        self.is_concat = True
        self.n_hidden = self.hiddensize
        leaky_relu_negative_slope = 0.2

        self.attn = nn.Linear(self.n_hidden * 2, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=1)
        self.linearFourth = nn.Linear(liner_fourth, 32)
        
    def forward(self, state, left, inputad):
        # print('####  Start GCNN.py #####')
        # print("##### GCNN.py ###### state.shape ########", state.shape)
        # print('-------')
        # print("##### GCNN.py ###### left ########", left)
        # print('-------')
        # print("##### GCNN.py ###### inputad ########", inputad.shape)
        # print('-------')
        # print("##### GCNN.py ###### hiddensize ########", self.hiddensize)
        # print("=========== END GCNN.py ===========")
        if left is not None:
            state = torch.cat([left, state], dim=1)
        #state = torch.cat([left, state], dim=1)
        # state = self.linear(state)
        # degree2 = inputad
        # s = state.size(1)

        # print("##### GCNN.py ###### state shape before SubConnect ########", state.shape)
        # state = self.subconnect(state, lambda _x: self.lstm(torch.bmm(degree2, state).reshape(-1, self.hiddensize), (torch.zeros(_x.reshape(-1, self.hiddensize).size()).cuda(), _x.reshape(-1, self.hiddensize)))[1].reshape(-1, s, self.hiddensize)) #state + torch.matmul(degree2, state)
        
        # print("##### GCNN.py ###### state shape after SubConnect ########", state.shape)
        # # lambda _x: self.lstm(torch.bmm(degree2, state).reshape(-1, self.hiddensize), (torch.zeros(_x.reshape(-1, self.hiddensize).size()).cuda(), _x.reshape(-1, self.hiddensize)))[1].reshape(-1, s, self.hiddensize)
        
        
        # state = self.linearSecond(state)
        # if left is not None:
        #     state = state[:,left.size(1):,:]

        # ============== GAT CODE ================
        
        self.n_heads = state.shape[-1]
        n_nodes = state.shape[1]
        front = state.shape[0]
        

        # ============= Forward Code ==============
        # print("##### GCNN.py ###### state.shape ########", state.shape)
        # g = self.linearThird(state).view(n_nodes, self.n_heads, self.n_hidden)
        g = self.linearThird(state).view(front, n_nodes, self.n_hidden)
        # g -> torch.Size([1, 580, 32])
        g_repeat = g.repeat(n_nodes, 1, 1)
        # g_repeat -> torch.Size([580, 580, 32])

        g_repeat_interleave = g.repeat_interleave(n_nodes, dim=0)
        # g_repeat_interleave -> torch.Size([580, 580, 32])

        g_concat = torch.cat([g_repeat_interleave, g_repeat], dim=-1)
        # g_concat -> torch.Size([580, 580, 64])

        # g_concat = g_concat.view(n_nodes, n_nodes, 2 * self.n_hidden)

        e = self.activation(self.attn(g_concat))
        # e -> torch.Size([580, 580, 32])

        e_mat = e.view(front, n_nodes, n_nodes)
        # e_mat -> torch.Size([32, 580, 580])

        dense_inputad = inputad.to_dense()

        e_masked = e_mat.masked_fill(dense_inputad == 0, 0)
        # e_masked -> torch.Size([32, 580, 580])

        state = self.activation(self.linearFourth(e_masked))
        state = state.reshape(front, n_nodes, 32)

        a = self.softmax(e)

        # Apply dropout regularization
        a = self.dropout(a)

        # state -> torch.Size([1, 580, 32])

        # e = e.squeeze(-1)

        a = self.softmax(state)

        # Apply dropout regularization
        a = self.dropout(a)

        if list(a.shape) == list(g.shape):
          attn_res = a * g
        else:
          attn_res = torch.einsum('ijh,jhf->ihf', a.transpose(1,2), g)

        state = attn_res.reshape(front, n_nodes, 32)

        # ================= END ===================

        state = self.linear(state)
        degree2 = inputad
        s = state.size(1)

        #print("##### GCNN.py ###### state shape before SubConnect ########", state.shape)
        state = self.subconnect(state, lambda _x: self.lstm(torch.bmm(degree2, state).reshape(-1, self.hiddensize), (torch.zeros(_x.reshape(-1, self.hiddensize).size()).cuda(), _x.reshape(-1, self.hiddensize)))[1].reshape(-1, s, self.hiddensize)) #state + torch.matmul(degree2, state)
        
        #print("##### GCNN.py ###### state shape after SubConnect ########", state.shape)
        # lambda _x: self.lstm(torch.bmm(degree2, state).reshape(-1, self.hiddensize), (torch.zeros(_x.reshape(-1, self.hiddensize).size()).cuda(), _x.reshape(-1, self.hiddensize)))[1].reshape(-1, s, self.hiddensize)
        
        
        state = self.linearSecond(state)

        
        #print("##### GCNN.py ###### return state ########", state.shape)
        #print("=========== END GCNN.py ===========")
        return state#self.dropout(state)[:,50:,:]

