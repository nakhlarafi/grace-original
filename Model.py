import torch.nn as nn
import torch.nn.functional as F
import torch
from Transfomer import TransformerBlock
from postionEmbedding import PositionalEmbedding
from LayerNorm import LayerNorm
from SubLayerConnection import *
import numpy as np


class NlEncoder( nn.Module ):
    def __init__(self, args):
        super( NlEncoder, self ).__init__()
        self.embedding_size = args.embedding_size
        self.nl_len = args.NlLen
        self.word_len = args.WoLen
        self.char_embedding = nn.Embedding( args.Vocsize, self.embedding_size )
        self.feed_forward_hidden = 4 * self.embedding_size
        self.conv = nn.Conv2d( self.embedding_size, self.embedding_size, (1, self.word_len) )
        self.transformerBlocks = nn.ModuleList(
            [TransformerBlock( self.embedding_size, 8, self.feed_forward_hidden, 0.1 ) for _ in range(3)] )
        self.token_embedding = nn.Embedding( args.Nl_Vocsize, self.embedding_size - 1 )
        self.token_embedding1 = nn.Embedding( args.Nl_Vocsize, self.embedding_size  -2 )

        self.text_embedding = nn.Embedding( 20, self.embedding_size )
        
        self.resLinear = nn.Linear( self.embedding_size, 2 )
        self.pos = PositionalEmbedding( self.embedding_size )
        self.loss = nn.CrossEntropyLoss()
        self.norm = LayerNorm( self.embedding_size )
        self.lstm = nn.LSTM( self.embedding_size // 2, int( self.embedding_size / 4 ), batch_first=True,
                             bidirectional=True )
        self.conv = nn.Conv2d( self.embedding_size, self.embedding_size, (1, 10) )
        self.resLinear2 = nn.Linear( self.embedding_size, 1 )

    def forward(self, input_node, inputtype, inputad, res, inputtext, linenode, linetype, linemus, modification, churn):
        nlmask = torch.gt(input_node, 0)
        resmask = torch.eq(input_node, 2)
        inputad = inputad.float()

        # linemus_norm = linemus.float() / torch.max(linemus)
        # modification = modification.float() / torch.max(modification) # Normalize linetype
        # print('Modification: ', modification.shape)
        # print('Inputtext: ', inputtext.shape)
        nodeem = self.token_embedding(input_node)
        nodeem = torch.cat([nodeem, inputtext.unsqueeze(-1).float()], dim=-1)
        x = nodeem

        lineem = self.token_embedding1(linenode)
        # print('Lineem: ', lineem.shape)
        # print('Nodeem: ', nodeem.shape)
        # print('modification: ', modification.shape)
        # print('linenode: ', linenode.shape)
        # lineem = torch.cat([lineem, linemus_norm.unsqueeze(-1).float(), linetype_norm.unsqueeze(-1).float()], dim=-1)  # include linetype_norm
        lineem = torch.cat([lineem, modification.unsqueeze(-1).float(), churn.unsqueeze(-1).float()], dim=-1)
        x = torch.cat([x, lineem], dim=1)
        for trans in self.transformerBlocks:
            x = trans.forward(x, nlmask, inputad)

        x = x[:, :input_node.size(1)]
        resSoftmax = F.softmax(self.resLinear2(x).squeeze(-1).masked_fill(resmask == 0, -1e9), dim=-1)
        loss = -torch.log(resSoftmax.clamp(min=1e-10, max=1)) * res
        loss = loss.sum(dim=-1)

        return loss, resSoftmax, x
