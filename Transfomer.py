import torch.nn as nn
from GGANN import GGANN,SpGGANN
from SubLayerConnection import SublayerConnection
from LayerNorm import LayerNorm

class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        #self.Tconv_forward = SpGGANN(hidden,attn_heads)
        self.Tconv_forward = GGANN(hidden,attn_heads)
        self.sublayer4 = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(hidden)

    def forward(self, x, mask, inputP, counter):
        if counter == 0:
          print("============ Start ==============")
          print("##### Transfomer.py ###### x.shape ########", x.shape)
          print('-------')
          print("##### Transfomer.py ###### mask ########", mask.shape)
          print('-------')
          print("##### Transfomer.py ###### inputP.shape ########", inputP.shape)
          print("========= End Transformer ===========")
        x = self.sublayer4(x, lambda _x: self.Tconv_forward.forward(_x, None, inputP))
        x = self.norm(x)
        print("##### Transfomer.py ###### x ########", x.shape)
        print("========= End Transformer ===========")
        return self.dropout(x)
