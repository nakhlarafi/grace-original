import torch
from fast_gat import GraphAttentionNetwork

depth = 3
heads = 3
input_dim = 3
inner_dim = 2

net = GraphAttentionNetwork(depth, heads, input_dim, inner_dim)