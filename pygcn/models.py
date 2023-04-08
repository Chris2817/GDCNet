import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import numpy as np
import math

from torch.nn import init,LayerNorm

from pygcn.layers import *
from functools import partial
import pdb


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, group, num_layers, input_droprate, hidden_droprate, drop1, drop2, use_bn=False):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(GraphConvolution(nfeat, nhid, group, drop_rate=drop1))
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(nhid))
        for _ in range(num_layers - 2):
            self.convs.append(
                GraphConvolution(nhid, nhid, group, drop_rate=drop2))
            self.bns.append(torch.nn.BatchNorm1d(nhid))
        self.convs.append(GraphConvolution(nhid, nclass, group, drop_rate=drop2))
        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.use_bn = use_bn

    def forward(self,x, adj):

        x = F.dropout(x, p=self.input_droprate, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            
            x = conv(x, adj[i])
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.hidden_droprate, training=self.training)
        x = self.convs[-1](x, adj[-1])

        return x

class DropAttr(nn.Module):
    def __init__(self,nnode, nfeat, nhid, nclass,group, num_layers, input_droprate,
     hidden_droprate, drop1, drop2, sample, num_heads=1, drop=0.6, attn_drop=0.5, 
     drop_path=0.7, mlp_ratio=1., qkv_bias=False, act_layer=nn.GELU, 
     norm_layer=nn.LayerNorm, use_bn=False):
        super(DropAttr, self).__init__()
        self.gcn = GCN(nfeat, nhid, nclass, group, num_layers, input_droprate, hidden_droprate, 
            drop1, drop2, use_bn)
        self.sample = sample
        self.nnode = nnode
        self.nclass = nclass

    def forward(self,x,adj):

        
        output_list = []
        if self.training:
            sample = self.sample
        else:
            sample = 1
        for k in range(sample):
            
            output = self.gcn(x,adj)
            output_list.append(torch.log_softmax(output,dim=-1))

        return output_list





        




