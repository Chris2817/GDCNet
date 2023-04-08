import math

import torch
import numpy as np
import torch.nn as nn
from itertools import repeat
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import collections
import pdb
import scipy.sparse as sp
from pygcn.utils import sparse_mx_to_torch_sparse_tensor
from torch_sparse import matmul


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

to_2tuple = _ntuple(2)

class GraphConvolution(Module):
    def __init__(self, in_features, out_features, group,drop_rate,bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.droprate = drop_rate
        self.dim = in_features
        self.group = group


        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input, adj):
        # pdb.set_trace()
        outputs = torch.zeros_like(input)
        group = self.group
        dims0 = torch.tensor(list(range(self.dim)))
        if self.dim>=self.group:
            remain = self.dim%group
            if remain == 0:
                split_dims0 = list(torch.split(dims0,(self.dim)//group))
            else:
                split_dims0 = list(torch.split(dims0[:-remain],(self.dim)//group))
                split_dims0[-1]=torch.cat((split_dims0[-1],dims0[-remain:]))
        else:
            split_dims0 = list(torch.split(dims0,1))
        
        for i in range(len(split_dims0)):
            tmp = input[:,split_dims0[i]]
            outputs[:,split_dims0[i]] = matmul(adj[i],tmp,reduce="add")
        output = outputs

        output = torch.mm(output, self.weight)
        

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

