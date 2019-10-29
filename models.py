import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout
        self.L = nn.Linear(nclass, nclass)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj[0]))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.L(self.gc2(x, adj[1]))
        return F.log_softmax(x, dim=1)

class Mix(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(Mix, self).__init__()
        self.GCN = GCN(nfeat[0], nhid, nhid, dropout)
        self.GIN = GCN(nfeat[1], nhid, nhid, dropout)
        self.L = nn.Linear(nhid*2, nclass)

    def forward(self, x, adj):
        x1 = self.GCN(x[0], adj[0])
        x2 = self.GIN(x[1], adj[1])
        x = self.L(torch.cat((x1,x2), dim=1))
        return F.log_softmax(x, dim=1)
