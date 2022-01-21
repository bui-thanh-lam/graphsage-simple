import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import random
from src import utils

class MeanAggregator(nn.Module):

    def __init__(self, cuda=False, gcn=False, num_sample=10, adj_lists=None):
        super().__init__()

        self.cuda = cuda
        self.gcn = gcn
        self.num_sample = num_sample
        self.adj_lists = adj_lists

    def forward(self, features, nodes):
        num_sample = self.num_sample
        self.features = features
        to_neighs = [self.adj_lists[int(node)] for node in nodes]
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh,
                                        num_sample,
                                        )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        if self.gcn:
            for i, samp_neigh in enumerate(samp_neighs):
                samp_neigh.add(nodes[i])
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        mask = nn.Parameter(torch.zeros(len(samp_neighs), len(unique_nodes)), requires_grad=False)
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        # if self.cuda:
        #     mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)
        # if self.cuda:
        #     embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        # else:
        embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
        to_feats = mask.mm(embed_matrix)
        return to_feats

class Encoder(nn.Module):
    def __init__(self, features, feature_dim,
            embed_dim, gcn=False, cuda=False):
        super().__init__()

        self.features = features
        self.feat_dim = feature_dim

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, embeds, nodes):

        if not self.gcn:
            # if self.cuda:
            #     self_feats = self.features(torch.LongTensor(nodes).cuda())
            # else:
            self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, embeds], dim=1)
        else:
            combined = embeds
        combined = F.relu(self.weight.mm(combined.t()))
        return combined

class SupervisedGraphSage(nn.Module):

    def __init__(self, **args):
        super().__init__()
        
        self.args = args
        num_layers = self.args['num_layers']
        feat_dim = self.args['n_features']
        embed_dim = self.args['embed_dim']
        cuda = self.args['cuda']
        num_sample = self.args['num_sample']
        num_classes = self.args['num_classes']
        
        self.adj_lists = self.args['adj_lists']
        self.features = self.args['features']

        #aggregators
        self.aggregators = nn.ModuleList([MeanAggregator(gcn=True, cuda=cuda, num_sample=num_sample, adj_lists=self.adj_lists) for _ in range(num_layers)])

        #fully connected layers
        self.fcs = nn.ModuleList([Encoder(features=self.features, feature_dim=feat_dim, embed_dim=embed_dim, cuda=cuda, gcn=True)])
        for layer in range(self.args['num_layers'] - 1):
            self.fcs.append(Encoder(features=self.features, feature_dim=embed_dim, embed_dim=embed_dim, cuda=cuda, gcn=True))

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        #nodes = nodes.numpy().tolist()
        embeds = self.aggregators[0](self.features, nodes)
        embeds = self.fcs[0](embeds, nodes)
        for layer in range(1, self.args['num_layers'] - 1):
            embeds = self.aggregators[layer](embeds, nodes)
            embeds = self.fcs[layer](embeds, nodes)

        scores = self.weight.mm(embeds)
        return F.sigmoid(scores.t()).squeeze()
