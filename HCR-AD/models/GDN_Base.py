# @Time    : 2022/11/21 1:57 下午
# @Author  : tang
# @File    : GDN_Base.py


# @Time    : 2022/9/28 9:29 上午
# @Author  : tang
# @File    : GDN_Grace.py
import math
import numpy as np
import torch
import torch.nn as nn
import time

import torch.nn as nn
class GDN_Model(nn.Module):
    def __init__(self,edge_index_sets, node_num,dim,topk):
        super(GDN_Model,self).__init__()
        self.edge_index_sets = edge_index_sets

        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(edge_index_sets)
        self.topk = topk

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None
        self.Linear = nn.Linear(15,64)

        self.init_params()


    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index):     #128x51x15, 128x2x2550
        x = data.clone().detach()
        edge_index_sets = self.edge_index_sets

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()  #6528 x 15

        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num)

            batch_edge_index = self.cache_edge_index_sets[i]

            all_embeddings = self.embedding(torch.arange(node_num))
            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            weights = weights_arr.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
            cos_ji_mat = cos_ji_mat / normed_mat

            dim = weights.shape[-1]
            topk_num = self.topk

            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

            self.learned_graph = topk_indices_ji


            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num)
        input_embedding = self.Linear(x)
        sensor_embedding = all_embeddings
        graph_x = torch.cat((input_embedding,sensor_embedding),dim = 1)
        return graph_x, batch_gated_edge_index





def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1,batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i*edge_num:(i+1)*edge_num] += i*node_num

    return batch_edge_index.long()

