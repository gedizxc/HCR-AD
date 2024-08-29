# @Time    : 2022/11/21 2:10 下午
# @Author  : tang
# @File    : Hetero_Gconv.py
import torch
import torch.nn.functional as F

from torch_geometric.nn import SAGEConv


# 异构的gnn模型
class GConv(torch.nn.Module):

    def __init__(self, hidden_channels, out_channels):
        super(GConv, self).__init__()
        # 两层gcn
        self.conv = SAGEConv((128, 128), hidden_channels)
        self.conv1 = SAGEConv((hidden_channels, hidden_channels), out_channels)

    def forward(self, x_dict, edge_index_dict): #dict2,dict4
        # 数据流
        x = self.conv(x_dict, edge_index_dict)
        x = F.relu(x)
        x = self.conv1(x, edge_index_dict)
        x = F.relu(x)
        return x
