# @Time    : 2022/11/21 1:28 下午
# @Author  : tang
# @File    : Swat_Graph_metadata.py
import numpy as np
import torch
from torch_geometric import transforms as T
from itertools import chain
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data, HeteroData



#获取异构图的metadata数据结构
def Swat_heterograph_data():
    # A原始编号
    A = [
        0, 1, 5, 6, 7, 8, 16, 17, 18, 25, 26, 27, 34, 35, 36, 37, 38, 39, 40, 41,
        44, 45, 46, 47
    ]
    # 异构图的metadata
    metadata = (['A', 'B'], [('A', 'link', 'A'), ('A', 'link', 'B'),
                             ('B', 'link', 'B'), ('B', 'link', 'A')])  #

    # 异构图的包含元路径BAB, BAA metadata
    metadata_path = (
        ['A', 'B'],
        [
            ('B', 'metapath_0', 'B'),
            ('B', 'metapath_1', 'A'),
            ('A', 'metapath_2', 'B'),
        ],
    )
    # A_list 图中所有A类的编号
    A = np.array(A)
    A_list = [A + i * 51 for i in range(128)]
    A_list = list(chain(*A_list))
    # B_list 图中所有B类的编号
    B_list = [i for i in range(51 * 128) if i not in A_list]

    A_Class = torch.Tensor(A_list).to(torch.long)
    B_Class = torch.Tensor(B_list).to(torch.long)

    # 加入元路径的转换
    metapaths = [
        [('B', 'A'), ('A', 'B')],
        [('B', 'A'), ('A', 'A')],
        [('A', 'A'), ('A', 'B')],
    ]
    transform = T.AddMetaPaths(metapaths=metapaths,
                               drop_orig_edges=True,
                               drop_unconnected_nodes=True)

    return metadata, metadata_path,transform,A_Class,B_Class

# 获取异构图具体数据
def get_heter_data(x, edge_index,A_Class,B_Class):
    # 获取对应A, B的feat
    data_a = x[A_Class]
    data_b = x[B_Class]
    adj = to_dense_adj(edge_index).squeeze()
    # 获取AA, AB, BA, BB的edge
    A_A_edge = adj[A_Class][:, A_Class]
    A_B_edge = adj[A_Class][:, B_Class]
    B_B_edge = adj[B_Class][:, B_Class]
    B_A_edge = adj[B_Class][:, A_Class]
    data_hetero = HeteroData()
    data_hetero['A'].x = data_a
    data_hetero['B'].x = data_b
    # 将edge转换成edge_index
    data_hetero['A', 'link',
                'A'].edge_index = A_A_edge.nonzero().t().contiguous()
    data_hetero['A', 'link',
                'B'].edge_index = A_B_edge.nonzero().t().contiguous()
    data_hetero['B', 'link',
                'B'].edge_index = B_B_edge.nonzero().t().contiguous()
    data_hetero['B', 'link',
                'A'].edge_index = B_A_edge.nonzero().t().contiguous()
    return data_hetero