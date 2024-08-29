# @Time    : 2022/11/21 1:38 下午
# @Author  : tang
# @File    : STGAD.py
import torch
import torch.nn as nn

from models.GDN_Base import GDN_Model
from models.Hetero_Gconv import *
from torch_geometric.nn import to_hetero
from util import drop_edge, drop_feature, calc_infoNCE, calc_infoNCE_metapath
from datasets.Swat_Graph_metadata import *
from util.meta_path import get_a_id
import random


class STGAD_Model(nn.Module):
    def __init__(self,edge_index_sets, feature_len, dim, topk,
                 hidden_channels, out_channels, metadata, metadata_path,mask_len,num_layer,d_model,A_Class,B_Class,transform,dropout=0):
        super().__init__()
        self.A_Class = A_Class
        self.B_Class = B_Class
        self.transform1 = transform
        #利用Vi获取同质图的图数据
        self.GDN_model = GDN_Model(edge_index_sets,feature_len,dim,topk)

        #transformer
        self.src_mask = None
        self.src_key_padding = None

        self.embedding = nn.Linear(feature_len, d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=16, dropout=0.1, batch_first=True)
        self.encoder_trans = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layer)

        self.decoder = nn.Linear(15, mask_len)
        self.Linear1 = nn.Linear(d_model, feature_len)

        self.init_weight()

        # 节点级别图对比学习
        self.encoder = GConv(hidden_channels, out_channels)
        self.encoder = to_hetero(self.encoder, metadata, aggr="sum")

        # 元路径级别的对比学习
        self.encoder1 = GConv(hidden_channels, out_channels)
        self.encoder1 = to_hetero(self.encoder1, metadata_path, aggr="sum")

    def init_weight(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self,x,y, mask_x,mask_y ,edge_index,mask_loss_flag, node_loss_flag, meta_loss_flag):
        criterion = nn.MSELoss()
        #先算transforme的回归预测
        trans_x = x.transpose(1, 2).float()  # 128 15 51
        trans_x = self.embedding(trans_x)  # 128 15 512
        trans_x = self.encoder_trans(trans_x)  # 128 15 512
        trans_x = trans_x.transpose(1, 2)  # 128 512 15
        pre = torch.sum(trans_x, 2)  # 128 512
        pre = torch.relu(pre)
        pre = self.Linear1(pre)  # 过一个（512，51） -> 128 51

        pre_loss = criterion(pre,y)

        if mask_loss_flag == True:
       # mask任务的自回归预测
            mask_x = mask_x.transpose(1, 2).float()  # 125 15 51
            mask_x = self.embedding(mask_x)  # 128 15 512
            mask_x = self.encoder_trans(mask_x)  # 128 15 512
            mask_x = self.Linear1(mask_x)  # 128 15 51
            mask_x = mask_x.transpose(1, 2)  # 128 51 15
            mask_x = torch.relu(mask_x)
            mask_pre = self.decoder(mask_x)  # 125 51 3
            mask_loss = criterion(mask_pre,mask_y)
        else:
            mask_loss = torch.tensor(0.0)
        # 先获取同质图数据
        data_x, data_edge_index = self.GDN_model(x, edge_index)

        if node_loss_flag ==True:
            #节点对比：数据增强
            data_x_1 = drop_feature(data_x)
            data_x_2 = drop_feature(data_x)
            edge_index_1 = drop_edge(data_edge_index) #2x91927
            edge_index_2 = drop_edge(data_edge_index) #2x91372

            #转异质图
            data_hetero_1 = get_heter_data(data_x_1, edge_index_1,self.A_Class,self.B_Class)
            data_hetero_2 = get_heter_data(data_x_2, edge_index_2,self.A_Class,self.B_Class)

            #开始编码对比
            x_dict_1 = self.encoder(data_hetero_1.x_dict,      #{'A':{tensor 3072x128},'B':{3456x128}} #跳不进去的
                                    data_hetero_1.edge_index_dict)
            x_dict_2 = self.encoder(data_hetero_2.x_dict,
                                    data_hetero_2.edge_index_dict)

            # 合并A, B feat
            x_1 = torch.concat(tuple(x_dict_1.values()), dim=0)
            x_2 = torch.concat(tuple(x_dict_2.values()), dim=0)
            # 计算 grace_losss
            node_loss = calc_infoNCE(x_1, x_2)
        else:
            node_loss = torch.tensor(0.0)

        if meta_loss_flag == True:
            # 构建元路径的异构图
            data_hetero = get_heter_data(data_x, data_edge_index,self.A_Class,self.B_Class)
            adj = to_dense_adj(data_edge_index).squeeze()
            # 获取元路径对应的正，负样本编码
            (a1, b1_p, b1_p_e, b1_n, a1_n_e, aa1b_s, aa1b_e) = get_a_id(adj)

            # 获取q
            a1_feat = data_hetero['A'].x[a1]
            b1_feat = data_hetero['B'].x[b1_p]
            b1_feat_e = data_hetero['B'].x[b1_p_e]
            qs = b1_feat + b1_feat_e + a1_feat  # 3x128

            # 添加元路径，model计算
            data_hetero = self.transform1(data_hetero)
            x_dict_metapath = self.encoder1(data_hetero.x_dict,
                                            data_hetero.edge_index_dict)

            # 获取 正样本和负样本数据
            a1_p = x_dict_metapath['A'][a1]
            b1_p_start = x_dict_metapath['B'][b1_p]
            b1_p_end = x_dict_metapath['B'][b1_p_e]
            kps = b1_p_start + b1_p_end + a1_p
            # qs相当于是信息传递更新前的正样本表征，kps是之后的

            b1_n_start = x_dict_metapath['B'][b1_n]
            a1_p_end = x_dict_metapath['A'][a1_n_e]  # baa，应该是a1_n_end?

            ng_size = min(b1_n_start.shape[0], a1_p_end.shape[0])

            aab_size = min(len(aa1b_s), len(aa1b_e))

            # aaa = x_dict_metapath['A'][aa1a_s[:aaa_size]] + x_dict_metapath['A'][
            # aa1a_e[:aaa_size]] + a1_p
            aab = x_dict_metapath['A'][aa1b_s[:aab_size]] + x_dict_metapath['B'][
                aa1b_e[:aab_size]] + a1_p  # 8x128

            kns = b1_n_start[:ng_size] + a1_p_end[:ng_size] + a1_p  # 这里拿的是BAA
            kns = torch.vstack((aab, kns))  # 两者之和就是负样本的   #15x128

            # 计算元路径的loss
            meta_path_loss = calc_infoNCE_metapath(qs, kps, kns)
        else:
            meta_path_loss = torch.tensor(0.0)
        return pre,mask_loss, pre_loss, node_loss,meta_path_loss


def randint_generation(min, max, mount):
    list = []
    while len(list) != mount:
        unit = random.randint(min, max)
        if unit not in list:
            list.append(unit)
    return list


# input:x:128x51x15     (bath_size,feature_num,stride_win)
# output:mask_x:128x51x15, mask_y:128x51x3
def mask_func(x):
    Mask_Matrix = torch.ones_like(x)
    ground_truth = torch.zeros(128, 51, 3)

    for i in range(Mask_Matrix.size(0)):  # 迭代每一个seq_len，选出3行进行mask
        index1, index2, index3 = randint_generation(0, Mask_Matrix.size(2) - 1, 3)
        ground_truth[i, :, 0] = x[i, :, index1]
        ground_truth[i, :, 1] = x[i, :, index2]
        ground_truth[i, :, 2] = x[i, :, index3]

        Mask_Matrix[i, :, index1] = -1  # mask处置为-1
        Mask_Matrix[i, :, index2] = -1
        Mask_Matrix[i, :, index3] = -1
    Mask_x = x * Mask_Matrix
    return Mask_x, ground_truth

def embedding_Linear(x):
    embedding = nn.Linear(x.size(-1),512)
    x = embedding(x)
    return x




