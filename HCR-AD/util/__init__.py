import torch
import torch.nn.functional as F

from torch_geometric.utils import dropout_adj


# 计算元路径的loss
def calc_infoNCE_metapath(qs, kps, kns, tau=0.25):
    loss = 0.0
    qs = F.normalize(qs)  #没经过卷积的正样本特征
    kps =F.normalize(kps) #卷积过的正样本
    kns =F.normalize(kns) #卷积过的负样本

    for i in range(len(qs)):    #
        q = qs[i]
        kp = kps[i]
        nor = torch.exp(torch.dot(q,kp)/tau)
        de =0.0
        for kn in kns:
            de+= torch.exp(torch.dot(q,kn)/tau)
        loss+= (-1*(torch.log(nor/(de)))) #之前没有加nor

    return loss/3


# 计算Grace 中的infoNCE, 对两次数据q， k计算info NCE
def calc_infoNCE(q, k):
    # 求dot
    total = torch.mm(q, k.transpose(0, 1))
    # 求exp,并最后求和
    nce = torch.sum(torch.diag(F.log_softmax(total, dim=1)))   #实际上就是拿到负的交叉熵
    # 求平均
    return -1 * nce / q.shape[0]


# 样本随机mask feature
def drop_feature(x, drop_prob=0.3):
    device = x.device
    # 随机分布，取prob的mask idx
    drop_mask = torch.empty(
        (x.size(1), ), dtype=torch.float32).uniform_(0, 1) < drop_prob
    drop_mask = drop_mask.to(device)
    x = x.clone()
    # 对取到的idx赋值0
    x[:, drop_mask] = 0
    return x


# 按比例随机删除edge
def drop_edge(edge_index, drop_prob=0.3):
    edge_index, _ = dropout_adj(edge_index, p=drop_prob)
    return edge_index
