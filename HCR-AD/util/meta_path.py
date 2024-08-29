# @Time    : 2022/11/21 3:52 下午
# @Author  : tang
# @File    : meta_path.py
# 通过原始的edge，随即获取A1， 以及对应的BAB, BAA的编号
from itertools import chain

import torch
import numpy as np
def get_a_id(adj):
    A = [
        0, 1, 5, 6, 7, 8, 16, 17, 18, 25, 26, 27, 34, 35, 36, 37, 38, 39, 40, 41,
        44, 45, 46, 47
    ]
    A = np.array(A)
    A_list = [A + i * 51 for i in range(128)]
    A_list = list(chain(*A_list))
    # B_list 图中所有B类的编号
    B_list = [i for i in range(51 * 128) if i not in A_list]
    # 随机获取a1
    while True: #BAB,  BAA 8 AAB7  #SWAT AB,2,    WADI: A,B
        a1 = np.random.choice(A_list)
        # 获取bab中的 第一个B在异构图的编号
        to_a1 = torch.nonzero(adj[:, a1].view(-1)).view(-1)  #先找出20个邻居
        to_a1_temp = to_a1.clone()
        b1 = to_a1.apply_(lambda x: B_list.index(x) if x in B_list else 0) #邻居中是a的先置0方便剔除出去
        b1_chose = b1[torch.nonzero(b1)].view(-1)  #找出邻居中有B的n个，此时拿到的索引是B_list中的，不是全局的
        # 获取baa中的 第一个B在异构图的编号
        if len(b1_chose) < 10:
            continue
        b1_p = b1_chose[:3]
        b1_n = b1_chose[3:10]       #BAA这种选7条
        ##T.ToUndirected
        a1_temp = to_a1_temp.apply_(lambda x: A_list.index(x) if x in A_list else 0)
        a1_candidates = a1_temp[torch.nonzero(a1_temp)].view(-1)
        if (len(a1_candidates)) < 8:       #想找AAB中的A
            continue

        aa1b_s = a1_candidates[:8]      #也就是AAB这种选8条

        # 获取bab中的 第二个B在异构图的编号
        from_a1 = torch.nonzero(adj[a1, :].view(-1)).view(-1)
        from_a1_temp = from_a1.clone()
        b1_pe = from_a1.apply_(lambda x: B_list.index(x) if x in B_list else 0)
        b1_temp = b1_pe[torch.nonzero(b1_pe)].view(-1)
        if len(b1_temp) < 11:
            continue

        b1_pe = b1_temp[:3]
        aa1b_e = b1_temp[3:11]

        # 获取baa中的 第二个a在异构图的编号
        from_a1_a = from_a1_temp.apply_(lambda x: A_list.index(x)
                                   if x in A_list else 0)
        from_a1_a = from_a1_a[torch.nonzero(from_a1_a)].view(-1)

        # aa1a_e = from_a1_a[:len(aa1a_s)]

        if len(from_a1_a) < 7:
            continue
        from_a1_e = from_a1_a[:7]

        # 获取a1在异构图中的编号
        a1 = A_list.index(a1)

        return a1, b1_p, b1_pe, b1_n, from_a1_e, aa1b_s, aa1b_e