# # @Time    : 2022/10/29 8:54 下午
# # @Author  : tang
# # @File    : Transformer.py
#
# import torch.nn as nn
# import torch
# import random
#
# class Transformer_Model(nn.Module):
#     def __init__(self, mask_len, num_Layer, d_model, dropout=0):
#         super(Transformer_Model, self).__init__()
#         self.src_mask = None
#         self.src_key_padding = None
#
#         self.embedding = nn.Linear(51,512)
#
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=16, dropout=0.1, batch_first=True)
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_Layer)
#
#
#
#         self.decoder = nn.Linear(15, mask_len)
#         self.Linear = nn.Linear(d_model, 51)
#
#         self.init_weight()
#
#
#
#     def init_weight(self):
#         initrange = 0.1
#         self.decoder.bias.data.zero_()
#         self.decoder.weight.data.uniform_(-initrange, initrange)
#
#         # self.embedding.bias.data.zero_()
#         # self.embedding.weight.data.uniform_(-initrange, initrange)
#
#     # input: x mask_x:128x51x15
#     # output-> pre:128x51, mask_pre:128x51x3   ,
#
#     def forward(self, x, mask_x):
#
#         x = x.transpose(1,2).float() #128 15 51
#         x = self.embedding(x)  #128 15 512
#         x = self.encoder(x) #128 15 512
#         x = x.transpose(1,2) #128 512 15
#         pre = torch.sum(x,2) # 128 512
#         pre = torch.relu(pre)
#         pre = self.Linear(pre) #过一个（512，51） -> 128 51
#
#         mask_x = mask_x.transpose(1,2).float()    #125 15 51
#         mask_x = self.embedding(mask_x)   #128 15 512
#         mask_x = self.encoder(mask_x)    #128 15 512
#         mask_x = self.Linear(mask_x)    #128 15 51
#         mask_x = mask_x.transpose(1,2)   # 128 51 15
#         mask_x = torch.relu(mask_x)
#         mask_pre = self.decoder(mask_x)  #125 51 3
#
#
#
#         return mask_pre, pre
#
#
#
#
# def randint_generation(min, max, mount):
#     list = []
#     while len(list) != mount:
#         unit = random.randint(min, max)
#         if unit not in list:
#             list.append(unit)
#     return list
#
#
# # input:x:128x51x15     (bath_size,feature_num,stride_win)
# # output:mask_x:128x51x15, mask_y:128x51x3
# def mask_func(x):
#     Mask_Matrix = torch.ones_like(x)
#     ground_truth = torch.zeros(128, 51, 3)
#
#     for i in range(Mask_Matrix.size(0)):  # 迭代每一个seq_len，选出3行进行mask
#         index1, index2, index3 = randint_generation(0, Mask_Matrix.size(2) - 1, 3)
#         ground_truth[i, :, 0] = x[i, :, index1]
#         ground_truth[i, :, 1] = x[i, :, index2]
#         ground_truth[i, :, 2] = x[i, :, index3]
#
#         Mask_Matrix[i, :, index1] = -1  # mask处置为-1
#         Mask_Matrix[i, :, index2] = -1
#         Mask_Matrix[i, :, index3] = -1
#     Mask_x = x * Mask_Matrix
#     return Mask_x, ground_truth
#
# def embedding_Linear(x):
#     embedding = nn.Linear(x.size(-1),512)
#     x = embedding(x)
#     return x