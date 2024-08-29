import numpy as np
import torch

import torch.nn as nn
import time

import torch
import os.path as osp

import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from util.env import *



from torch.optim import Adam
import torch.optim as optim

from models.STGAD import embedding_Linear,mask_func
from test import test
import matplotlib.pyplot as plt


def train_model(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None,
                test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):
    epoch = config['epoch']
    min_loss = 1e+8
    early_stop_win = 15

    stop_improve_count=0

    dataloader = train_dataloader
    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)
    #optimizer = Adam(model.parameters(), lr=5e-4, weight_decay=0.001)
    #optimizer = optim.Adam(model.parameters(), lr=1e-2,weight_decay=0.00) #1e-2:0.73,1e-3是垃圾
    criterion = nn.MSELoss()
    train_loss_list = []
    pre_loss_list = []
    mask_loss_list=[]
    node_loss_list = []
    meta_path_loss_list=[]
    model.train()

    device = get_device()


    for name, param in model.named_parameters():  # 查看可优化的参数有哪些
        if param.requires_grad:
            print(name)

    for i_epoch in range(6):

        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        epoch_loss = 0
        for i, [x, y, label,edge_index] in enumerate(train_dataloader):  # x:128x51x15, y:128x51, label:128,128 2 2550
            # mask预测
            x, y, edge_index = [
                item.float() for item in [x, y, edge_index]
            ]
            mask_x, mask_y = mask_func(x)
            pre, mask_loss, pre_loss, node_loss,meta_path_loss = model(x,
                                                            y,
                                                            mask_x,
                                                            mask_y,
                                                            edge_index,
                                                            mask_loss_flag = True,
                                                            node_loss_flag = True,
                                                         meta_loss_flag = True)


            optimizer.zero_grad()
            loss = mask_loss+pre_loss+node_loss+meta_path_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            train_loss_list.append(loss.item())
            pre_loss_list.append(pre_loss.item())
            mask_loss_list.append(mask_loss.item())
            node_loss_list.append(node_loss.item())
            meta_path_loss_list.append(meta_path_loss.item())

            epoch_loss += loss.item()

            if i % 10 == 0:
                print("epoch:{},i:{},   pre_loss:{:.4f},mask_loss:{},"
                      "node_loss:{},meta_pathloss:{}, Loss:{:.4f}".format
                      (i_epoch,  i, pre_loss, mask_loss, node_loss,meta_path_loss,loss))

            i = i+1

# each epoch
        print('epoch ({}) (Loss:{:.8f}, epoch_loss:{:.8f})'.format(
                        i_epoch,
                        epoch_loss/len(dataloader), epoch_loss), flush=True
            )
        if val_dataloader is not None:
            val_loss, val_result = test(model, val_dataloader)

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                break

        else:
            if epoch_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = epoch_loss

    plt.figure()
    plt.plot(train_loss_list,label='train_loss')
    plt.plot(pre_loss_list, label='pre_loss')
    plt.plot(mask_loss_list, label='mask_loss')
    plt.plot(node_loss_list, label='node_loss')
    plt.plot(meta_path_loss_list, label='meta_path_loss')
    plt.legend()
    plt.show()
