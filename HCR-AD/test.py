# @Time    : 2022/10/30 4:18 下午
# @Author  : tang
# @File    : test.py
import torch.nn as nn
import time
import torch
from util.env import *

from models.STGAD import embedding_Linear,mask_func
def test(model, dataloader):
    # test
    device = get_device()
    loss_func = nn.MSELoss(reduction='mean')

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0
    epoch_loss = 0

    criterion = nn.MSELoss()
    for x, y, labels,edge_index in dataloader:
        x, y, labels = [item.float() for item in [x, y, labels]]

        with torch.no_grad():
            mask_x, mask_y = mask_func(x)
            pre,mask_loss, pre_loss, node_loss,meta_path_loss = model(x,y,mask_x,mask_y,edge_index,
                                                            mask_loss_flag = False,
                                                            node_loss_flag = False,
                                                            meta_loss_flag = False)


            #loss = loss_func(predicted.float(), y.float())
            # mask_loss = criterion(mask_pre, mask_y)
            # pre_loss = criterion(pre.float(), y.float())
            loss = mask_loss+pre_loss+node_loss+meta_path_loss

            labels = labels.unsqueeze(1).repeat(1, pre.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = pre
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, pre), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

        test_loss_list.append(loss.item())
        epoch_loss += loss.item()

    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()
    test_labels_list = t_test_labels_list.tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]