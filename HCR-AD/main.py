
import time
import pandas as pd
import numpy as np
import torch

from torch.utils.data import DataLoader, random_split, Subset
import sys
from datetime import datetime
from util.env import get_device, set_device

import os
import argparse
from pathlib import Path
import random
import sklearn
from datasets.TimeDataset import TimeDataset
from train import train_model

from test import test
from evaluate import get_err_scores, get_val_performance_data, get_full_err_scores
from util.net_struct import *
from util.preprocess import *
from datasets.Swat_Graph_metadata import *
from models.STGAD import *



class Main():
    def __init__(self, train_config, env_config, debug=False):
        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset']
        train_data= pd.read_csv(f'./data/{dataset}/train.csv', sep=',', index_col=0)
        test_data = pd.read_csv(f'./data/{dataset}/test.csv', sep=',', index_col=0)

        if 'attack' in train_data.columns:
            train_data = train_data.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)  # 列名
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc,
                                      list(train_data.columns),
                                      feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
        self.feature_map = feature_map

        train_dataset_indata = construct_data(train_data,
                                              feature_map,
                                              labels=0)
        test_dataset_indata = construct_data(test_data,
                                             feature_map,
                                             labels=test_data.attack.tolist())
        # train_data = torch.from_numpy(train_data.values)
        # test_data = torch.from_numpy(test_data.values)

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        #构建滑动窗口形式的数据集
        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index,mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata,fc_edge_index, mode='test', config=cfg)


        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'], val_ratio = train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset


        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                            shuffle=False, num_workers=0,drop_last=True)

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        #获取异构图的metadata
        metadata, metadata_path, transform1, A_Class, B_Class = Swat_heterograph_data()

        #self.model = Transformer_Model(3 ,3, 512)
        self.model = STGAD_Model(edge_index_sets,
                           len(feature_map),
                           dim = train_config['dim'],
                           topk = train_config['topk'],
                           hidden_channels = train_config['GConv_hidden_channel'],
                           out_channels = train_config['GConv_out_channel'],
                           metadata = metadata,
                           metadata_path = metadata_path,
                           mask_len = train_config['mask_len'] ,
                           num_layer = train_config['encoder_num_layer'],
                           d_model = train_config['d_model'],
                           A_Class = A_Class,
                           B_Class = B_Class,
                           transform = transform1

                            )




    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index+val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index+val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)


        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                shuffle=True,drop_last=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                shuffle=False,drop_last=True)

        return train_dataloader, val_dataloader


    def run(self):
        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]

            self.train_log = train_model(self.model,
                                   model_save_path,
                                   config=train_config,
                                   train_dataloader=self.train_dataloader,
                                   val_dataloader=self.val_dataloader,
                                   test_dataloader=self.test_dataloader,
                                   test_dataset=self.test_dataset,
                                   train_dataset=self.train_dataset,
                                   dataset_name=self.env_config['dataset']
                                   )

            self.model.load_state_dict(torch.load(model_save_path))
            best_model = self.model

            _, self.test_result = test(best_model, self.test_dataloader)
            _, self.val_result = test(best_model, self.val_dataloader)

            self.get_score(self.test_result, self.val_result)

    def get_score(self, test_result, val_result):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()

        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        #top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)

        print('finish time:')
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
        print('=========================** Result **============================\n')
        top1_best_info =[0,0,0]
        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}')
        print(f'F1 score: {info[0]}')
        print(f'AUC score: {info[3]}')



    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']

        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type=int, default=128)
    parser.add_argument('-epoch', help='train epoch', type=int, default=1)
    parser.add_argument('-slide_win', help='slide_win', type=int, default=15)
    parser.add_argument('-dim', help='embedding_dimension', type=int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type=int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type=str, default='')
    parser.add_argument('-dataset', help='WADI / SWAT', type=str, default='SWAT')
    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type=int, default=0)
    parser.add_argument('-comment', help='experiment comment', type=str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type=int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type=int, default=256)
    parser.add_argument('-decay', help='decay', type=float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.1)
    parser.add_argument('-topk', help='topk num', type=int, default=20)
    parser.add_argument('-report', help='best / val', type=str, default='val')
    parser.add_argument('-load_model_path', help='trained model path', type=str, default='')
    parser.add_argument('-GConv_hidden_channel', help='Gconv', type=int, default=64)
    parser.add_argument('-GConv_out_channel', help='Gconv', type=int, default=128)
    parser.add_argument('-d_model', help='model_dimension', type=int, default=512)
    parser.add_argument('-mask_len', help='mask_tesk', type=int, default=3)
    parser.add_argument('-encoder_num_layer', help='tranformer_encoder', type=int, default=3)


    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
        'GConv_hidden_channel':args.GConv_hidden_channel,
        'GConv_out_channel':args.GConv_out_channel,
        'd_model':args.d_model,
        'mask_len':args.mask_len,
        'encoder_num_layer':args.encoder_num_layer
    }

    env_config = {
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }

    main = Main(train_config, env_config, debug=False)
    main.run()
