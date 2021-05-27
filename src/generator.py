# -*- encoding: utf-8 -*-
'''
@File    :   generator.py
@Time    :   2021/05/21 20:23:18
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

FRACTION = {'read_comment': 0.2, 'like': 0.2, 'click_avatar': 0.1, 'forward': 0.1}

class OfflineData(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


class OnlineData(Dataset):
    def __init__(self, x):
        super().__init__()
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index]


class DataGenerator():
    def __init__(self, config, mode='offline', features=['user_id', 'item_id']):
        self.mode = mode
        self.target = config.get('Model', 'target')
        self.features = features
        self.bs = config.getint('Train', 'batch_size')
        self.num_workers = config.getint('Data', 'num_workers')
        self._load_data()
        if mode == 'offline':
            self._split_data()


    def _load_data(self):
        with open(r'./data/train.pkl', 'rb') as f:
            self.train = pickle.load(f)
        with open(r'./data/test.pkl', 'rb') as f:
            self.test = pickle.load(f)

    
    def _split_data(self):
        self.test = self.train.loc[self.train['date_']==14]
        self.train = self.train.loc[self.train['date_']<14]
        
    

    def get_feature_info(self):
        vocabulary_size = {}
        for feat in self.features:
            '''拼接起来再去得到vocabulary_size, 为了考虑冷启动物品'''
            vocabulary_size[feat] = int(max(pd.concat([self.train, self.test])[feat])) + 1
        return vocabulary_size

    
    def make_train_loader(self):
        self.train_pos = self.train.loc[self.train[self.target]==1]
        self.train_neg = self.train.loc[self.train[self.target]==0]
        self.train_neg = self.train_neg.sample(frac=FRACTION[self.target])
        self.train_sample = pd.concat([self.train_pos, self.train_neg])
        #TODO: 在这里创建向量的时候可以加上device
        trainset = OfflineData(torch.tensor(self.train_sample[self.features].values),
                               torch.tensor(self.train_sample[self.target].values))
        return DataLoader(trainset, 
                          batch_size=self.bs,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=lambda x: collate_feat(x, self.features))


    def make_test_loader(self):
        if self.mode == 'offline':
            testdata = OfflineData(torch.tensor(self.test[self.features].values), 
                                   torch.tensor(self.test[self.target].values))
        elif self.mode == 'online':
            testdata = OnlineData(torch.tensor(self.test[self.features]))
        return DataLoader(dataset=testdata,
                          batch_size=testdata.__len__(),
                          collate_fn=lambda x: collate_feat(x, self.features, self.mode))



def collate_feat(data, features=['user_id', 'item_id'], mode='offline'):
    '''(bs, 2)-->(x, y)'''
    batch_data = {}
    x = list(map(lambda x: x[0], data))
    x = torch.stack(x)  # (bs, 5)
    for i, feat in enumerate(features):
        batch_data[feat] = x[:, i].long()

    if mode == 'offline':
        y = list(map(lambda x: x[1], data))
        y = torch.stack(y)  # (bs, 1)
        y = y.unsqueeze(1)
        return batch_data, y
    
    elif mode == 'online':
        return batch_data



