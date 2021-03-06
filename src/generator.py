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

# 0.4, 0.4, 0.2, 0.2
FRACTION = {'read_comment': 0.4, 'like': 0.33, 'click_avatar': 0.33, 'forward': 0.2}
TARGET = ['read_comment', 'like', 'click_avatar', 'forward']

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
        self.features = features
        self.bs = config.getint('Train', 'batch_size')
        self.num_workers = config.getint('Data', 'num_workers')
        self.target = None
        self.multi_task = config.getboolean('Model', 'multi_task')
        #self._preprocess()
        self._load_data()
        self.feature_info = self.get_feature_info()
        if mode == 'offline':
            self._split_data()
        self.seed = config.getint('Train', 'seed')
        self.pretrain = config.getboolean('Model', 'pretrain')


    def _load_data(self):
        '''加载数据集'''
        with open(r'./data/train.pkl', 'rb') as f:
            self.train = pickle.load(f)
        with open(r'./data/test.pkl', 'rb') as f:
            self.test = pickle.load(f)

    
    def _split_data(self):
        '''线下测评的话, 分割数据集, 用分割出的测试集代替加载的擦拭及'''
        self.test = self.train.loc[self.train['date_']==14]
        self.train = self.train.loc[self.train['date_']<14]

    
    def _preprocess(self):
        '''对数据进行预处理'''
        data = pd.concat(self.train, self.test)
        # 创建新的id
        self.user_id = data[['userid']].drop_duplicates()
        self.user_id['user_id'] = np.arange(self.user_id.shape[0]) + 1
        self.item_id = data[['feedid']].drop_duplicates()
        self.item_id['item_id'] = np.arange(self.item_id.shape[0]) + 1
        # 拼接新的id
        self.train = self.train.merge(self.item_id, on='feedid')
        self.train = self.train.merge(self.user_id, on='userid')
        self.test = self.test.merge(self.item_id, on='feedid')
        self.test = self.test.merge(self.user_id, on='userid')
        # 删除旧的id
        self.train.drop(columns=['feedid', 'userid'], inplace=True)
        self.test.drop(columns=['feedid', 'userid'], inplace=True)
    

    def get_feature_info(self):
        '''获取feature的信息, 当前只获取类别特征的词表长度'''
        vocabulary_size = {}
        for feat in self.features:
            '''拼接起来再去得到vocabulary_size, 为了考虑冷启动物品'''
            vocabulary_size[feat] = int(max(pd.concat([self.train, self.test])[feat])) + 1
        return vocabulary_size

    
    def make_train_loader(self):
        '''获取训练集的loader'''
        if self.multi_task:
            self.train['label'] = self.train['read_comment'] + self.train['like'] + self.train['click_avatar'] + self.train['forward']
            train_pos = self.train.loc[self.train['label']>0]
            train_neg = self.train.loc[self.train['label']==0]
            train_neg = train_neg.sample(frac=0.4, random_state=self.seed)
            train_sample = pd.concat([train_neg, train_pos])
            trainset = OfflineData(torch.tensor(train_sample[self.features].values), 
                                   torch.tensor(train_sample[TARGET].values))
        else:
            self.train_pos = self.train.loc[self.train[self.target]==1]
            # 取出所有的负样本后进行采样, 然后和正样本集进行拼接
            self.train_neg = self.train.loc[self.train[self.target]==0]
            self.train_neg = self.train_neg.sample(frac=FRACTION[self.target], random_state=self.seed)
            self.train_sample = pd.concat([self.train_pos, self.train_neg])
            #TODO: 在这里创建向量的时候可以加上device
            trainset = OfflineData(torch.tensor(self.train_sample[self.features].values),
                                torch.tensor(self.train_sample[self.target].values))

        if self.pretrain:
            '''将所有目标的正样本作为正样本进行预训练'''
            label = self.train['read_comment']
            label = 0
            for target in TARGET:
                label += self.train[target]
            # 把所有目标有1的赋为1. 否则会出现大于1的
            label.loc[label>0] = 1
            trainset = OfflineData(torch.tensor(self.train[self.features].values), 
                                   torch.tensor(label.values))

        return DataLoader(trainset, 
                          batch_size=self.bs,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=lambda x: collate_feat(x, self.features, multi_task=self.multi_task))


    def make_test_loader(self):
        '''获取测试集的loader'''
        if self.mode == 'offline':
            if self.multi_task:
                testdata = OfflineData(torch.tensor(self.test[self.features].values), 
                                       torch.tensor(self.test[TARGET].values))
            else:
                testdata = OfflineData(torch.tensor(self.test[self.features].values), 
                                       torch.tensor(self.test[self.target].values))
        elif self.mode == 'online':
            testdata = OnlineData(torch.tensor(self.test[self.features].values))
        
        if self.pretrain:
            if self.mode == 'offline':
                '''构建预训练任务的测试集'''
                label = self.test['read_comment']
                label = 0
                for target in TARGET:
                    label += self.test[target]
                label.loc[label>0] = 1
                testdata = OfflineData(torch.tensor(self.test[self.features].values), 
                                       torch.tensor(label.values))
            else:
                testdata = OnlineData(torch.tensor(self.test[self.features].values))

        bs = min(10000, testdata.__len__())

        return DataLoader(dataset=testdata,
                          batch_size=bs,
                          collate_fn=lambda x: collate_feat(x, self.features, self.mode))


    def make_pretrain_loader(self):
        '''将所有目标的正样本作为正样本进行预训练'''
        label = None
        for target in TARGET:
            label += self.train[target]
        pretraindata = OfflineData(torch.tensor(self.train[self.features].values), 
                                   torch.tensor(label.values))
        return DataLoader(pretraindata, 
                          batch_size=self.bs,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=lambda x: collate_feat(x, self.features))
    

    def make_pretest_loader(self):
        '''构建预训练任务的测试集'''
        label = None
        for target in TARGET:
            label += self.train[target]
        pretraintest = OfflineData(torch.tensor(self.train[self.features].values), 
                                   torch.tensor(label.values))
        return DataLoader(dataset=pretraintest,
                          batch_size=pretraintest.__len__(),
                          collate_fn=lambda x: collate_feat(x, self.features, self.mode))



def collate_feat(data, features=['user_id', 'item_id'], mode='offline', 
                 multi_task=False):
    '''
    聚合样本, 把一个batch数据组合成字典形式
    线下数据: (bs, 2)-->(x, y)
    线上数据: (bs)-->(x)
    '''
    batch_data = {}

    if mode == 'offline':
        x = list(map(lambda x: x[0], data)) # 取出全部特征数据
    elif mode == 'online':
        x = data

    x = torch.stack(x)  # (bs, feat_num)
    for i, feat in enumerate(features):
        batch_data[feat] = x[:, i].long()

    if mode == 'offline':
        y = list(map(lambda x: x[1], data))
        y = torch.stack(y)  # (bs, 1)
        y = y.unsqueeze(1)
        return batch_data, y
    
    elif mode == 'online':
        return batch_data



