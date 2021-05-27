# -*- encoding: utf-8 -*-
'''
@File    :   deepfm.py
@Time    :   2021/05/24 23:05:22
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
from collections import OrderedDict
from model.basemodel import BaseModel
import torch
import torch.nn as nn


class DeepFM(BaseModel):
    def __init__(self, config, feat_list):
        super(DeepFM, self).__init__(config)
        
        self.EMdict = nn.ModuleDict({})
        self.FMLinear = nn.ModuleDict({})
        input_size = 0
        for feat in feat_list:
            self.FMLinear[feat.feat_name] = nn.Embedding(feat.vocabulary_size, 1)
            self.EMdict[feat.feat_name] = nn.Embedding(feat.vocabulary_size, feat.embedding_dim)
            input_size += feat.embedding_dim
        
        self.dnn = nn.Sequential(OrderedDict([
            ('L1', nn.Linear(input_size, 200)),
            ('act1', nn.ReLU()),
            ('L2', nn.Linear(200, 200)), 
            ('act2', nn.ReLU()),
            ('L3', nn.Linear(200, 1))
        ]))
        
        self.out = nn.Sigmoid()

    def forward(self, x):
        EMlist = []
        fmlinear = 0
        '''get embedding list'''
        for key in x.keys():
            EMlist.append(self.EMdict[key](x[key]))
            fmlinear += self.FMLinear[key](x[key])  # (bs, 1)
        
        
        '''FM'''
        in_fm = torch.stack(EMlist, dim=1) # (bs, feat_num, em_dim)
        square_of_sum = torch.pow(torch.sum(in_fm, dim=1), 2)  # (bs, em_dim)
        sum_of_square = torch.sum(in_fm ** 2, dim=1)    # (bs, em_dim)
        yFM = 1 / 2 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)   # (bs, 1)
        yFM += fmlinear

        '''DNN'''
        in_dnn = torch.cat(EMlist, dim=1)    # (bs, em_dim*feat_num)
        yDNN = self.dnn(in_dnn) # (bs, 1)

        y = self.out(yFM + yDNN)

        return y.float()



