# -*- encoding: utf-8 -*-
'''
@File    :   dataset.py
@Time    :   2021/05/21 20:23:18
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import pickle
from torch.utils.data import Dataset


class TrainData(Dataset):
    def __init__(self):
        super(TrainData, self).__init__()
        with open('./data/train.pkl') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def forward(self, index):
        return self.data[index]










