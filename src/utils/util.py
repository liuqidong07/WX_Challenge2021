# -*- encoding: utf-8 -*-
'''
@File    :   util.py
@Time    :   2021/05/30 20:26:30
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import torch
import pickle
import random
import pandas as pd
import numpy as np


def load_pretrain():
    with open(r'./data/ocr_embedding_32.pkl', 'rb') as f:
        ocr_em = pickle.load(f)

    ocr_em = torch.tensor(ocr_em).float()
    return ocr_em


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



