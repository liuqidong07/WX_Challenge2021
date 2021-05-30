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


def load_pretrain():
    with open(r'./data/ocr_embedding.pkl', 'rb') as f:
        ocr_em = pickle.load(f)

    ocr_em = torch.tensor(ocr_em).float()
    return ocr_em



