# -*- encoding: utf-8 -*-
'''
@File    :   submit.py
@Time    :   2021/05/27 21:42:22
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import configparser
from utils.selection import *
from layers.input import sparseFeat
from generator import DataGenerator
from model.deepfm import DeepFM_MT
from model.mmoe import MMOE
import setproctitle
import torch
import pandas as pd
import time
import pickle


TARGETS = ['read_comment', 'like', 'click_avatar', 'forward']
BEST_EPOCH = {'read_comment': 2, 'like': 1, 'click_avatar': 1, 'forward': 1}
mode = 'online'

def submit(config):
    model_folder = r'./save_model/MT/'
    m_section = config['Model']['model']
    v_dim = config.getint(m_section, 'embedding_dim')

    '''载入数据, 管理特征'''
    features = ['user_id', 'item_id', 'author_id', 'item_song', 'item_singer', 'item_ocr', 'item_seconds']
    
    df = {}
    '''对每个目标训练一个模型'''
    print('\n' + 'Start: ')
    data_generator = DataGenerator(config, mode=mode, features=features)

     # 构建输入特征列表
    voca_dict = data_generator.feature_info
    feat_list = []
    for feat in features:
        if feat == 'item_ocr':
            feat_list.append(sparseFeat('item_ocr', voca_dict['item_ocr'], 32))
        else:
            feat_list.append(sparseFeat(feat, voca_dict[feat], v_dim))
    
    if config['Model']['model'].lower() == 'deepfm':
        model = DeepFM_MT(config, feat_list)
    elif config['Model']['model'].lower() == 'mmoe':
        model = MMOE(config, feat_list, task_num=4, expert_num=8)
    else:
        raise NotImplementedError

    # 加载模型参数
    # TODO: 暂时全部使用第一个epoch训练出的模型
    model.load_state_dict(torch.load(model_folder  + 'epoch_1.ckpt'))
    if config.getboolean('Device', 'cuda'):
        model.to('cuda:' + config.get('Device', 'device_tab'))
        
    test_loader = data_generator.make_test_loader()
    user, item, pred = [], [], []
    for batch in test_loader:
        with torch.no_grad():
            x = model._move_device(batch)
            y = model(x)
            y = torch.cat(y, dim=1)
            pred.append(y)
            user.append(x['user_id'])
            item.append(x['item_id'])

    user = torch.cat(user, dim=0)
    item = torch.cat(item, dim=0)
    pred = torch.cat(pred, dim=0)
    
    pred = pred.to('cpu').squeeze().detach().numpy()
    item = item.to('cpu').squeeze().detach().numpy()
    user = user.to('cpu').squeeze().detach().numpy()

    for i, target in enumerate(TARGETS):
        df[target] = pred[:, i]
    print('Prediction is completed !')

    #df['user_id'] = x['user_id'].to('cpu').detach().tolist()
    #df['item_id'] = x['item_id'].to('cpu').detach().tolist()
    df['user_id'] = user
    df['item_id'] = item
    df = pd.DataFrame(df)
    df = df[['user_id', 'item_id'] + TARGETS]
    df = df.reset_index(drop=True)
    df = transform(df)
    now_str = time.strftime("%m%d-%H%M", time.localtime())
    df.to_csv('./submit/' + now_str + '.csv', index=False)



def transform(df):
    '''将userid和feedid转换为旧的id'''
    with open(r'./data/transform_id.pkl', 'rb') as f:
        user_id, item_id = pickle.load(f)
    df = pd.merge(df, item_id, on='item_id', how='left')
    df = pd.merge(df, user_id, on='user_id', how='left')
    df = df[['userid', 'feedid'] + TARGETS]
    return df


if __name__ == '__main__':
    setproctitle.setproctitle("Qidong's Competition")
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    config.set('Model', 'multi_task', '1')
    submit(config)


