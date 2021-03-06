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
import setproctitle
import torch
import pandas as pd
import time
import pickle


TARGETS = ['read_comment', 'like', 'click_avatar', 'forward']
BEST_EPOCH = {'read_comment': 1, 'like': 1, 'click_avatar': 1, 'forward': 1}
mode = 'online'

def submit(config):
    model_folder = r'./save_model/'
    m_section = config['Model']['model']
    v_dim = config.getint(m_section, 'embedding_dim')

    '''载入数据, 管理特征'''
    features = ['user_id', 'item_id', 'author_id', 'item_song', 'item_singer', 'item_ocr', 'item_seconds']
    
    df = {}
    '''对每个目标训练一个模型'''
    for target in TARGETS:
        print('\n' + 'Start: ' + target)
        config.set('Model', 'target', target)
        data_generator = DataGenerator(config, mode=mode, features=features)
        voca_dict = data_generator.feature_info
        feat_list = []
        for feat in features:
            if feat == 'item_ocr':
                feat_list.append(sparseFeat('item_ocr', voca_dict['item_ocr'], 32))
            else:
                feat_list.append(sparseFeat(feat, voca_dict[feat], v_dim))

    
        model = select_model(m_section)(config, feat_list)
        # 加载模型参数
        # TODO: 暂时全部使用第一个epoch训练出的模型
        model.load_state_dict(torch.load(model_folder + target + 
                                         '/' + 'epoch_' + 
                                         str(BEST_EPOCH[target]) + '.ckpt'))
        if config.getboolean('Device', 'cuda'):
            model.to('cuda:' + config.get('Device', 'device_tab'))
        
        test_loader = data_generator.make_test_loader()
        for batch in test_loader:
            x = model._move_device(batch)
        y = model(x)
        y = y.squeeze()
        model._end_log()    # 把handler进行清空

        df[target] = y.to('cpu').detach().tolist()
        print(target + ' is completed !')

    df['user_id'] = x['user_id'].to('cpu').detach().tolist()
    df['item_id'] = x['item_id'].to('cpu').detach().tolist()
    df = pd.DataFrame(df)
    df = df[['user_id', 'item_id'] + TARGETS]
    df = df.reset_index(drop=True)
    df = transform(df)
    now_str = time.strftime("%m%d-%H%M", time.localtime())
    df.to_csv('./submit/' + config.get('Train', 'seed') + '.csv', index=False)



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
    submit(config)


