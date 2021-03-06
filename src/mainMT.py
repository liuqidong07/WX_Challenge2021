# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2021/05/24 23:04:27
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

TARGETS = ['read_comment', 'like', 'click_avatar', 'forward']
WEIGHTS = {'read_comment': 4, 'like': 3, 'click_avatar': 2, 'forward': 1}

def main(config, mode='offline'):
    m_section = config['Model']['model']
    v_dim = config.getint(m_section, 'embedding_dim')

    '''载入数据, 管理特征, 构建数据生成器'''
    features = ['user_id', 'item_id', 'author_id', 'item_song', 'item_singer', 'item_ocr', 'item_seconds']
    data_generator = DataGenerator(config, mode=mode, features=features)
    
    # 记录最好的结果和训练轮次        
    # 构建输入特征列表
    voca_dict = data_generator.feature_info
    feat_list = []
    for feat in features:
        if feat == 'item_ocr':
            feat_list.append(sparseFeat('item_ocr', voca_dict['item_ocr'], 32))
        else:
            feat_list.append(sparseFeat(feat, voca_dict[feat], v_dim))
    
    if config['Model']['model'].lower() == 'deepfm':
        model = DeepFM_MT(config, feat_list, task_num=4)
    elif config['Model']['model'].lower() == 'mmoe':
        model = MMOE(config, feat_list, task_num=4, expert_num=8)
    else:
        raise NotImplementedError
    
    if config.getboolean('Device', 'cuda'):
        model.to('cuda:' + config.get('Device', 'device_tab'))
    model.fit(data_generator, mode)

    if mode == 'offline':
        model.save_best_model()
        metric = model.best_metric
        iteration = model.best_iteration
        

    if mode == 'offline':

        print('The best iteration is ' + str(iteration))
        print('The weighted uAUC is: %.5f' % metric)
    
    print('Mission Complete!')


if __name__ == '__main__':
    setproctitle.setproctitle("Qidong's Competition")
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    config.set('Model', 'multi_task', '1')
    main(config, mode='offline')    # 修改此处来切换线上和线下



