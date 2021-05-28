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

from scipy.sparse import data
from utils.selection import *
from layers.input import sparseFeat
from generator import DataGenerator
import setproctitle

TARGETS = ['read_comment', 'like', 'click_avatar', 'forward']
WEIGHTS = {'read_comment': 4, 'like': 3, 'click_avatar': 2, 'forward': 1}

def main(config, mode='offline'):
    m_section = config['Model']['model']
    v_dim = config.getint(m_section, 'embedding_dim')

    '''载入数据, 管理特征, 构建数据生成器'''
    features = ['user_id', 'item_id', 'author_id', 'item_song', 'item_singer', 'device']
    data_generator = DataGenerator(config, mode=mode, features=features)
    
    # 记录最好的结果和训练轮次
    metric = {}
    iteration = {}
    '''对每个目标训练一个模型'''
    for target in TARGETS:
        print('\n' + 'Start: ' + target)
        config.set('Model', 'target', target)
        
        # 构建输入特征列表
        voca_dict = data_generator.get_feature_info()
        feat_list = [sparseFeat(feat, voca_dict[feat], v_dim) for feat in features]
    
        model = select_model(m_section)(config, feat_list)
        if config.getboolean('Device', 'cuda'):
            model.to('cuda:' + config.get('Device', 'device_tab'))
        data_generator.target = target  # 设置生成器的目标
        model.fit(data_generator, mode)

        if mode == 'offline':
            model.save_best_model()
            metric[target] = model.best_metric
            iteration[target] = model.best_iteration
        
        del model

    if mode == 'offline':
        uAUC = 0
        scale = 0
        for target in TARGETS:
            uAUC += metric[target] * WEIGHTS[target]
            scale += WEIGHTS[target]
        uAUC = uAUC / scale
        print('The best iteration of each target is' + str(iteration))
        print('The best uAUC of each target is' + str(metric))
        print('The weighted uAUC is: %.5f' % uAUC)
    
    print('Mission Complete!')


if __name__ == '__main__':
    setproctitle.setproctitle("Qidong's Competition")
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    main(config, mode='offline')    # 修改此处来切换线上和线下



