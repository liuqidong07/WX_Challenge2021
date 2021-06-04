# -*- encoding: utf-8 -*-
'''
@File    :   pretrain.py
@Time    :   2021/06/02 11:09:47
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import configparser

from utils.selection import *
from utils.util import set_seed
from layers.input import sparseFeat
from generator import DataGenerator
import setproctitle


def pretrain(config, mode='offline'):
    m_section = config['Model']['model']
    v_dim = config.getint(m_section, 'embedding_dim')
    seed = config.getint('Train', 'seed')
    set_seed(seed)

    '''载入数据, 管理特征, 构建数据生成器'''
    features = ['user_id', 'item_id', 'author_id', 'item_song', 'item_singer', 'item_seconds']
    data_generator = DataGenerator(config, mode=mode, features=features)
    
    '''输出关键超参数'''
    print('learning rate: ' + str(config.get('Train', 'lr')))
    print('batch size: ' + str(config.get('Train', 'batch_size')))
    print('embedding dimension: ' + str(config.get(config['Model']['model'], 'embedding_dim')))
    print('L2 regularization: ' + str(config.get('Train', 'l2')))

    # 记录最好的结果和训练轮次
    metric = 0
    iteration = 0

    config.set('Model', 'target', 'pretrain')
        
    # 构建输入特征列表
    voca_dict = data_generator.get_feature_info()
    if 'item_ocr' in features:
        features.remove('item_ocr')
    feat_list = [sparseFeat(feat, voca_dict[feat], v_dim) for feat in features]
    if 'item_ocr' in features:
        feat_list.append(sparseFeat('item_ocr', voca_dict['item_ocr']))
    
    model = select_model(m_section)(config, feat_list)
    if config.getboolean('Device', 'cuda'):
        model.to('cuda:' + config.get('Device', 'device_tab'))
    data_generator.target = 'read_comment'  # 设置生成器的目标
    model.fit(data_generator, mode)

    model.save_best_model()
    metric = model.best_metric
    iteration = model.best_iteration

    print('The best iteration of pretrain is %d' % iteration)
    print('The pretrain uAUC is: %.5f' % metric)

    return metric
    


if __name__ == '__main__':
    setproctitle.setproctitle("Qidong's Competition")
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    pretrain(config, mode='offline')    # 修改此处来切换线上和线下







