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
import json

TARGETS = ['read_comment', 'like', 'click_avatar', 'forward']
WEIGHTS = {'read_comment': 4, 'like': 3, 'click_avatar': 2, 'forward': 1}

def main(config, mode='offline'):
    m_section = config['Model']['model']
    v_dim = config.getint(m_section, 'embedding_dim')

    '''载入数据, 管理特征, 构建数据生成器'''
    features = ['user_id', 'item_id', 'author_id', 'item_song', 'item_singer', 'device', 'item_ocr']
    data_generator = DataGenerator(config, mode=mode, features=features)
    
    '''输出关键超参数'''
    print('learning rate: ' + str(config.get('Train', 'lr')))
    print('batch size: ' + str(config.get('Train', 'batch_size')))
    print('embedding dimension: ' + str(config.get(self.config['Model']['model'], 'embedding_dim')))
    print('L2 regularization: ' + str(config.get('Train', 'l2')))

    # 记录最好的结果和训练轮次
    metric = {}
    iteration = {}
    '''对每个目标训练一个模型'''
    for target in TARGETS:
        print('\n' + 'Start: ' + target)
        config.set('Model', 'target', target)
        
        # 构建输入特征列表
        voca_dict = data_generator.get_feature_info()
        if 'item_ocr' in features:
            features.remove('item_ocr')
        feat_list = [sparseFeat(feat, voca_dict[feat], v_dim) for feat in features]
        if 'item_ocr' in features:
            feat_list.append(sparseFeat('item_ocr', voca_dict['item_ocr'], 512))
    
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

        res_dict = {}
        res_dict['iteration'] = iteration
        res_dict['metric'] = metric
        res_dict['uAUC'] = uAUC
        to_json(config, res_dict)
    
    print('Mission Complete!')

    return uAUC


def to_json(config, res_dict):
    params = {'lr': config.getfloat('Train', 'lr'),
              'bs': config.getint('Train', 'batch_size'),
              'embedding_dim': config.getint(config['Model']['model'], 'embedding_dim'),
              'l2': config.getfloat('Train', 'l2'),
              'optimizer': config.get('Train', 'optimizer')}
    res_dict['params'] = params
    file_name = './log/json/lr' + config.get('Train', 'lr') + \
                '_bs' + config.get('Train', 'batch_size') + \
                '_em' + config.get(config['Model']['model'], 'embedding_dim') + \
                '_l2' + config.get('Train', 'l2') + '.json'
    with open(file_name, 'w') as f:
        json.dump(res_dict, f)




if __name__ == '__main__':
    setproctitle.setproctitle("Qidong's Competition")
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    main(config, mode='offline')    # 修改此处来切换线上和线下



