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
import setproctitle

TARGETS = ['read_comment', 'like', 'click_avatar', 'forward']

def main(config):
    m_section = config['Model']['model']
    v_dim = config.getint(m_section, 'embedding_dim')

    '''载入数据, 管理特征'''
    features = ['user_id', 'item_id', 'author_id', 'item_song', 'item_singer']
    
    '''对每个目标训练一个模型'''
    for target in TARGETS:
        print('\n' + 'Start: ' + target)
        config.set('Model', 'target', target)
        data_generator = DataGenerator(config, mode='offline', features=features)
        voca_dict = data_generator.get_feature_info()
        feat_list = [sparseFeat(feat, voca_dict[feat], v_dim) for feat in features]
    
        model = select_model(m_section)(config, feat_list)
        if config.getboolean('Device', 'cuda'):
            model.to('cuda:' + config.get('Device', 'device_tab'))
        model.fit(data_generator)
        model.save_best_model()
        del model
    #test_loader = data_generator.make_test_loader()
    #model.evaluate(test_loader)


if __name__ == '__main__':
    setproctitle.setproctitle("Qidong's Competition")
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    main(config)



