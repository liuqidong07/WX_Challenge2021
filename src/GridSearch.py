# -*- encoding: utf-8 -*-
'''
@File    :   GridSearch.py
@Time    :   2021/05/31 22:24:42
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
from main import main
import setproctitle
import configparser


def grid_search():
    
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')

    lr_range = [0.0001]
    bs_range = [128, 512, 1024]
    em_range = [32, 128]

    best = 0
    best_params = {'lr': 0, 'bs': 0, 'em': 0}

    for lr in lr_range:
        for bs in bs_range:
            for em in em_range:
                config.set('Train', 'lr', str(lr))
                config.set('Train', 'batch_size', str(bs))
                config.set(config['Model']['model'], 'embedding_dim', str(em))
                res = main(config, 'offline')
                if res > best:
                    best = res
                    best_params['lr'] = lr
                    best_params['bs'] = bs
                    best_params['em'] = em
    
    print('The best uAUC is %.5f' % best)
    print('The bset params is ' + str(best_params))



if __name__ == '__main__':
    setproctitle.setproctitle("Qidong's grid search")
    grid_search()







