# -*- encoding: utf-8 -*-
'''
@File    :   agg_seed.py
@Time    :   2021/06/09 22:29:50
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
#from main import main
#from submit import submit
from mainMT import main
from submitMT import submit
import setproctitle
import configparser
import pandas as pd

seed_list = [2020, 2021, 2022, 2023, 2024]


if __name__ == '__main__':
    setproctitle.setproctitle("Qidong's Competition")
    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')
    config.set('Model', 'multi_task', '1')

    res = []
    '''切换种子进行训练和预测'''
    for seed in seed_list:
        config.set('Train', 'seed', str(seed))
        main(config, mode='online')    # 修改此处来切换线上和线下
        submit(config)
        res.append(pd.read_csv('./submit/' + str(seed) + '.csv'))
    
    '''把多个种子的结果进行平均'''
    TARGETS = ['read_comment', 'like', 'click_avatar', 'forward']
    res_df = res[0][['userid', 'feedid'] + TARGETS]
    res_df[TARGETS] = 0

    for target in TARGETS:
        for i in range(len(res)):
            res_df[target] = res_df[target] + res[i][target] / len(res)
    res_df.to_csv('./submit/agg.csv', index=False)

    print('All missions are completed!')

