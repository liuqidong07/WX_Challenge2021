# -*- encoding: utf-8 -*-
'''
@File    :   _loss.py
@Time    :   2021/04/26 21:09:36
@Author  :   Liu Qidong
@Version :   1.0
@Contact :   dong_liuqi@163.com
'''

# here put the import lib
import torch
import torch.nn as nn
import torch.nn.functional as F


class bpr_loss(nn.Module):
    '''
    Bayesian Personalized Ranking Loss
    loss = - (y_pos - y_neg).sigmoid().log().sum()
    '''
    def __init__(self):
        super(bpr_loss, self).__init__()

    def forward(self, y_ui, y_uj):
        y_uij = y_ui - y_uj     # shape: (bs, 1)
        y_uij = F.sigmoid(y_uij)
        y_uij = torch.log(y_uij + 1e-5)
        loss = -y_uij.squeeze(1).sum()
        return loss


class MT_Loss(nn.Module):
    '''
    Loss function for multi-task.
    '''
    def __init__(self, loss_type):
        super(MT_Loss, self).__init__()
        self.loss_type = loss_type

    def forward(self, y_pred, y_true, weights):
        loss = 0
        for i, w in enumerate(weights):
            if self.loss_type == 'bce':
                criterion = nn.BCELoss()
                y = y_true[:, i]
                y_ = y_pred[i].squeeze()
                loss += w * criterion(y_, y)
        return loss





