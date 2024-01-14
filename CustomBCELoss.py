import torch
import numpy as np
from torch import Tensor
from torch.nn import functional as F


class CustomBCELoss(object):
    def __init__(self):
        super(CustomBCELoss, self).__init__()

    def forward(self, pred_pos: Tensor, pred_neg: Tensor, target_pos: Tensor, target_neg: Tensor, pos_weight: Tensor):
        """通过自定义权重的方式对正负样本的交叉熵损失进行计算"""

        # 计算正样本损失
        pos_loss = F.binary_cross_entropy(pred_pos, target_pos.float(), weight=pos_weight, reduction='sum')

        # 计算负样本损失（负样本为默认的负样本数均值）
        if len(pred_neg) != 0:
            neg_loss = F.binary_cross_entropy(pred_neg, target_neg.float(), reduction='sum')
        else:
            neg_loss = 0

        all_loss = (pos_loss + neg_loss) / (len(pred_pos) + len(pred_neg))

        return all_loss.item()
