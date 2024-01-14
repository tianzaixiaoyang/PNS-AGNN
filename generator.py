import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from scipy import spatial
import numpy as np


class Generator(nn.Module):
    def __init__(self, n_node, node_emd_init):
        super(Generator, self).__init__()
        self.n_node = n_node
        # self.node_emd_init = node_emd_init

        self.embedding_matrix = nn.Parameter(torch.tensor(node_emd_init))
        self.bias = nn.Parameter(torch.zeros([self.n_node]))

        self.node_embedding = None
        self.node_neighbor_embedding = None
        self.node_neighbor_bias = None

    def all_score(self):
        """
        计算任意节点对的分数
        输出为一个 n_node * n_node的分数矩阵
        注意：源代码有加偏置
        """
        return (torch.matmul(self.embedding_matrix, torch.transpose(self.embedding_matrix, 0, 1)) + self.bias).detach()

    def score(self, node_id, node_neighbor_id):
        """
        表示向量的内积涵义为:两节点相连的评分(score)
        输入的node_id、node_neighbor_id为两个列表，并非两个单一的整数

        最终的形式为：[分数1，分数2，...]
        """
        self.node_embedding = self.embedding_matrix[node_id, :]
        self.node_neighbor_embedding = self.embedding_matrix[node_neighbor_id, :]
        self.node_neighbor_bias = self.bias[node_neighbor_id]

        return torch.sum(input=self.node_embedding * self.node_neighbor_embedding, dim=1) + self.node_neighbor_bias

    def loss(self, prob, reward):
        """
        Args:
            prob: D(Z) （表示：判别器对负样本给出的一个评分结果）
            reward: 强化学习的奖励因子

        原始的生成器损失函数为 minimize mean(log(1-D(Z))), Z为负样本

        但是原始的损失函数无法提供足够梯度，导致生成器得不到训练

        作为替代，实际运行时使用的是 maximize mean(log(D(Z)))

        因此，对 -mean(log(D(Z))) 梯度下降即可
        """
        l2_loss = lambda x: torch.sum(x * x) / 2 * config.lambda_gen
        prob = torch.clamp(input=prob, min=1e-5, max=1)  # 将"input"的值控制在min-max之间
        # 正则项
        regularization = l2_loss(self.node_embedding) + l2_loss(self.node_neighbor_embedding) + l2_loss(self.node_neighbor_bias)

        _loss = -torch.mean(torch.log(prob) * reward) + regularization

        return _loss

