import numpy as np

import config
import torch
import random
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import CustomBCELoss


class SageLayer(nn.Module):
    """
	更新当前节点的嵌入表示，算法1的第五行
	（若选择GCN，则当前节点的嵌入更新为上一时刻此节点的嵌入与其邻居的聚合进行全连接传播操作；
	    反之，则进行级联cat操作）
	"""

    def __init__(self, input_size, out_size, gcn=False):
        super(SageLayer, self).__init__()

        self.input_size = input_size
        self.out_size = out_size

        self.gcn = gcn
        self.weight = nn.Parameter(torch.FloatTensor(out_size, self.input_size if self.gcn else 2 * self.input_size))

        self.init_params()

    def init_params(self):
        for param in self.parameters():
            # nn.init.xavier_uniform_(param)
            init.kaiming_uniform_(param)

    def forward(self, self_feats, aggregate_feats):
        """
		Generates embeddings for a batch of nodes.
		nodes-- list of nodes
		"""
        if not self.gcn:
            combined = torch.cat([self_feats, aggregate_feats], dim=1)
        else:
            combined = aggregate_feats

        combined = F.relu(self.weight.mm(combined.t())).t()

        return combined


class GraphSage(nn.Module):

    def __init__(self, num_layers, input_size, out_size, raw_features, adj_lists, gcn=False, agg_func='MEAN'):
        super(GraphSage, self).__init__()

        self.input_size = input_size
        self.out_size = out_size
        self.num_layers = num_layers
        self.gcn = gcn
        self.agg_func = agg_func

        self.raw_features = raw_features
        self.adj_lists = adj_lists

        # 第一层输入的是原始特征矩阵，所以输入的维度为特征矩阵的列维度；而从第二层开始，输入的矩阵是第一层输出的表示向量矩阵（即人工设定的嵌入维度）
        for index in range(1, num_layers + 1):
            layer_size = out_size if index != 1 else input_size
            setattr(self, 'sage_layer' + str(index), SageLayer(layer_size, out_size, gcn=self.gcn))

    def forward(self, nodes_batch):
        """
		Generates embeddings for a batch of nodes.
		nodes_batch	-- batch of nodes to learn the embeddings
		"""
        lower_layer_nodes = list(nodes_batch)
        nodes_batch_layers = [(lower_layer_nodes,)]
        for i in range(self.num_layers):
            lower_samp_neighs, lower_layer_nodes_dict, lower_layer_nodes = self._get_unique_neighs_list(
                lower_layer_nodes, config.dropout[i])
            nodes_batch_layers.insert(0, (lower_layer_nodes, lower_samp_neighs, lower_layer_nodes_dict))

        assert len(nodes_batch_layers) == self.num_layers + 1

        pre_hidden_embs = self.raw_features
        # 从远阶向近阶进行聚合（比如k=3,那么先聚合3阶邻居信息到2阶；接着聚合2阶邻居信息到1阶；最后聚合1阶邻居信息到给定中心节点）
        for index in range(1, self.num_layers + 1):
            nb = nodes_batch_layers[index][0]
            pre_neighs = nodes_batch_layers[index - 1]

            # 聚合给定节点的邻居特征
            aggregate_feats = self.aggregate(nb, pre_hidden_embs, pre_neighs)
            # 获取当前层的维度信息
            sage_layer = getattr(self, 'sage_layer' + str(index))

            if index > 1:
                nb = self._nodes_map(nb, pre_neighs)
            # 通过聚合的邻居特征和上一时刻给定节点的向量表示，更新此刻给定节点的向量表示
            cur_hidden_embs = sage_layer(self_feats=pre_hidden_embs[nb],
                                         aggregate_feats=aggregate_feats)
            pre_hidden_embs = cur_hidden_embs

        return pre_hidden_embs

    def _nodes_map(self, nodes, neighs):
        """
        返回当前层中，中心节点在所有节点中所在的索引
        """
        layer_nodes, samp_neighs, layer_nodes_dict = neighs
        assert len(samp_neighs) == len(nodes)
        index = [layer_nodes_dict[x] for x in nodes]
        return index

    def _get_unique_neighs_list(self, nodes, dopout):
        _set = set
        to_neighs = [self.adj_lists[int(node)] for node in nodes]  # [{},{},{}...],列表中的每个集合与nodes里的节点是一一对应的，表示该节点的所有一阶邻居节点
        _sample = random.sample
        samp_neighs = [_set(_sample(to_neigh, math.ceil(len(to_neigh) * (1-dopout)))) for to_neigh in to_neighs]

        # sample_neighs：[{nodes[0]及其采样的邻居节点},{nodes[1]及其采样的邻居节点},{nodes[2]及其采样的邻居节点}...]
        samp_neighs = [samp_neigh | {nodes[i]} for i, samp_neigh in enumerate(samp_neighs)]

        # nodes及其所有的邻居节点组成的（唯一序号的）列表
        _unique_nodes_list = list(set.union(*samp_neighs))
        i = list(range(len(_unique_nodes_list)))

        unique_nodes = dict(
            list(zip(_unique_nodes_list, i)))  # _unique_nodes_list中的节点及其所在的索引组成的字典：{唯一节点1:0,唯一节点15:1,唯一节点56:2,...}
        return samp_neighs, unique_nodes, _unique_nodes_list

    def aggregate(self, nodes, pre_hidden_embs, pre_neighs):
        """
        聚合邻居节点对的特征信息（算法1的第四行）
        """
        global aggregate_feats
        unique_nodes_list, samp_neighs, unique_nodes = pre_neighs

        assert len(nodes) == len(samp_neighs)  # 判断是否每个中心节点都进行了邻居采样

        # 判断是否将每个中心节点的自循环加进去（中心节点也能从中心节点自身获取信息）
        indicator = [(nodes[i] in samp_neighs[i]) for i in range(len(samp_neighs))]
        assert (False not in indicator)

        # 如果gcn=False，那么无法从中心节点自身获取信息
        if not self.gcn:
            samp_neighs = [(samp_neighs[i] - {nodes[i]}) for i in range(len(samp_neighs))]

        # 若num_layers=1,输入的是原始特征矩阵，需要根据涉及的所有节点读取所需的特征向量；若num_layers>1,输入的是上层聚合后的嵌入矩阵
        if len(pre_hidden_embs) == len(unique_nodes):
            embed_matrix = pre_hidden_embs
        else:
            embed_matrix = pre_hidden_embs[torch.LongTensor(unique_nodes_list)]

        # 将当前涉及的所有节点生成一个矩阵，行：表示中心节点；列：表示在当前层的所有节点中，哪些是该中心节点所采样到的邻居节点
        mask = torch.zeros(len(samp_neighs), len(unique_nodes))

        # 该列表表示：当前层的所有节点编号对应的mask矩阵中的列索引
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        # 该列表表示：当前层的中心节点所在的行索引
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1

        if self.agg_func == 'MEAN':
            num_neigh = mask.sum(1, keepdim=True)
            for i, neigh in enumerate(num_neigh):
                if neigh == 0:
                    num_neigh[i] = 1

            mask = mask.div(num_neigh)
            aggregate_feats = mask.mm(embed_matrix)

        elif self.agg_func == 'MAX':
            indexs = [x.nonzero() for x in mask == 1]
            aggregate_feats = []
            for feat in [embed_matrix[x.squeeze()] for x in indexs]:
                if len(feat.size()) == 1:
                    aggregate_feats.append(feat.view(1, -1))
                else:
                    aggregate_feats.append(torch.max(feat, 0)[0].view(1, -1))
            aggregate_feats = torch.cat(aggregate_feats, 0)

        return aggregate_feats


class Discriminator(nn.Module):
    def __init__(self, features, adj_lists, n_node, adj_matrix_RA, num_classes=None):
        super(Discriminator, self).__init__()

        self.feature_matrix = torch.FloatTensor(features)
        self.adj_lists = adj_lists
        self.n_node = n_node
        self.adj_matrix_RA = adj_matrix_RA
        if num_classes is not None:
            self.num_classes = num_classes
            self.weight = nn.Parameter(torch.FloatTensor(self.num_classes, config.n_emb))
            init.xavier_uniform_(self.weight)

        self.bias = nn.Parameter(torch.zeros([self.n_node]))
        self.graphsage = GraphSage(config.num_layer, self.feature_matrix.shape[1], config.n_emb,
                                   self.feature_matrix, self.adj_lists, gcn=config.gcn,
                                   agg_func=config.agg_func)

        self.embedding_matrix = None

        self.node_embedding = None
        self.node_neighbor_embedding = None
        self.neighboor_bias = None

    def score(self, node_id, node_neighbor_id):
        """
        表示向量的内积涵义为:两节点相连的评分(score)
        输入的node_id、node_neighbor_id为两个列表，并非两个单一的整数

        最终的形式为：[分数1，分数2，...]
        """
        self.node_embedding = self.embedding_matrix[node_id, :]  # 从embedding_matrix中，筛选出列表node_id中指示的行的值，并组成一个新的嵌入向量矩阵
        self.node_neighbor_embedding = self.embedding_matrix[node_neighbor_id, :]
        self.neighboor_bias = self.bias[node_neighbor_id]

        return torch.sum(input=self.node_embedding * self.node_neighbor_embedding, dim=1) + self.neighboor_bias

    def self_sup_loss(self, node_id, node_neighbor_id, label):
        # 自监督损失(对抗网络的判别器)
        l2_loss = lambda x: torch.sum(x * x) / 2 * config.lambda_dis

        center_embeddings = self.graphsage(np.asarray(node_id))
        neighbor_embeddings = self.graphsage(np.asarray(node_neighbor_id))
        score = torch.sum(input=center_embeddings * neighbor_embeddings, dim=1) + self.bias[node_neighbor_id]
        prob = torch.sigmoid(score)

        # 提取正、负样本的位置(bool型)
        pos_idx = torch.tensor(label) == 1
        neg_idx = torch.tensor(label) == 0

        # 提取正样本、负样本对应的值
        prob_pos = prob[pos_idx]
        prob_neg = prob[neg_idx]

        # 设置正、负样本的标签（0或1）
        label_pos = torch.ones_like(prob_pos)
        label_neg = torch.zeros_like(prob_neg)

        # 读取正样本损失计算时的分布值
        pos_idx_list = [i.item() for i in pos_idx.nonzero()]  # 在每一个批次中，将bool型的索引转换为整数类型
        pos_weight = []
        for index in pos_idx_list:
            pos_weight.append(self.adj_matrix_RA[node_id[index]][node_neighbor_id[index]])

        assert len(pos_weight) == len(prob_pos)
        criterion = CustomBCELoss.CustomBCELoss()
        # 正则项
        regularization = l2_loss(center_embeddings) + l2_loss(neighbor_embeddings) + l2_loss(
            self.bias[node_neighbor_id])

        self_sup_loss = criterion.forward(prob_pos, prob_neg, label_pos, label_neg,
                                          torch.tensor(pos_weight).float()) + regularization

        return self_sup_loss

    def semi_sup_loss(self, node_id, labels, classification):
        logits = classification(self.graphsage(node_id))
        semi_sup_loss = -torch.sum(logits[range(logits.size(0)), labels], 0)
        semi_sup_loss = semi_sup_loss / len(node_id)

        return semi_sup_loss

    def reward(self, node_id, node_neighbor_id):
        return torch.log(1 + torch.exp(self.score(node_id, node_neighbor_id))).detach()
