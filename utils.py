import datetime
import os.path

import numpy as np
from collections import defaultdict
import re
import torch
import math
from src import config
import time


def str_list_to_float(str_list):
    return [float(item) for item in str_list]


def str_list_to_int(str_list):
    return [int(item) for item in str_list]


def read_edges_from_file(filename):
    """从某个特定的文件中，读取边
    edges：[[边1],[边2],[边3],...]
    """
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(re.split('[\s,]+', line)[:-1]) for line in lines]

    return edges


def read_edges(train_filename):
    """获取graphsage格式需要的邻接列表：
                adj_lists:{'0':{},'1':{}...}
    """
    adj_lists = defaultdict(set)
    train_edges = read_edges_from_file(train_filename)

    for edge in train_edges:
        if adj_lists.get(edge[0]) is None:
            adj_lists[edge[0]] = set()
        if adj_lists.get(edge[1]) is None:
            adj_lists[edge[1]] = set()
        adj_lists[edge[0]].add(edge[1])
        adj_lists[edge[1]].add(edge[0])
    return adj_lists


def reindex_id_from_feature_matrix(filename):
    """
    若给定的文件里，编号为引文的编号，则需将编号重定义在[0,n_nodes]的范围内；
    根据特征矩阵，对节点编号、标签进行编号重定义

    node_map:{原始节点编号(str)：规定在num_nodes以内的映射编号(int),....}，按行从上到下对节点进行编号重定义
    label_map:{'net':0,'work':1,...}按标签出现的先后顺序进行编号重定义
    """
    node_map = {}
    # label_map = {}
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            info = line.strip().split()
            node_map[info[0]] = i
            # if not info[-1] in label_map:
            #     label_map[info[-1]] = len(label_map)
    return node_map


def read_labels_from_feature_matrix(filename, n_node):
    """
    从特征矩阵将节点标签进行编号映射，然后再将每个节点的标签保存到标签矩阵中
    """
    label_map = {}
    labels = []
    with open(filename, "r") as f:
        for i, line in enumerate(f):
            info = line.strip().split()
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels.append(label_map[info[-1]])
    labels_matrix = np.zeros((n_node, len(label_map)))
    for i, label in enumerate(labels):
        labels_matrix[i, label] = 1.0
    return len(label_map), labels, labels_matrix


def read_labels_from_file(filename):
    labels = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            node, label = line.split()  # 分割每行中的标签和节点
            labels.append(int(label))
    return labels


def read_embeddings(filename, n_node, n_embed):
    """read pre_trained node embeddings
    节点的编号为几，则将其放到第几行
    """

    embedding_matrix = np.random.rand(n_node, n_embed)
    try:
        with open(filename, "r") as f:
            lines = f.readlines()[1:]  # skip the first line
            for line in lines:
                emd = line.split()
                embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
    except Exception as e:
        print("WARNNING: can not find the pre_embedding file")
    return embedding_matrix


def read_feature_matrix(filename, n_node, n_feats, old2new_idmapping):
    feature_matrix = np.random.rand(n_node, n_feats)
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
            for (oldnid, newnid) in old2new_idmapping:
                feature = lines[oldnid].split()
                feature_matrix[newnid, :] = str_list_to_float(feature[1:])
    except Exception as e:
        print("WARNNING: can not find the feature_matrix file")
    return feature_matrix


def read_feature_matrix_1(filename, n_node, n_feats):
    feature_matrix = np.random.rand(n_node, n_feats)
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                feature = line.split()
                feature_matrix[i, :] = str_list_to_float(feature[1:-1])
    except Exception as e:
        print("WARNNING: can not find the feature_matrix file")
    return feature_matrix


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for computation stability
    return e_x / e_x.sum()


def get_gnn_embeddings(gnn_model, n_node):
    # print('Loading embeddings from trained GraphSAGE model.')
    nodes = np.arange(n_node).tolist()
    b_sz = 500
    batches = math.ceil(len(nodes) / b_sz)

    embs = []
    for index in range(batches):
        nodes_batch = nodes[index * b_sz:(index + 1) * b_sz]
        embs_batch = gnn_model(nodes_batch)
        assert len(embs_batch) == len(nodes_batch)
        embs.append(embs_batch)

    assert len(embs) == batches
    embs = torch.cat(embs, 0)
    assert len(embs) == len(nodes)
    # print('Embeddings loaded.')
    return embs.detach()  # .detach():截断当前参数的梯度；若截断出来的数据未发生改变，则下次可继续进行梯度更新；若截断出来的数据发生变化，则当前数据不再进行梯度更新


def adjust_learning_rate(org_lr, epoch, decay):
    """
    以“反时限学习率衰减”对学习率进行调整
    """
    return org_lr / (1 + epoch * decay)


def lists_to_matrix(n_node, adj_lists):
    """将训练集放到全部节点的邻接矩阵上"""
    MatrixAdjacency = np.zeros([n_node, n_node])
    for row in list(adj_lists.keys()):
        if len(adj_lists[row]) == 0:
            continue
        for col in list(adj_lists[row]):
            MatrixAdjacency[row, col] = 1

    return MatrixAdjacency


def matrix_to_lists(MatrixAdjacency):
    adj_lists = defaultdict(set)
    n_node = len(MatrixAdjacency)
    for i in range(n_node):
        adj_lists[i] = set()

    for row in range(n_node):
        for col in range(n_node):
            if MatrixAdjacency[row][col] != 0:
                adj_lists[row].add(col)

    return adj_lists


def RA_similarity(MatrixAdjacency_train):
    """ 将相似度矩阵每行进行归一化"""
    # 计算基于共同邻居的相似度矩阵 S
    RA_Train = sum(MatrixAdjacency_train)
    RA_Train.shape = (RA_Train.shape[0], 1)
    MatrixAdjacency_Log = MatrixAdjacency_train / RA_Train
    MatrixAdjacency_Log = np.nan_to_num(MatrixAdjacency_Log)
    Matrix_similarity = np.dot(MatrixAdjacency_train, MatrixAdjacency_Log)
    # # print("相似度矩阵为：\n", Matrix_similarity)

    # PA
    # deg_row = sum(MatrixAdjacency_train)
    # deg_row.shape = (deg_row.shape[0], 1)
    # deg_row_T = deg_row.T
    # Matrix_similarity = np.dot(deg_row, deg_row_T)
    # print("相似度矩阵为：\n", Matrix_similarity)

    # 去掉对角线的元素（不考虑自身与自身相似性的情况）
    diag_Matrix_similarity = np.diag(np.diag(Matrix_similarity))
    Matrix_similarity = Matrix_similarity - diag_Matrix_similarity
    # print("去掉自循环的相似度矩阵为：\n", Matrix_similarity)

    # 计算对角矩阵 M
    sum_neighs = Matrix_similarity.sum(1)
    M = np.diag(sum_neighs)

    # 计算对角矩阵的逆
    M_inv = np.linalg.inv(M)

    # 计算归一化后的相似度矩阵
    R = np.matmul(M_inv, Matrix_similarity)
    # print("进行行归一化后的相似度矩阵为：\n", R)
    # R = R - R * MatrixAdjacency_train
    # print("去掉直接相连的边：\n", R)

    return R


def EvalEN(gGAN, epoch, method_name, edge_embed_method='hadamard'):
    return_val = 0
    gen_embedding_matrix = gGAN.generator.embedding_matrix.detach()
    index_node = np.arange(gen_embedding_matrix.shape[0]).astype(np.str).tolist()
    Xgen = dict(zip(index_node, gen_embedding_matrix))
    dis_embedding_matrix = gGAN.discriminator.embedding_matrix
    Xdis = dict(zip(index_node, dis_embedding_matrix))
    if config.app == "link_prediction":
        results_gen = gGAN.lpe.evaluate_ne(gGAN.train_test_split, Xgen, method=method_name + '_den_' + str(epoch),
                                           edge_embed_method=edge_embed_method)
        results_dis = gGAN.lpe.evaluate_ne(gGAN.train_test_split, Xdis, method=method_name + '_dis_' + str(epoch),
                                           edge_embed_method=edge_embed_method)

        auc_gen, auc_dis = results_gen.test_scores.auroc(), results_dis.test_scores.auroc()
        fscore_gen, fscore_dis = results_gen.test_scores.f_score(), results_dis.test_scores.f_score()
        acc_gen, acc_dis = results_gen.test_scores.accuracy(), results_dis.test_scores.accuracy()
        precision_gen, precision_dis = results_gen.test_scores.precision(), results_dis.test_scores.precision()

        # # 对于链接预测，还有一个指标：precision@K
        metric_name, vals = results_gen.get_all(precatk_vals=config.precatk_vals)
        index = len(config.precatk_vals)

        return_val = fscore_gen

        if epoch == "pre_train":
            write_line = '\n {now_time}\n ' \
                         'epoch: {e}\n ' \
                         '\t gen:\t auc:{auc_g:.4f}\t f_score:{fscore_g:.4f}\t acc:{acc_g:.4f}\t precision:{precision_g:.4f}\n' \
                         '\t dis:\t auc:{auc_d:.4f}\t f_score:{fscore_d:.4f}\t acc:{acc_d:.4f}\t precision:{precision_d:.4f}\n'.format(
                now_time=str(datetime.datetime.now()),
                e=epoch,
                auc_g=auc_gen, fscore_g=fscore_gen, acc_g=acc_gen, precision_g=precision_gen,
                auc_d=auc_dis, fscore_d=fscore_dis, acc_d=acc_dis, precision_d=precision_dis)

            write_detail_gen = '\n {}\n\t auroc:{:.4f}\t f_score:{:.4f}\t acc:{:.4f}\t precision:{:.4f}\n'.format(
                str(datetime.datetime.now()), auc_gen, fscore_gen, acc_gen, precision_gen)
            write_detail_dis = '\n {}\n\t auroc:{:.4f}\t f_score:{:.4f}\t acc:{:.4f}\t precision:{:.4f}\n'.format(
                str(datetime.datetime.now()), auc_dis, fscore_dis, acc_dis, precision_dis)
            write_precision_K = '\n {}\n' \
                                'epoch: {}\n' \
                                '\t {}:{:.4f}\t {}:{:.4f}\t {}:{:.4f}\t {}:{:.4f}\t {}:{:.4f}\t {}:{:.4f}\t {}:{:.4f}\t {}:{:.4f}\n'.format(
                str(datetime.datetime.now()), epoch, metric_name[-index], vals[-index], metric_name[-index + 1],
                vals[-index + 1], metric_name[-index + 2], vals[-index + 2],
                metric_name[-index + 3], vals[-index + 3], metric_name[-index + 4], vals[-index + 4],
                metric_name[-index + 5], vals[-index + 5], metric_name[-index + 6], vals[-index + 6],
                metric_name[-index + 7], vals[-index + 7])
        else:
            write_line = 'epoch: {e}\n ' \
                         '\t gen:\t auc:{auc_g:.4f}\t f_score:{fscore_g:.4f}\t acc:{acc_g:.4f}\t precision:{precision_g:.4f}\n' \
                         '\t dis:\t auc:{auc_d:.4f}\t f_score:{fscore_d:.4f}\t acc:{acc_d:.4f}\t precision:{precision_d:.4f}\n'.format(
                e=epoch,
                auc_g=auc_gen, fscore_g=fscore_gen, acc_g=acc_gen, precision_g=precision_gen,
                auc_d=auc_dis, fscore_d=fscore_dis, acc_d=acc_dis, precision_d=precision_dis)

            write_detail_gen = '\t auroc:{:.4f}\t f_score:{:.4f}\t acc:{:.4f}\t precision:{:.4f}\n'.format(auc_gen,
                                                                                                           fscore_gen,
                                                                                                           acc_gen,
                                                                                                           precision_gen)
            write_detail_dis = '\t auroc:{:.4f}\t f_score:{:.4f}\t acc:{:.4f}\t precision:{:.4f}\n'.format(auc_dis,
                                                                                                           fscore_dis,
                                                                                                           acc_dis,
                                                                                                           precision_dis)
            write_precision_K = 'epoch: {}\n\t {}:{:.4f}\t {}:{:.4f}\t {}:{:.4f}\t {}:{:.4f}\t {}:{:.4f}\t {}:{:.4f}\t {}:{:.4f}\t {}:{:.4f}\n'.format(
                epoch, metric_name[-index], vals[-index], metric_name[-index + 1],
                vals[-index + 1], metric_name[-index + 2], vals[-index + 2],
                metric_name[-index + 3], vals[-index + 3], metric_name[-index + 4], vals[-index + 4],
                metric_name[-index + 5], vals[-index + 5], metric_name[-index + 6], vals[-index + 6],
                metric_name[-index + 7], vals[-index + 7])

        if not os.path.exists(config.results_path):
            os.makedirs(config.results_path)
        with open(config.results_filename, "a+") as fp:
            fp.writelines(write_line)
        with open(config.results_path + "precision@K_" + str(config.num_layer) + ".txt", "a+") as fp:
            fp.writelines(write_precision_K)

        for i in range(2):
            with open(config.train_detail_filename[i], "a+") as fp:
                if i == 0:
                    fp.writelines(write_detail_gen)
                else:
                    fp.writelines(write_detail_dis)

    elif config.app == "node_classification":
        results_gen = gGAN.nce.evaluate_ne(Xgen, method_name=method_name + '_gen_' + str(epoch))
        # results_dis = gGAN.nce.evaluate_ne(Xdis, method_name=method_name + '_dis_' + str(epoch))

        # results_gen包含了num_shuff次的数据（本次设定为5，于是要进行循环5次）
        for i in results_gen:
            i.params['eval_time'] = time.time()
        # for i in results_dis:
        #     i.params['eval_time'] = time.time()

        gGAN.scoresheet.log_results(results_gen)  # 保存时，默认保存num_shuff次的平均值
        # # gGAN.scoresheet.log_results(results_dis)

        f1_micro_1 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[0].method)]['f1_micro'])
        f1_micro_2 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[1].method)]['f1_micro'])
        f1_micro_3 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[2].method)]['f1_micro'])
        f1_micro_4 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[3].method)]['f1_micro'])
        f1_micro_5 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[4].method)]['f1_micro'])
        f1_micro_6 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[5].method)]['f1_micro'])
        f1_micro_7 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[6].method)]['f1_micro'])
        f1_micro_8 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[7].method)]['f1_micro'])
        f1_micro_9 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[8].method)]['f1_micro'])

        return_val = (f1_micro_1 + f1_micro_2 + f1_micro_3 + f1_micro_4 + f1_micro_5 + f1_micro_6 + f1_micro_7 + f1_micro_8 + f1_micro_9) / 9

        f1_macro_1 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[0].method)]['f1_macro'])
        f1_macro_2 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[1].method)]['f1_macro'])
        f1_macro_3 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[2].method)]['f1_macro'])
        f1_macro_4 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[3].method)]['f1_macro'])
        f1_macro_5 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[4].method)]['f1_macro'])
        f1_macro_6 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[5].method)]['f1_macro'])
        f1_macro_7 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[6].method)]['f1_macro'])
        f1_macro_8 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[7].method)]['f1_macro'])
        f1_macro_9 = np.mean(
            gGAN.scoresheet._scoresheet[str(config.dataset)][str(results_gen[8].method)]['f1_macro'])

        with open(config.results_path + "f1_micro_" + str(config.num_layer) + ".txt", "a+") as fp:
            if epoch == "pre_train":
                fp.writelines(
                    '\n {} \n \t f1_micro_1: {:.4f} \t f1_micro_2: {:.4f} \t f1_micro_3: {:.4f} \t f1_micro_4: {:.4f} \t f1_micro_5: {:.4f} '
                    '\t f1_micro_6: {:.4f} \t f1_micro_7: {:.4f} \t f1_micro_8: {:.4f} \t f1_micro_9: {:.4f}'.format(
                        datetime.datetime.now(), f1_micro_1, f1_micro_2, f1_micro_3, f1_micro_4, f1_micro_5,
                        f1_micro_6, f1_micro_7, f1_micro_8, f1_micro_9))
            else:
                fp.writelines(
                    '\n \t f1_micro_1: {:.4f} \t f1_micro_2: {:.4f} \t f1_micro_3: {:.4f} \t f1_micro_4: {:.4f} \t f1_micro_5: {:.4f} '
                    '\t f1_micro_6: {:.4f} \t f1_micro_7: {:.4f} \t f1_micro_8: {:.4f} \t f1_micro_9: {:.4f}'.format(
                        f1_micro_1, f1_micro_2, f1_micro_3, f1_micro_4, f1_micro_5,
                        f1_micro_6, f1_micro_7, f1_micro_8, f1_micro_9))

        with open(config.results_path + "f1_macro_" + str(config.num_layer) + ".txt", "a+") as fp:
            if epoch == "pre_train":
                fp.writelines(
                    '\n {} \n \t f1_macro_1: {:.4f} \t f1_macro_2: {:.4f} \t f1_macro_3: {:.4f} \t f1_macro_4: {:.4f} \t f1_macro_5: {:.4f} '
                    '\t f1_macro_6: {:.4f} \t f1_macro_7: {:.4f} \t f1_macro_8: {:.4f} \t f1_macro_9: {:.4f}'.format(
                        datetime.datetime.now(), f1_macro_1, f1_macro_2, f1_macro_3, f1_macro_4, f1_macro_5,
                        f1_macro_6, f1_macro_7, f1_macro_8, f1_macro_9))
            else:
                fp.writelines(
                    '\n \t f1_macro_1: {:.4f} \t f1_macro_2: {:.4f} \t f1_macro_3: {:.4f} \t f1_macro_4: {:.4f} \t f1_macro_5: {:.4f} '
                    '\t f1_macro_6: {:.4f} \t f1_macro_7: {:.4f} \t f1_macro_8: {:.4f} \t f1_macro_9: {:.4f}'.format(
                        f1_macro_1, f1_macro_2, f1_macro_3, f1_macro_4, f1_macro_5,
                        f1_macro_6, f1_macro_7, f1_macro_8, f1_macro_9))

    else:
        raise Exception('The task {} does not exist'.format(config.app))

    return return_val


if __name__ == "__main__":
    import networkx as nx
    from copy import deepcopy
    import scipy.sparse as sp

    n_node = 5
    lists = {0:{1,2,4},1:{0,2},2:{0,1,3},3:{2,4},4:{0,3}}
    adj = lists_to_matrix(n_node=n_node, adj_lists=lists)
    adj = sp.csr_matrix(adj)
    graphMain = nx.from_numpy_matrix(adj.todense())
    listClique = list(nx.find_cliques(graphMain))  # 从图中寻找极大团
    tmp = deepcopy(np.matrix(adj.todense()))  # 对矩阵进行复制
    for i in listClique:
        for j in i:
            for k in i:
                if j != k:
                    adj[j, k] = len(i) - 1
                    adj[k, j] = len(i) - 1
    print(tmp)
    print(adj)