import torch

import config
import utils
import numpy as np
import torch.nn.functional as F

#### 需要修改的参数 ####
date_set = "citeseer"
measure = ["dot", "cos_similarity"][0]
# algorithm = ["algorithm1","algorithm2"][0]

print(date_set)
for K in [5, 10, 15, 20, 30, 50, 70, 80, 100]:
    ##### 第一种方式：node_classification #####

    ##### Citeseer、RA #####
    graph_filename = "D:/aggGAN_二/data/" + date_set + "/output/" + date_set + "_pre.cites"
    graph_train = utils.read_edges(graph_filename)
    n_node = len(graph_train.keys())
    adj_matrix = utils.lists_to_matrix(n_node, graph_train)
    Similarity_matrix_RA = utils.RA_similarity(adj_matrix)
    Similarity_row_topK_node = [sorted(range(len(row)), key=lambda i: row[i], reverse=True)[:K] for row in Similarity_matrix_RA]
    Similarity_topK_all_nodes = [elem for sub_lst in Similarity_row_topK_node for elem in sub_lst]
    Similarity_nodes = len(Similarity_topK_all_nodes)

    emb_filename = r"~\teach\node_classification\citeseer\citeseer.emb"
    emb = utils.read_embeddings(emb_filename, n_node, config.n_emb)

    if measure == "dot":
        # 采用直接内积的方式
        Model_matrix_vector = np.dot(emb, np.transpose(emb))
    else:
        # 采用余弦相似度的方式
        mod_row = np.linalg.norm(emb, axis=1, keepdims=True)
        norm_emb = emb / mod_row
        Model_matrix_vector = np.dot(norm_emb, np.transpose(norm_emb))

    # diag_Matrix_similarity = np.diag(np.diag(Model_matrix_vector))
    # Model_matrix_vector = Model_matrix_vector - diag_Matrix_similarity
    Model_row_topK_node = [sorted(range(len(row)), key=lambda i: row[i], reverse=True)[:K] for row in Model_matrix_vector]

    count = 0
    for i in range(n_node):
        intersection = set(Similarity_row_topK_node[i]).intersection(set(Model_row_topK_node[i]))
        count += len(intersection)

    print("训练的向量表示的top-{}与基于图结构相似度的top-{}的比例为：{:.4f}".format(K, K, count / Similarity_nodes))
