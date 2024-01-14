import datetime
import random

import networkx as nx
import torch
import torch.nn as nn
import os
import sys
import tqdm
import pickle
import numpy as np
import multiprocessing
import collections
from discriminator import Discriminator
from generator import Generator
import config
import utils

from src.EvalNE.evalne.utils import preprocess as pp
from src.EvalNE.evalne.evaluation.evaluator import LPEvaluator
from src.EvalNE.evalne.evaluation.evaluator import NCEvaluator
from src.EvalNE.evalne.evaluation.score import Scoresheet
from src.EvalNE.evalne.evaluation.split import EvalSplit
from src.EvalNE.evalne.utils import split_train_test as stt
from src.EvalNE.evalne.utils.viz_utils import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class graphGAN(object):
    def __init__(self):
        print("reading graphs...")
        ###############################################################
        #  加载原始“边文件”，并进行预处理：
        #  （去掉自循环边、按照边集重定义节点编号以及去掉一些”不重要“的节点和边） #
        ###############################################################
        G1 = pp.load_graph(config.org_edges_filename, delimiter="\t", directed=config.directed)  # G1包含整个数据集的节点和边
        self.all_edges_G, id_mapping = pp.prep_graph(G1)  # 若 maincc=True，则去掉了一些”不重要“的节点

        self.n_node = len(self.all_edges_G.nodes)
        self.root_nodes = sorted(self.all_edges_G.nodes)

        #############################################################################
        #  （1）获取预处理后的边集情况并保存："dataset/output/stats.txt"
        #  （2）将预处理后的边集保存为新的文件："dataset/output/dataset_prep.cites"         #
        #############################################################################
        if not os.path.exists(config.output_filename):
            os.makedirs(config.output_filename)
        pp.get_stats(G1, config.output_filename + "stats_org.txt")
        pp.get_stats(self.all_edges_G, config.output_filename + "stats_pre.txt")
        if os.path.exists(config.new_edges_filename):
            os.remove(config.new_edges_filename)
        pp.save_graph(self.all_edges_G, output_path=config.new_edges_filename, delimiter='\t', write_stats=False,
                      write_dir=False)
        # 整个原始图的邻接列表
        self.adj_lists = utils.read_edges(config.new_edges_filename)

        if config.app == "link_prediction":
            ########################################################################################
            # link_prediction：(1)对边集进行训练集、测试集的划分,并保存："dataset/train_test_split/...csv  #
            #                  (2)定义模型评估的“评估器“                                               #
            ########################################################################################
            self.train_test_split = EvalSplit()
            # 如果对数据集已经进行了划分，则读取已有的即可；否则，需要对数据集进行划分并保存
            if os.path.exists(config.train_test_split + "trE_0_" + str(config.lp_train_frac) + ".csv"):
                self.train_test_split.read_splits(config.train_test_split, 0,
                                                  nw_name=config.dataset,
                                                  directed=config.directed,
                                                  verbose=True)
            else:
                train_e, train_e_false, test_e, test_e_false = self.train_test_split.compute_splits(self.all_edges_G,
                                                                                                    nw_name=config.dataset,
                                                                                                    train_frac=config.lp_train_frac,
                                                                                                    split_alg="fast")
                stt.store_train_test_splits(config.train_test_split, train_E=train_e, train_E_false=train_e_false,
                                            test_E=test_e, test_E_false=test_e_false)

            self.lpe = LPEvaluator(self.train_test_split, dim=config.n_emb)

            self.graph_train = utils.read_edges(config.train_test_split + "trE_0_" + str(config.lp_train_frac) + ".csv")

            adj_matrix_train = utils.lists_to_matrix(self.n_node, self.graph_train)
            self.adj_matrix_RA = utils.RA_similarity(adj_matrix_train)  # 用来保存正样本中，每个样本的损失占比
            self.graph = utils.matrix_to_lists(self.adj_matrix_RA)

        elif config.app == "node_classification":
            ##############################################
            # node_classification：不需要对边集进行划分
            #                      (1)读取标签             #
            #                      (2)定义模型评估的“评估器“  #
            ###############################################
            self.graph_train = self.adj_lists
            adj_matrix_train = utils.lists_to_matrix(self.n_node, self.graph_train)
            self.adj_matrix_RA = utils.RA_similarity(adj_matrix_train)  # 用来保存正样本中，每个样本的损失占比
            self.graph = utils.matrix_to_lists(self.adj_matrix_RA)

            self.labels = pp.read_labels(config.labels_filename, delimiter="\t", idx_mapping=id_mapping)
            self.nce = NCEvaluator(self.all_edges_G, self.labels, nw_name=config.dataset, num_shuffles=1,
                                   traintest_fracs=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], trainvalid_frac=0.6,
                                   dim=config.n_emb)
            self.scoresheet = Scoresheet()

        else:
            raise Exception("Unknown task: {}".format(config.app))

        # read pre_emb matrix
        print("generating initial embeddings for the generator...")
        node_embed_init_g = np.random.uniform(-0.1, 0.1, (self.n_node, config.n_emb))

        print("reading feature_matrix for the discriminator...")
        feature_matrix_d = utils.read_feature_matrix(filename=config.feature_matrix_filename,
                                                     n_node=self.n_node,
                                                     n_feats=config.num_feats,
                                                     old2new_idmapping=id_mapping)

        # 初始化生成器和判别器
        print("Initializing the generator...")
        self.generator = Generator(n_node=self.n_node, node_emd_init=node_embed_init_g)
        print("Initializing the discriminator...")
        self.discriminator = Discriminator(features=feature_matrix_d, adj_lists=self.adj_lists, n_node=self.n_node, adj_matrix_RA=self.adj_matrix_RA)
        print("Initializing the embedded vector matrix of discriminator...")
        self.discriminator.embedding_matrix = utils.get_gnn_embeddings(self.discriminator.graphsage, self.n_node)

        # construct BFS-tree
        self.trees = None
        if os.path.isfile(config.cache_filename):  # 若BFS-trees已经创建，则只需读取即可（整个训练过程，只需在程序第一次运行时进行创建）
            print("reading BFS-trees from cache...")
            pickle_file = open(config.cache_filename, 'rb')
            self.trees = pickle.load(pickle_file)
            pickle_file.close()
        else:
            print("constructing BFS-trees...")
            pickle_file = open(config.cache_filename, 'wb')
            if config.multi_processing:
                self.construct_trees_with_mp(self.root_nodes)
            else:
                self.trees = self.construct_trees(self.root_nodes)
            pickle.dump(self.trees, pickle_file)
            pickle_file.close()

    def construct_trees_with_mp(self, nodes):
        """use the multiprocessing to speed up trees construction

        Args:
            nodes: the list of nodes in the graph
        """

        cores = multiprocessing.cpu_count() // 2  # 对商向下取整
        pool = multiprocessing.Pool(cores)
        new_nodes = []
        n_node_per_core = self.n_node // cores
        for i in range(cores):
            if i != cores - 1:
                new_nodes.append(nodes[i * n_node_per_core: (i + 1) * n_node_per_core])
            else:
                new_nodes.append(nodes[i * n_node_per_core:])
        self.trees = {}
        trees_result = pool.map(self.construct_trees, new_nodes)
        for tree in trees_result:
            self.trees.update(tree)

    def construct_trees(self, nodes):
        """use BFS algorithm to construct the BFS-trees

        Args:
            nodes: the list of nodes in the graph
        Returns:
            trees: dict, root_node_id -> tree, where tree is a dict: node_id -> list: [father, child_0, child_1, ...]
        """

        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            trees[root][root] = [root]
            used_nodes = set()
            queue = collections.deque([root])
            while len(queue) > 0:
                cur_node = queue.popleft()
                used_nodes.add(cur_node)
                for sub_node in self.graph_train[cur_node]:
                    if sub_node not in used_nodes:
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
                        queue.append(sub_node)
                        used_nodes.add(sub_node)
        return trees

    def prepare_data_for_d(self):
        """generate positive and negative samples for the discriminator, and record them in the txt file"""
        print("preparing data for  discriminator...")
        center_nodes = []
        neighbor_nodes = []
        labels = []
        for i in tqdm.tqdm(self.root_nodes):
            if np.random.rand() < config.update_ratio:  # 更新率越小，同时纳入的源节点越少，准备的数据越少
                pos = list(self.graph[i])
                neg, _ = self.sample(i, self.trees[i], len(pos), for_d=True)
                if len(pos) != 0 and neg is not None:
                    # positive samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(pos)
                    labels.extend([1] * len(pos))

                    # negative samples
                    for n in neg:
                        if n in pos:
                            neg.remove(n)
                    center_nodes.extend([i] * len(neg))
                    neighbor_nodes.extend(neg)
                    labels.extend([0] * len(neg))
        return center_nodes, neighbor_nodes, labels

    def prepare_data_for_g(self):
        """sample nodes for the generator"""
        print("preparing data for generator...")
        paths = []
        for i in tqdm.tqdm(self.root_nodes):
            if np.random.rand() < config.update_ratio:
                sample, paths_from_i = self.sample(i, self.trees[i], config.n_sample_gen, for_d=False)
                if paths_from_i is not None:
                    paths.extend(paths_from_i)
        node_pairs = list(map(self.get_node_pairs_from_path, paths))
        node_1 = []
        node_2 = []
        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                node_1.append(pair[0])
                node_2.append(pair[1])

        reward = self.discriminator.reward(node_1, node_2)
        return node_1, node_2, reward

    def sample(self, root, tree, sample_num, for_d):
        """ sample nodes from BFS-tree
        Args:
            root: int, root node
            tree: dict, BFS-tree
            sample_num: the number of required samples
            for_d: bool, whether the samples are used for the generator or the discriminator
        Returns:
            samples: list, the indices of the sampled nodes(采样到的负样本，用于判别器训练)
            paths: list, paths from the root to the sampled nodes（每个负样本所在的路径，用于生成器的数据准备）
        """

        all_score = self.generator.all_score().numpy()
        samples = []
        paths = []
        n = 0

        while len(samples) < sample_num:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)
            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:  # the tree only has a root
                    return None, None
                if for_d:  # skip 1-hop nodes (positive samples)
                    if node_neighbor == [root]:
                        # in current version, None is returned for simplicity
                        return None, None
                    if root in node_neighbor:
                        node_neighbor.remove(root)
                relevance_probability = all_score[current_node, node_neighbor]  # 获得中心节点及其所有邻居节点的相似度分数
                relevance_probability = utils.softmax(relevance_probability)
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]  # select next node
                paths[n].append(next_node)
                if next_node == previous_node:  # terminating condition
                    samples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return samples, paths

    @staticmethod
    def get_node_pairs_from_path(path):
        """
        given a path from root to a sampled node, generate all the node pairs within the given windows size
        e.g., path = [1, 0, 2, 4, 2], window_size = 2 -->"此处的4即为采样到的负样本"
        node pairs= [[1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]]
        :param path: a path from root to the sampled node
        :return pairs: a list of node pairs
        """

        path = path[:-1]
        pairs = []
        for i in range(len(path)):
            center_node = path[i]
            for j in range(max(i - config.window_size, 0), min(i + config.window_size + 1, len(path))):
                if i == j:
                    continue
                node = path[j]
                pairs.append([center_node, node])
        return pairs

    def write_embeddings_to_file(self):
        """write embeddings of the generator and the discriminator to files"""

        modes = [self.generator, self.discriminator]
        if not os.path.exists(config.results_path):
            os.makedirs(config.results_path)
        for i in range(2):
            if i == 0:
                embedding_matrix = modes[i].embedding_matrix.detach().numpy()
            else:
                embedding_matrix = modes[i].embedding_matrix

            index = np.array(range(self.n_node)).reshape(-1, 1)
            embedding_matrix = np.hstack([index, embedding_matrix])
            embedding_list = embedding_matrix.tolist()
            embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                             for emb in embedding_list]
            if not os.path.exists(config.results_path):
                os.makedirs(config.results_path)
            with open(config.emb_filenames[i], "w+") as f:
                lines = [str(self.n_node) + "\t" + str(config.n_emb) + "\n"] + embedding_str
                f.writelines(lines)


if __name__ == "__main__":
    gAGN = graphGAN()
    emb = utils.read_embeddings(config.emb_filenames[0], 2485, 128)
    labels = []
    for x in gAGN.labels:
        labels.append(x[1])
    plot_emb2d(emb, colors=labels)
