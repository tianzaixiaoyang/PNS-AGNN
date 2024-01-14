import torch
import numpy as np

import config
import utils
import random
from torch import nn
from utils import get_gnn_embeddings
from graphGAN_RA import graphGAN
from EvalNE.evalne.utils.viz_utils import *


def show_progress(item_name, epoch, batch_index, batch_num, loss):
    """
    可视化并保存进度
    """
    barlen = 40
    ok_len = int((batch_index + 1) / batch_num * barlen)

    epoch_info = 'epoch:{}'.format(epoch + 1)
    loss_info = 'loss: {}'.format(loss)
    batch_info = '{}/{}'.format(batch_index + 1, batch_num)
    bar_str = '[' + '>' * ok_len + '-' * (barlen - ok_len) + ']'
    info_end = '\r'  # 行首
    info_list = [item_name, epoch_info, batch_info, bar_str, loss_info]

    if batch_index + 1 == batch_num:
        info_end = '\n'  # 换行

    progress_info = ' '.join(info_list)
    print(progress_info, end=info_end, flush=True)


def D_step(gGAN, optmizer):
    optimizer_D = optmizer

    center_nodes = []
    neighbor_nodes = []
    labels = []

    for d_epoch in range(config.n_epochs_dis):
        if d_epoch % config.dis_interval == 0:
            center_nodes, neighbor_nodes, labels = gGAN.prepare_data_for_d()
            print("Data preparation is completed, start training the discriminator...")

        train_size = len(center_nodes)
        start_list = list(range(0, train_size, config.batch_size_dis))  # 设置步长，从center_nodes挑选节点，进行mini_batch训练
        np.random.shuffle(start_list)

        all_loss = 0
        for i, start in enumerate(start_list):
            end = start + config.batch_size_dis

            loss_self_sup = gGAN.discriminator.self_sup_loss(node_id=center_nodes[start:end],
                                                             node_neighbor_id=neighbor_nodes[start:end],
                                                             label=labels[start:end])

            loss = loss_self_sup

            optimizer_D.zero_grad()
            loss.backward()
            optimizer_D.step()

            all_loss = all_loss + loss.detach().numpy()
            avg_loss = all_loss / (i + 1)
            show_progress('dis', d_epoch, i, len(start_list), avg_loss)
    gGAN.discriminator.embedding_matrix = get_gnn_embeddings(gGAN.discriminator.graphsage, gGAN.n_node)


def G_step(gGAN, optmizer):
    optimizer_G = optmizer

    node_1 = []
    node_2 = []
    reward = []
    for g_epoch in range(config.n_epochs_gen):
        if g_epoch % config.gen_interval == 0:
            node_1, node_2, reward = gGAN.prepare_data_for_g()
            print("Data preparation is completed, start training the generator...")

        train_size = len(node_1)
        start_list = list(range(0, train_size, config.batch_size_gen))
        np.random.shuffle(start_list)

        all_loss = 0
        for i, start in enumerate(start_list):
            end = start + config.batch_size_gen
            score = gGAN.generator.score(node_id=np.array(node_1[start:end]),
                                         node_neighbor_id=np.array(node_2[start:end]))

            prob = torch.sigmoid(score)
            loss = gGAN.generator.loss(prob=prob,
                                       reward=reward[start:end])

            optimizer_G.zero_grad()
            loss.backward()
            optimizer_G.step()

            all_loss = all_loss + loss.detach().numpy()
            avg_loss = all_loss / (i + 1)
            show_progress('gen', g_epoch, i, len(start_list), avg_loss)


def train():
    # loading the model from the latest checkpoint if exists
    # if os.path.isfile(config.dis_state_dict_filename):
    #     checkpoint_dis = torch.load(config.dis_state_dict_filename)
    #     gGAN.discriminator.load_state_dict(checkpoint_dis)
    #     print('restore models successfully!!!')
    #     gGAN.discriminator.embedding_matrix = get_gnn_embeddings(gGAN.discriminator.graphsage, gGAN.n_node)

    gGAN = graphGAN()

    optimizer_D = torch.optim.AdamW(gGAN.discriminator.parameters())
    optimizer_G = torch.optim.AdamW(gGAN.generator.parameters())

    gGAN.write_embeddings_to_file()
    max_val = utils.EvalEN(gGAN, epoch="pre_train", method_name="aggGAN")

    print("start training...")
    for epoch in range(config.n_epochs):
        print(" epoch %d " % epoch)

        # D-steps
        optimizer_D.param_groups[0]["lr"] = utils.adjust_learning_rate(org_lr=config.lr_dis,
                                                                       epoch=epoch,
                                                                       decay=0)
        D_step(gGAN, optimizer_D)

        # G-steps
        optimizer_G.param_groups[0]["lr"] = utils.adjust_learning_rate(org_lr=config.lr_gen,
                                                                       epoch=epoch,
                                                                       decay=0)

        G_step(gGAN, optimizer_G)

        x = utils.EvalEN(gGAN, epoch=epoch, method_name="aggGAN")
        if x > max_val:
            max_val = x
            gGAN.write_embeddings_to_file()

    if config.app == "node_classification":
        gGAN.scoresheet.write_all(config.results_path + "eval_" + str(config.num_layer) + ".txt")

        # 根据保存的最佳结果进行可视化任务(颜色为节点对应的标签)
        emb = utils.read_embeddings(config.emb_filenames[0], gGAN.n_node, config.n_emb)
        labels = []
        for x in gGAN.labels:
            labels.append(x[1])
        plot_emb2d(emb, colors=labels, filename=config.results_path + "visual_" + str(config.num_layer) + ".pdf")

    print("training completes")


if __name__ == "__main__":
    print("Designing random number seed...")
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(1)
    train()
