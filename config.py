modes = ["gen", "dis"]

# training settings
batch_size_gen = 64  # batch size for the generator
batch_size_dis = 256  # batch size for the discriminator
lambda_gen = 1e-5  # l2 loss regulation weight for the generator
lambda_dis = 1e-5  # l2 loss regulation weight for the discriminator
n_sample_gen = 5  # 20 number of samples for the generator

lr_gen = 1e-5  # learning rate for the generator
lr_dis = 1e-9  # learning rate for the discriminator

n_epochs = 100  # number of outer loops
n_epochs_gen = 30  # number of inner loops for the generator
n_epochs_dis = 30  # number of inner loops for the discriminator

gen_interval = n_epochs_gen  # sample new nodes for the generator for every gen_interval iterations
dis_interval = n_epochs_dis  # sample new nodes for the discriminator for every dis_interval iterations
update_ratio = 1  # updating ratio when choose the trees(选+择树时的更新比率)

rcmd_K = (2, 10, 20, 50, 100)

# model saving
load_model = False  # whether loading existing model for initialization
save_steps = 10

# other hyper-parameters
n_emb = 128
multi_processing = False  # whether using multi-processing to construct BFS-trees
window_size = 2  # 获得根节点和采样节点之间的路径时，组成节点对的移动窗口大小

# application and dataset settings
app = ["link_prediction", "node_classification"][0]
# select the dataset
dataset_num = 2
dataset = ["lastfm", "citeseer", "cora", "wiki"][dataset_num]
num_feats = [7842, 3703, 1433, 2405][dataset_num]
directed = [False, False, False, True][dataset_num]

lp_train_frac = 0.7
# 聚合函数
agg_func = ["MEAN", "MAX"][1]
# 聚合邻居的阶数
num_layer = 2
gcn = False
dropout = [1/4, 0]

# Read the BFS tree in batches(If set to 1, there is no batching)
cache_batch = [1, 2, 10, 5][0]

# project path
project_path = "D:/aggGAN_二/"
# path settings

# 原始边集、属性矩阵、标签的路径
org_edges_filename = project_path + "data/" + dataset + "/" + dataset + ".cites"
org_feature_filename = project_path + "data/" + dataset + "/" + dataset + ".content"
org_labels_filename = project_path + "data/" + dataset + "/" + dataset + ".labels"

# 经过EvalNE预处理后的边集、属性矩阵的路径（标签可以根据节点映射从原文件中读取）
output_filename = project_path + "data/" + dataset + "/output/"
new_edges_filename = output_filename + dataset + "_pre.cites"

lp_train_filename = project_path + "data/" + dataset + "/train_test_split/" + "trE_0.csv"
lp_test_filename = project_path + "data/" + dataset + "/train_test_split/" + "teE_0.csv"
lp_test_neg_filename = project_path + "data/" + dataset + "/train_test_split/" + "negTeE_0.csv"
labels_filename = project_path + "data/" + dataset + "/" + dataset + ".labels"

precatk_vals = [10, 100, 200, 300, 500, 800, 1000, 1500]

# pretrain_embedding
pretrain_emb_filename_g = project_path + "pre_train/" + app + "/" + dataset + "_pre_train.emb"

if app == "link_prediction":
    cache_filename = project_path + "cache/" + app + "/" + dataset + "_" + str(lp_train_frac) + ".pkl" + "_0"
else:
    cache_filename = project_path + "cache/" + app + "/" + dataset + ".pkl"

train_test_split = project_path + "data/" + dataset + "/train_test_split/"

feature_matrix_filename = project_path + "data/" + dataset + "/" + dataset + ".content"

# 输出结果保存路径
results_path = project_path + "results/" + app + "/" + dataset + "/"
results_filename = results_path + dataset + str(num_layer) + ".txt"
train_detail_filename = [results_path + "gen_detail_" + str(num_layer) + ".txt",
                         results_path + "dis_detail_" + str(num_layer) + ".txt"]
emb_filenames = [results_path + dataset + "_gen_" + str(num_layer) + ".emb",
                 results_path + dataset + "_dis_" + str(num_layer) + ".emb"]

# Model training logs
# dis_state_dict_filename = project_path + "model_params/" + dataset + "_dis.pt"
# gen_state_dict_filename = project_path + "model_params/" + dataset + "_gen.pt"




