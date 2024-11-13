import sys
import time
import signal
import argparse
import time, os

import pandas as pd

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(1,'../../')
import numpy as np
import torch
import data
from models import *
from comm import CommNetMLP
from utils import *
from action_utils import parse_action_args
from evaluator import Evaluator
from args import get_args

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN, KMeans
import seaborn as sns

# Given some data, runs 2D PCA on it and plots the results.
def plot_comms(_data, special=None, _pca=None, _ax=None):
    if _data.shape[1] > 2:
        if _pca is None:
            _pca = PCA(n_components=2)
            _pca.fit(_data)
        transformed = _pca.transform(_data)
    else:
        transformed = _data
    x = transformed[:, 0]
    y = transformed[:, 1]
    if _ax is None:
        fig, _ax = plt.subplots()
    pcm = _ax.scatter(x, y, s=20, marker='o', c='gray')
    if special is not None:
        special_transformed = _pca.transform(special) if _pca is not None else special
        _ax.scatter(special_transformed[:, 0], special_transformed[:, 1], s=30, c='red')
    return _pca


torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')


def load(path):
    # d = torch.load(path)
    # policy_net.load_state_dict(d['policy_net'])

    load_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), "models")
    print(f"load directory is {load_path}")
    log_path = os.path.join(args.load, args.env_name, args.exp_name, "seed" + str(args.seed), "logs")
    print(f"log dir directory is {log_path}")
    save_path = load_path

    if 'model.pt' in os.listdir(load_path):
        print(load_path)
        model_path = os.path.join(load_path, "model.pt")

    else:
        all_models = sort([int(f.split('.pt')[0]) for f in os.listdir(load_path)])
        model_path = os.path.join(load_path, f"{all_models[-1]}.pt")

    d = torch.load(model_path)
    policy_net.load_state_dict(d['policy_net'])

parser = get_args()
init_args_for_env(parser)
args = parser.parse_args()

if args.ic3net:
    args.commnet = 1
    args.hard_attn = 1
    args.mean_ratio = 0

    # For TJ set comm action to 1 as specified in paper to showcase
    # importance of individual rewards even in cooperative games
    # if args.env_name == "traffic_junction":
    #     args.comm_action_one = True
# Enemy comm
args.nfriendly = args.nagents
if hasattr(args, 'enemy_comm') and args.enemy_comm:
    if hasattr(args, 'nenemies'):
        args.nagents += args.nenemies
    else:
        raise RuntimeError("Env. needs to pass argument 'nenemy'.")

env = data.init(args.env_name, args, False)

num_inputs = env.observation_dim
args.num_actions = env.num_actions

# Multi-action
if not isinstance(args.num_actions, (list, tuple)): # single action case
    args.num_actions = [args.num_actions]
args.dim_actions = env.dim_actions
args.num_inputs = num_inputs

# Hard attention
if args.hard_attn and args.commnet:
    # add comm_action as last dim in actions
    args.num_actions = [*args.num_actions, 2]
    args.dim_actions = env.dim_actions + 1

# Recurrence
if args.commnet and (args.recurrent or args.rnn_type == 'LSTM'):
    args.recurrent = True
    args.rnn_type = 'LSTM'

if not hasattr(args,'comm_action_zero'):
    args.comm_action_zero = False

if not hasattr(args,'comm_action_one'):
    args.comm_action_one = False

parse_action_args(args)

if args.seed == -1:
    args.seed = np.random.randint(0,10000)
torch.manual_seed(args.seed)
# torch.manual_seed(158)
print(args)
print(args.seed)

if args.commnet:
    policy_net = CommNetMLP(args, num_inputs, train_mode=False)
elif args.random:
    policy_net = Random(args, num_inputs)

# this is what we are working with for IC3 Net predator prey.
elif args.recurrent:
    policy_net = RNN(args, num_inputs)
else:
    policy_net = MLP(args, num_inputs)

load(args.load)
policy_net.eval()
if not args.display:
    display_models([policy_net])

# share parameters among threads, but not gradients
for p in policy_net.parameters():
    p.data.share_memory_()

evaluator = Evaluator(args, policy_net, data.init(args.env_name, args))

st_time = time.time()

all_stats = []


all_comms_to_full = {}
all_comms_to_act = {}
all_comms_to_loc = {}
total_comm_actions = np.zeros((50,args.max_steps))
t_delta = 5
for i in range(100):
    ep, stat, all_comms, comms_to_loc, comms_to_act, comms_to_full, comm_action_episode = evaluator.run_episode()
    all_stats.append(stat)

    if args.env_name == 'predator_prey':
        ## Huao: Augment predator locations with if prey in FOV
        for k, v in comms_to_full.items():
            prey_locs = comms_to_loc[k]
            actions = comms_to_act[k]
            np_k = k
            if all_comms_to_full.get(np_k) is None:
                all_comms_to_full[np_k] = {}
            matching_vals = all_comms_to_full.get(np_k)
            for val,prey_loc, action in zip(v,prey_locs,actions):
                prey_observed = abs(val[0]-prey_loc[0]) <=args.vision and abs(val[1]-prey_loc[1]) <=args.vision
                # loc_tuple = (val[0], val[1], int(prey_observed), action)
                loc_tuple =  (val[0],val[1],int(prey_observed),prey_loc[0],prey_loc[1],action)
                if loc_tuple not in matching_vals.keys():
                    matching_vals[loc_tuple] = 0
                matching_vals[loc_tuple] += 1


        for k, v in comms_to_loc.items():
            predator_locs = comms_to_full[k]
            np_k = k
            if all_comms_to_loc.get(np_k) is None:
                all_comms_to_loc[np_k] = {}
            matching_vals = all_comms_to_loc.get(np_k)
            for val,predator_loc in zip(v,predator_locs):
                prey_observed = abs(val[0] - predator_loc[0]) <= args.vision and abs(val[1] - predator_loc[1]) <= args.vision
                loc_tuple = (val[0], val[1], int(prey_observed))
                if loc_tuple not in matching_vals.keys():
                    matching_vals[loc_tuple] = 0
                matching_vals[loc_tuple] += 1
    elif args.env_name == 'mini_dragon':
        for k, v in comms_to_full.items():
            np_k = k
            if all_comms_to_full.get(np_k) is None:
                all_comms_to_full[np_k] = {}
            matching_vals = all_comms_to_full.get(np_k)
            for val in v:
                if val not in matching_vals.keys():
                    matching_vals[val] = 0
                matching_vals[val] += 1

# for action_level, all_comms_to_loc in enumerate([all_comms_to_loc0, all_comms_to_loc1, all_comms_to_loc2]):
# for action_level, all_comms_to_loc in enumerate([all_comms_to_loc0]):
    # print("All comms to loc", all_comms_to_loc)
# print("action level", action_level)
total_episode_time = time.time() - st_time
average_stat = {}
for key in all_stats[0].keys():
    average_stat[key] = np.mean([stat.get(key) for stat in all_stats])
print("average stats is: ", average_stat)
print("time taken per step ", total_episode_time / stat['num_steps'])


directory = os.path.join('figs','neurips',args.exp_name)
try:
    os.makedirs(directory)
except FileExistsError:
    print(f"Directory '{directory}' already exists.")
except PermissionError:
    print(f"Permission denied. Unable to create directory '{directory}'.")

matrix = np.array(list(all_comms_to_full.keys()))
#matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)

clustering = DBSCAN(eps = 3, min_samples = 10)
#clustering = DBSCAN(eps = 0.65, min_samples = 8)
#clustering = KMeans(n_clusters=8, random_state=0, n_init="auto")
clustering.fit(matrix)
cluster_labels = clustering.labels_
num_cluster = max(cluster_labels)+1

print(num_cluster)
# Create a t-SNE model and transform the data
tsne = TSNE(n_components=2, perplexity=15, random_state=28, init='random', learning_rate=200)
vis_dims = tsne.fit_transform(matrix)

x = [x for x, y in vis_dims]
y = [y for x, y in vis_dims]

for category in range(num_cluster):
    xs = np.array(x)[cluster_labels == category]
    ys = np.array(y)[cluster_labels == category]
    if category <20:
        color = plt.get_cmap("tab20")(category)
    else:
        color =plt.get_cmap("tab20")(category-20)

    plt.scatter(xs, ys, color=color, alpha=0.3)

    # avg_x = x[components[category]]
    # avg_y = y[components[category]]
    #
    # plt.scatter(avg_x, avg_y, marker="x", color=color, s=100)
plt.title("Clusters identified visualized in language 2d using t-SNE")
plt.tight_layout()
#plt.savefig(os.path.join('figs','neurips',args.exp_name,'comm_space.png'))
plt.show()
plt.close()


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from ast import literal_eval

def decode_comm(vec,offline_data):
    sim = offline_data['embedding'].apply(lambda x: cosine_similarity(x,vec)).values
    max_ind = np.argmax(sim)
    return offline_data.iloc[max_ind]['crop_comm']

def bleu(references, candidate):

    references_tokenized = [word_tokenize(ref.lower()) for ref in references]
    candidate_tokenized = word_tokenize(candidate.lower())

    # Calculate BLEU score
    bleu_score = sentence_bleu(references_tokenized, candidate_tokenized, smoothing_function=SmoothingFunction().method4)
    return bleu_score
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



offline_data = pd.read_csv(args.data_path)
offline_data["embedding"] = offline_data.ada_embedding.apply(literal_eval).apply(np.array)

for category in range(num_cluster):
    sub_matrix = matrix[cluster_labels == category]
    grid0 = np.zeros((5, 6))
    grid1 = np.zeros((4,1))
    grid2 = np.zeros((2,5))
    # grid = np.zeros((args.dim, args.dim))
    # print("locs items", locs.items())
    total_count0 = 0
    total_count1 = 0
    print(category)
    for comm in sub_matrix:
        loc_tuples = all_comms_to_full[tuple(comm)]

        for loc_tuple in loc_tuples.keys():
            count = loc_tuples[loc_tuple]

            print(decode_comm(comm,offline_data))
            print(loc_tuple,count)
            grid0[loc_tuple[0], loc_tuple[1]] += count
            grid1[loc_tuple[2],0] += count
            k = loc_tuple[5] % 5
            p = loc_tuple[5] // 5
            grid2[p,k] += count





    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    # print("protos np ", protos_np)
    for c in range(num_cluster):
        xs = np.array(x)[cluster_labels== c]
        ys = np.array(y)[cluster_labels== c]
        avg_x = xs.mean()
        avg_y = ys.mean()
        colorway = 'red' if c == category else 'gray'
        ax[0].scatter(avg_x, avg_y, marker="x", color=colorway, s=25)

    max = np.max([grid0.max(), grid1.max(), grid2.max()])
    sns.heatmap(grid0, ax=ax[1],cmap = 'mako',
                vmin=0, vmax=max)
    sns.heatmap(grid1, ax=ax[2],
                vmin=0, vmax=max)
    sns.heatmap(grid2, ax=ax[3],
                vmin=0, vmax=max)
    ax[1].set_title('Location')
    ax[2].set_title('Surface Color')
    ax[3].set_title('Action')

    plt.tight_layout()
    # plt.savefig(
    #     "figs/customize_embeddings_256_llm/kmean_cluster_" + str(
    #         category))
    #plt.savefig(os.path.join('figs', 'neurips',args.exp_name, '_cluster_'+str(category)))
    plt.show()
    plt.close()
#


# def get_weighted_loc(_loc_dict):
#     total_count = 0
#     summed = np.zeros(2)
#     for loc, count in _loc_dict.items():
#         if loc[2]!=1:
#             continue
#         summed += count * np.asarray([loc[0],loc[1]])
#         total_count += count
#     if total_count != 0:
#         return summed / total_count
#     return summed, total_count
#
#


#
# # Lastly, compute a metric of correlation between distance in comm space and distance in grid.
# proto_dists = []
# space_dists = []
# weights = []
# for proto1, locs1 in all_comms_to_loc.items():
#     for proto2, locs2 in all_comms_to_loc.items():
#         if np.array_equal(proto1, proto2):
#             continue
#         # proto_dist = np.linalg.norm(np.asarray(proto1) - np.asarray(proto2)) / (np.sqrt(args.comm_dim))
#         proto_dist = cosine_similarity(np.asarray(proto1), np.asarray(proto2))
#         avg1, weight1 = get_weighted_loc(locs1)
#         avg2, weight2 = get_weighted_loc(locs2)
#         space_dist = np.linalg.norm(avg1 - avg2) / (4 * np.sqrt(2))
#         proto_dists.append(proto_dist)
#         space_dists.append(space_dist)
#         weights.append(weight1+weight2)
#
# proto_dists = np.array(proto_dists).reshape(-1, 1)
#
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# import numpy as np
# import scipy.stats as stats
#
#
# # Create the linear regression object
# reg = LinearRegression()
#
# # Fit the model with sample weights
# reg.fit(proto_dists, space_dists, sample_weight=weights)
#
#
# # Calculate R-squared
# r_squared = reg.score(proto_dists, space_dists, sample_weight=weights)
#
#
#
# print("Coefficients:", reg.coef_)
# print("Intercept:", reg.intercept_)
# print("R-squared:", r_squared)






#
# total_sim = []
# total_count = []
# total_bleu = []
#
#
#
# for comm, locs in all_comms_to_full.items():
#     for loc, count in locs.items():
#         sim = 0
#         num = 0
#         references = []
#         # predator_y, predator_x, prey_in_fov, action = loc
#         predator_y, predator_x, prey_in_fov,prey_y,prey_x, action = loc
#         llm_coms = offline_data[(offline_data['predator_y'] == predator_y) & (offline_data['predator_x'] == predator_x) & (offline_data['prey_in_fov'] == prey_in_fov) & (offline_data['action'] == action)]
#         # llm_coms = offline_data[
#         #     (offline_data['predator_y'] == predator_y) & (offline_data['predator_x'] == predator_x) & (
#         #             offline_data['prey_y'] == prey_y) & (offline_data['prey_x'] == prey_x)& (offline_data['action'] == action)]
#
#         if llm_coms.shape[0] >0:
#             for i, row in llm_coms.iterrows():
#                 embedding = row["embedding"]
#                 sim += cosine_similarity(comm, embedding)
#                 num += 1
#                 references.append(row['crop_comm'])
#             candidate = decode_comm(comm, offline_data)
#
#             total_bleu.append(bleu(references, candidate))
#             total_sim.append(sim / num)
#             total_count.append(count)
#
# print('Cosine similarity:', np.average(total_sim, weights=total_count))
# print('BLEU score:', np.average(total_bleu, weights=total_count))
