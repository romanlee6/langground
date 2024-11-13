import sys
import time
import signal
import argparse
import time, os
from collections import namedtuple
#os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(1,'../')
import numpy as np
import torch
import data
from models import *
from comm import CommNetMLP
from utils import *
from action_utils import *
from evaluator import Evaluator
from args import get_args
from inspect import getargspec
from LLM.pp_llm import *
import pandas as pd
from ast import literal_eval


from openai import OpenAI

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')


Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))


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
if args.display:
    env.env.init_curses()
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


all_comms = []
episode = []
# epoch = 3



stat = dict()
info_comm = dict()
info = dict()
switch_t = -1

comm_action_episode = np.zeros(args.max_steps)

prev_hid = torch.zeros(1, args.nagents, args.hid_size)

done = False
step = 0
history = []
moveRT = []


"""
MARL part
"""


# text_env.init_args(parser)
# args = parser.parse_args()
# args.dim = 5
# args.vision = 1
# args.seed = 244
# args.mode = 'cooperative'
# args.save_path = 'data/pp_prompt/'
args.model = 'gpt-4-turbo-preview'
args.temperature = 0
# args.exp_name = 'comm'
args.allow_comm = False
args.belief = False
args.nfriendly = args.nagents
args.save_path = 'exp_log_entropy'



agents = []
for i in range(args.nagents):
    agent = GPTAgent(args, i)
    agents.append(agent)

replace_agent_ids = [0]

data_path = args.data_path
offline_data = pd.read_csv(data_path)
offline_data["embedding"] = offline_data.ada_embedding.apply(literal_eval).apply(np.array)

text_comms = ['','','']
info['replace_action'] = []


client = OpenAI(api_key = 'na')

def word_embed(text, embeddings, model="text-embedding-3-large"):
    #crop comm
    text = text.split('.')[0]
    text = text.replace(',',';')
    e = embeddings[embeddings['crop_comm'] == text]
    if e.shape[0] > 0:
        return e.iloc[0]['embedding']
    else:
        if text == '':
            text = ' '
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model, dimensions=256).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def decode_comm(vec,offline_data):
    vec = vec.detach().numpy()
    sim = offline_data['embedding'].apply(lambda x: cosine_similarity(x,vec)).values
    max_ind = np.argmax(sim)
    return offline_data.iloc[max_ind]['crop_comm']




for epoch in range(12,20):
    DATA_PATH = os.path.join(args.save_path, args.model, args.exp_name, 'seed' + str(args.seed),'1_LLM', str(epoch)) + '/'
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    reset_args = getargspec(env.reset).args
    for i in range(epoch):
        if 'epoch' in reset_args:
            state = env.reset(epoch)
        else:
            state = env.reset()

    for a in agents:
        a.reset()

    if args.display:
        env.env.render()
    step = 0
    done = False
    text_comms = ['','','']
    for t in range(args.max_steps):
        for i in range(args.nfriendly):
            print(i, "text_comms[i]")
        ##LLM part
        text_obs = env.env.translate_observation_to_description(state[0], args.allow_comm, text_comms)
        text_comms = []

        info['replace_action'] = []
        info['agent_id_replace'] = []
        info['llm_comm'] = []

        last_loc = env.env.predator_loc.copy()

        for i in replace_agent_ids:
            action, comm = agents[i].step(text_obs[i])

            text_comms.append(comm)
            # explanations.append(exp)

            info['replace_action'].append(agents[i].decode_action(action))

            info['agent_id_replace'].append(i)
            info['llm_comm'].append(torch.Tensor(word_embed(text_comms[i], offline_data)))
            info['replace_comm'] = True



        ## MARL part
        misc = dict()
        info['step_t'] = t
        if t == 0 and args.hard_attn and args.commnet:
            info_comm['comm_action'] = np.zeros(args.nagents, dtype=int)
        # Hardcoded to record communication for agent 1 (prey)
        info['record_comms'] = 1


        # recurrence over time
        if args.recurrent:
            if (args.rnn_type == 'LSTM' or args.rnn_type == 'GRU') and t == 0:
                prev_hid = policy_net.init_hidden(batch_size=state.shape[0])

            x = [state, prev_hid]
            action_out, value, prev_hid, proto_comms = policy_net(x, info)
            if (t + 1) % args.detach_gap == 0:
                if (args.rnn_type == 'LSTM' or args.rnn_type == 'GRU'):
                    prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                else:
                    prev_hid = prev_hid.detach()
        else:
            x = state
            action_out, value, h, prob = policy_net(x, info_comm)
            proto_comms = policy_net.encoded_info

        probs = torch.exp(action_out)
        entropy = -torch.sum(probs * action_out, dim=-1).detach().numpy()


        if hasattr(env.env, 'get_avail_actions'):
            avail_actions = torch.from_numpy(np.array(env.env.get_avail_actions()))
            action_out = action_out + torch.clamp(torch.log(avail_actions), min=-1e10)
            action_out = torch.nn.functional.log_softmax(action_out, dim=-1)
        action = select_action(args, action_out, eval_mode=True)
        action, actual = translate_action(args, env, action)

        ## replace action with LLM output
        # print(actual, 'before')
        for i , act in enumerate(info['replace_action']):
            actual[0][i] = act
        # print(actual, 'after')

        next_state, reward, done, info = env.step(actual)


        # proto_comms = policy_net.proto_comm
        #
        # if len(replace_agent_ids) > 0:
        #     j = replace_agent_ids[-1] + 1
        # else:
        #     j = 0
        # while j < args.nfriendly:
        #     text_comms.append(decode_comm(proto_comms[j],offline_data))
        #     j+=1



        if args.hard_attn and args.commnet:
            # info_comm['comm_action'] = action[-1] if not args.comm_action_one else np.ones(args.nagents, dtype=int)
            # print(info_comm['comm_action'][0])
            comm_action_episode[t] += info_comm['comm_action'][0]
            # print("before ", stat.get('comm_action', 0), info_comm['comm_action'][:args.nfriendly])
            stat['comm_action'] = stat.get('comm_action', 0) + info_comm['comm_action'][:args.nfriendly]
            all_comms.append(info_comm['comm_action'][:args.nfriendly])
            if hasattr(args, 'enemy_comm') and args.enemy_comm:
                stat['enemy_comm'] = stat.get('enemy_comm', 0) + info_comm['comm_action'][args.nfriendly:]

        if 'alive_mask' in info:
            misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
        else:
            misc['alive_mask'] = np.ones_like(reward)

        # env should handle this make sure that reward for dead agents is not counted
        # reward = reward * misc['alive_mask']

        stat['reward'] = stat.get('reward', 0) + reward[:args.nfriendly]
        if hasattr(args, 'enemy_comm') and args.enemy_comm:
            stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[args.nfriendly:]

        done = done or t == args.max_steps - 1

        episode_mask = np.ones(reward.shape)
        episode_mini_mask = np.ones(reward.shape)

        if done:
            episode_mask = np.zeros(reward.shape)
        else:
            if 'is_completed' in info:
                episode_mini_mask = 1 - info['is_completed'].reshape(-1)

        for i in range(args.nagents):
            #text_comms[i]
            summary = {'step': step, 'agent_id': i, 'obs': text_obs[i], 'action': actual[0][i], 'comm': 'text_comms[i]','entropy':entropy[0][i],'reached_prey':env.env.reached_prey[i],
                    'reward': reward[i], 'prey_loc': env.env.prey_loc[0], 'predator_loc': last_loc[i]}
            print(summary)
            with open(DATA_PATH + 'summary.csv', 'a+', encoding='utf-8') as f:
                for k, v in summary.items():
                    f.write(str(v).replace(',', ';').replace('\n', ''))
                    f.write(',')
                f.write('\n')


        if args.display:
            # for i in range(args.nfriendly):
            #     print(i,text_comms[i])

            env.env.render()

        step += 1

        trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
        episode.append(trans)
        state = next_state
        if done:
            break

    stat['num_steps'] = t + 1
    stat['steps_taken'] = stat['num_steps']

    if hasattr(env, 'reward_terminal'):
        reward = env.reward_terminal()
        # We are not multiplying in case of reward terminal with alive agent
        # If terminal reward is masked environment should do
        # reward = reward * misc['alive_mask']

        episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
        stat['reward'] = stat.get('reward', 0) + reward[:args.nfriendly]
        if hasattr(args, 'enemy_comm') and args.enemy_comm:
            stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[args.nfriendly:]


    if hasattr(env, 'get_stat'):
        merge_stat(env.get_stat(), stat)
