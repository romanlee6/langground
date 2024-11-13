from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *
from ic3net_envs import predator_prey_env

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask', 'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))

from math import log10, floor
def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)

class Evaluator:
    def __init__(self, args, policy_net, env):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        self.display = args.display
        self.last_step = False
        self.total_length = 0
        self.total = 0

    def run_episode(self, epoch=1):

        all_comms = []
        episode = []
        reset_args = getargspec(self.env.reset).args
        if 'epoch' in reset_args:
            state = self.env.reset(epoch)
        else:
            state = self.env.reset()
        should_display = self.display  # and self.last_step

        if should_display:
            self.env.display()
        stat = dict()
        info_comm = dict()
        info= dict()
        switch_t = -1

        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)
        comms_to_prey_loc = {} # record action 0
        comms_to_prey_act = {} # record action 1
        comms_to_loc_full = {} # record all
        comm_action_episode = np.zeros(self.args.max_steps)
        total_length = 0
        total = 0
        for t in range(self.args.max_steps):
            misc = dict()
            info['step_t'] = t
            if t == 0 and self.args.hard_attn and self.args.commnet:
                info_comm['comm_action'] = np.zeros(self.args.nagents, dtype=int)
            # Hardcoded to record communication for agent 1 (prey)
            info['record_comms'] = 1
            # recurrence over time
            if self.args.recurrent:
                if (self.args.rnn_type == 'LSTM' or  self.args.rnn_type == 'GRU') and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

                x = [state, prev_hid]
                action_out, value, prev_hid, proto_comms = self.policy_net(x, info)
                # proto_comms = self.policy_net.message.detach().cpu().numpy()
                # if isinstance(self.env.env.env, predator_prey_env.PredatorPreyEnv):
                if self.args.env_name == 'predator_prey':
                    # tuple_comms = tuple(proto_comms.detach().numpy())
                    for i in range(0, len(self.env.env.predator_loc)):
                        p = self.env.env.predator_loc[i]
                        proto = proto_comms[0][i]
                        tuple_comms = tuple(proto)
                        if comms_to_loc_full.get(tuple_comms) is None:
                            comms_to_loc_full[tuple_comms] = []
                        comms_to_loc_full[tuple_comms].append(tuple(p))
                        if comms_to_prey_loc.get(tuple_comms) is None:
                            comms_to_prey_loc[tuple_comms] = []
                        comms_to_prey_loc[tuple_comms].append(tuple(self.env.env.prey_loc[0]))

                        if hasattr(self.env.env, 'get_avail_actions'):
                            avail_actions = torch.from_numpy(np.array(self.env.env.get_avail_actions()))
                            action_out = action_out + torch.clamp(torch.log(avail_actions), min=-1e10)
                            action_out = torch.nn.functional.log_softmax(action_out, dim=-1)
                        action = select_action(self.args, action_out, eval_mode=True)
                        action, actual = translate_action(self.args, self.env, action)

                        if comms_to_prey_act.get(tuple_comms) is None:
                            comms_to_prey_act[tuple_comms] = []
                        comms_to_prey_act[tuple_comms].append(action[0][i])

                elif self.args.env_name == 'traffic_junction':
                    # print("car loc", self.env.env.car_loc)
                    # print("paths", self.env.env.car_loc)

                    for i in range(0, len(self.env.env.car_loc)):
                        p = self.env.env.car_loc[i]
                        # print(p)
                        # continue
                        # print(proto_comms.shape, len(self.env.env.car_loc))
                        proto = proto_comms[i]
                        action_i = self.env.env.car_last_act[i]
                        if self.env.env.car_route_loc[i] != -1:
                            # if p[0] == 0 and p[1] == 0 or info_comm['comm_action'][i] == 0 or self.policy_net.get_null_action()[i] == 1:
                            if p[0] == 0 and p[1] == 0 or info_comm['comm_action'][i] == 0:# or self.policy_net.get_null_action()[i] == 1:
                                continue
                            # print("path", p, proto.shape)
                            # print(t, "proto", proto, proto.shape)
                            # print(info_comm['comm_action'][i])
                            if self.args.use_compositional:
                                proto = proto.reshape(-1, self.policy_net.composition_dim)
                                full_proto_bool = False
                                if full_proto_bool:
                                    full_proto = []
                                    for j in range(len(proto)):
                                        full_proto.append(round_sig(proto[j][0], sig=2))
                                        full_proto.append(round_sig(proto[j][1], sig=2))
                                    tuple_comms = tuple(full_proto)
                                    if comms_to_loc_full.get(tuple_comms) is None:
                                        comms_to_loc_full[tuple_comms] = []
                                    comms_to_loc_full[tuple_comms].append(tuple(p))

                                else:
                                    unique = {}
                                    for j in range(len(proto)):
                                        total += 1
                                        tuple_comms = tuple([round_sig(proto[j][0], sig=2), round_sig(proto[j][1], sig=2)])
                                        if unique.get(tuple_comms) is None:
                                            unique[tuple_comms] = True
                                            total_length += 1
                                        # tuple_comms = tuple(proto[j])
                                        if comms_to_loc_full.get(tuple_comms) is None:
                                            comms_to_loc_full[tuple_comms] = []
                                        comms_to_loc_full[tuple_comms].append(tuple(p))

                            else:
                                for index in range(len(proto)):
                                    proto[index] = round_sig(proto[index], sig=2)
                                tuple_comms = tuple(proto)
                                if comms_to_loc_full.get(tuple_comms) is None:
                                    comms_to_loc_full[tuple_comms] = []
                                comms_to_loc_full[tuple_comms].append(tuple(p))
                            # print(action_i)
                            if action_i == 0:
                                if comms_to_prey_loc.get(tuple_comms) is None:
                                    comms_to_prey_loc[tuple_comms] = []
                                # print("path", self.env.env.chosen_path[0])
                                comms_to_prey_loc[tuple_comms].append(tuple(p))
                            else:
                                if comms_to_prey_act.get(tuple_comms) is None:
                                    comms_to_prey_act[tuple_comms] = []
                                comms_to_prey_act[tuple_comms].append(tuple(p))


                if (t + 1) % self.args.detach_gap == 0:
                    if (self.args.rnn_type == 'LSTM' or  self.args.rnn_type == 'GRU'):
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                action_out, value, h, prob = self.policy_net(x, info_comm)
                proto_comms = self.policy_net.encoded_info
                # if isinstance(self.env.env.env, predator_prey_env.PredatorPreyEnv):
                if self.args.env_name == 'predator_prey':
                    tuple_comms = tuple(proto_comms.detach().numpy())
                    if comms_to_prey_loc.get(tuple_comms) is None:
                        comms_to_prey_loc[tuple_comms] = []
                    comms_to_prey_loc[tuple_comms].append(tuple(self.env.env.env.prey_loc[0]))
                elif self.args.env_name == 'traffic_junction':
                    # print("car loc", self.env.env.car_loc)
                    # print("paths", self.env.env.car_loc)
                    for i in range(0, len(self.env.env.car_loc)):
                        p = self.env.env.car_loc[i]
                        # print(p)
                        proto = proto_comms[0][i]
                        action_i = self.env.env.car_last_act[i]
                        if self.env.env.car_route_loc[i] != -1:
                            # print("path", p, proto.shape)
                            tuple_comms = tuple(proto)
                            # print("tuple comms", proto.shape)
                            if comms_to_loc_full.get(tuple_comms) is None:
                                comms_to_loc_full[tuple_comms] = []
                            comms_to_loc_full[tuple_comms].append(tuple(p))

                            # print(action_i)
                            if action_i == 0:
                                if comms_to_prey_loc.get(tuple_comms) is None:
                                    comms_to_prey_loc[tuple_comms] = []
                                # print("path", self.env.env.chosen_path[0])
                                comms_to_prey_loc[tuple_comms].append(tuple(p))
                            else:
                                if comms_to_prey_act.get(tuple_comms) is None:
                                    comms_to_prey_act[tuple_comms] = []
                                comms_to_prey_act[tuple_comms].append(tuple(p))


            # if hasattr(self.env.env, 'get_avail_actions'):
            #     avail_actions = np.array(self.env.env.get_avail_actions())
            #     action_mask = avail_actions==np.zeros_like(avail_actions)
            #     action_out[0, action_mask] = -1e10
            #     action_out = torch.nn.functional.log_softmax(action_out, dim=-1)
            if hasattr(self.env.env, 'get_avail_actions'):
                avail_actions = torch.from_numpy(np.array(self.env.env.get_avail_actions()))
                action_out= action_out + torch.clamp(torch.log(avail_actions), min=-1e10)
                action_out = torch.nn.functional.log_softmax(action_out, dim=-1)
            action = select_action(self.args, action_out, eval_mode=True)
            action, actual = translate_action(self.args, self.env, action)

            if self.args.env_name == 'mini_dragon':
            # print("car loc", self.env.env.car_loc)
            # print("paths", self.env.env.car_loc)
                for index, agent_id in enumerate(self.env.env.agents):
                    id2index = {'alpha': 0, 'bravo': 1, 'charlie': 2}
                    key_tuple = {}
                    agent = self.env.env.agents[agent_id]
                    roomIndexDict = {0: 0, 3: 1, 5: 2, 6: 3, 8: 4}
                    room = roomIndexDict[agent.node.id]
                    if agent.bomb is not None:
                        bomb = agent.bomb.id
                        sequence = list(agent.bomb._full_sequence[agent.bomb._current_step:])
                    else:
                        bomb = -1
                        sequence = []
                    while len(sequence) < 3:
                        sequence.append(-1)

                    agent_id = {0:'alpha',1:'bravo',2:'charlie'}[index]

                    proto = proto_comms[0][index]
                    action_i = actual[0][index]

                    obs_tuple = (room, bomb, int(sequence[0]), int(sequence[1]), int(sequence[2]), action_i)
                    #
                    # for index in range(len(proto)):
                    #     proto[index] = round_sig(proto[index], sig=2)
                    tuple_comms = tuple(proto)
                    if comms_to_loc_full.get(tuple_comms) is None:
                        comms_to_loc_full[tuple_comms] = []
                    comms_to_loc_full[tuple_comms].append(obs_tuple)
                    # print(action_i)

                    if comms_to_prey_loc.get(tuple_comms) is None:
                        comms_to_prey_loc[tuple_comms] = []
                    # print("path", self.env.env.chosen_path[0])
                    comms_to_prey_loc[tuple_comms].append(obs_tuple[0])

                    if comms_to_prey_act.get(tuple_comms) is None:
                        comms_to_prey_act[tuple_comms] = []
                    comms_to_prey_act[tuple_comms].append(action_i)

            next_state, reward, done, info = self.env.step(actual)
            if self.args.env_name == 'traffic_junction':
                done = done or self.env.env.has_failed
            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet:
                # info_comm['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)
                # print(info_comm['comm_action'][0])
                comm_action_episode[t] += info_comm['comm_action'][0]
                # print("before ", stat.get('comm_action', 0), info_comm['comm_action'][:self.args.nfriendly])
                stat['comm_action'] = stat.get('comm_action', 0) + info_comm['comm_action'][:self.args.nfriendly]
                all_comms.append(info_comm['comm_action'][:self.args.nfriendly])
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info_comm['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info:
                misc['alive_mask'] = info['alive_mask'].reshape(reward.shape)
            else:
                misc['alive_mask'] = np.ones_like(reward)

            # env should handle this make sure that reward for dead agents is not counted
            # reward = reward * misc['alive_mask']

            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]

            done = done or t == self.args.max_steps - 1

            episode_mask = np.ones(reward.shape)
            episode_mini_mask = np.ones(reward.shape)

            if done:
                episode_mask = np.zeros(reward.shape)
            else:
                if 'is_completed' in info:
                    episode_mini_mask = 1 - info['is_completed'].reshape(-1)

            if should_display:
                self.env.display()

            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask, next_state, reward, misc)
            episode.append(trans)
            state = next_state
            if done:
                break
        self.total_length += total_length
        self.total += total

        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']

        if hasattr(self.env, 'reward_terminal'):
            reward = self.env.reward_terminal()
            # We are not multiplying in case of reward terminal with alive agent
            # If terminal reward is masked environment should do
            # reward = reward * misc['alive_mask']

            episode[-1] = episode[-1]._replace(reward = episode[-1].reward + reward)
            stat['reward'] = stat.get('reward', 0) + reward[:self.args.nfriendly]
            if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                stat['enemy_reward'] = stat.get('enemy_reward', 0) + reward[self.args.nfriendly:]


        if hasattr(self.env, 'get_stat'):
            merge_stat(self.env.get_stat(), stat)
        return episode, stat, all_comms, comms_to_prey_loc, comms_to_prey_act, comms_to_loc_full, comm_action_episode
