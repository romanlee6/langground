from collections import namedtuple
from inspect import getargspec
import numpy as np
import torch
from torch import optim
import torch.nn as nn
from utils import *
from action_utils import *
import time
from contrast_comm import Contrastive
# from pympler.tracker import SummaryTracker

Transition = namedtuple('Transition', ('state', 'action', 'action_out', 'value', 'episode_mask',
                                       'episode_mini_mask', 'next_state',
                                       'reward', 'misc'))


class Trainer(object):
    def __init__(self, args, policy_net, env, multi=False):
        self.args = args
        self.policy_net = policy_net
        self.env = env
        # print("1trainer", getargspec(self.env.reset).args, self.env.reset)
        self.display = False
        self.last_step = False
        if self.args.optim_name == "RMSprop":
            self.optimizer = optim.RMSprop(policy_net.parameters(),
                lr = args.lrate, alpha=0.97, eps=1e-6)
        elif self.args.optim_name == "Adadelta":
            self.optimizer = optim.Adadelta(policy_net.parameters())#, lr = args.lrate)
        # if self.args.scheduleLR:
        #     self.load_scheduler(start_epoch=0)
        self.params = [p for p in self.policy_net.parameters()]
        # self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = torch.device('cpu')
        if multi:
            self.device = torch.device('cpu')
        print("Device:", self.device)
        self.success_metric = 0
        self.epoch_success = 0
        self.cur_epoch_i = 0

        # traffic junction success
        if self.args.env_name == "traffic_junction":
            if self.args.difficulty == 'easy':
                # self.success_thresh = .90
                self.success_thresh = .97
            elif self.args.difficulty == 'medium':
                self.success_thresh = .86
                # self.success_thresh = .9
            elif self.args.difficulty == 'hard':
                self.success_thresh = .70
        else:
            self.success_thresh = 1.0


        # reward communication when false
        self.args.gating_punish = True

        self.reward_epoch_success = 0
        self.reward_success = 0
        self.cur_reward_epoch_i = 0

        # reward tuning
        self.last_error = None
        self.total_error = None

        # traffic junction curriculum
        self.begin_tj_curric = False
        self.tj_epoch_success = 0
        self.tj_success = 0
        self.tj_epoch_i = 0

        # communication curriculum with hard constraint
        self.min_budget = 0.05
        self.policy_net.budget = self.args.budget
        self.end_comm_curric = True
        self.comm_epoch_i = 0
        self.comm_epoch_success = 0
        self.comm_success = 0

        # if comunication has converged at budget
        self.comm_converge = False
        # self.comm_scheduler = optim.lr_scheduler.ConstantLR(self.optimizer, factor=0.01)
        self.loss_autoencoder = None
        self.loss_min_comm = None
        self.best_model_reward = -np.inf

        #supervised loss from LLM dataset
        self.supervised_loss = None

        self.args.scheduleLR = False
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, max_lr=args.lrate, base_lr=args.lrate/10., step_size_up=args.epoch_size)
        self.beta = 0.01

        if args.contrastive:
            self.contrast_obj = Contrastive(self.args)
            self.contrastive_loss_rand = None
            self.contrastive_loss_future = None

        # self.tracker = SummaryTracker()


    def get_episode(self, epoch, random=False):
        episode = []
        reset_args = getargspec(self.env.reset).args
        # print(reset_args, " trainer", self.env.reset)
        if 'epoch' in reset_args:
            state = self.env.reset(epoch, success=self.begin_tj_curric)
        else:
            state = self.env.reset()
        if self.args.contrastive:
            self.contrast_obj.reset()
            self.contrast_obj.random_rollout()
        should_display = self.display and self.last_step
        if should_display:
            self.env.display()
        stat = dict()
        info_comm = dict()
        switch_t = -1

        # one is used because of the batch size.
        prev_hid = torch.zeros(1, self.args.nagents, self.args.hid_size)
        if self.args.ic3net:
            stat['budget'] = self.policy_net.budget

        # episode_comm = torch.zeros(self.args.nagents)
        if self.args.timmac and not random:
            self.policy_net.reset()
        if random:
            inputs = []
        episode_comm = []
        for t in range(self.args.max_steps):
            # print(t)
            misc = dict()
            if t == 0 and self.args.hard_attn and self.args.commnet and not random:
                info_comm['comm_action'] = np.zeros(self.args.nagents, dtype=int)
                # info_comm['comm_budget'] = np.zeros(self.args.nagents, dtype=int)
                info_comm['step_t'] = t  # episode step for resetting communication budget
                stat['comm_action'] = np.zeros(self.args.nagents, dtype=int)[:self.args.nfriendly]

            # recurrence over time
            if self.args.recurrent and not random:
                if (self.args.rnn_type == 'LSTM' or  self.args.rnn_type == 'GRU') and t == 0:
                    prev_hid = self.policy_net.init_hidden(batch_size=state.shape[0])

                x = [state, prev_hid]
                action_out, value, prev_hid, comm_prob = self.policy_net(x, info_comm)
                if self.args.contrastive:
                    self.contrast_obj.update_policy_rollout(state, self.policy_net.y)
                # episode_comm += comm_action

                # this seems to be limiting how much BPTT happens.
                if (t + 1) % self.args.detach_gap == 0:
                    if self.args.rnn_type == 'LSTM' or self.args.rnn_type == 'GRU':
                        prev_hid = (prev_hid[0].detach(), prev_hid[1].detach())
                    else:
                        prev_hid = prev_hid.detach()
            else:
                x = state
                if random:
                    inputs.append(x)
                action_out, value, comm_prob = self.policy_net(x, info_comm)



            if self.args.vae or self.args.use_vqvib:
                decoded, mu, log_var = self.policy_net.decode()
                if self.loss_autoencoder == None:
                    self.loss_autoencoder = torch.nn.functional.mse_loss(decoded, x[0])
                else:
                    self.loss_autoencoder += torch.nn.functional.mse_loss(decoded, x[0])
                # add KLD
                KLD = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=-1))
                self.loss_autoencoder += self.beta * KLD
            elif self.args.use_compositional:

                comp_loss = self.policy_net.compositional_loss()
                if self.loss_autoencoder == None:
                    self.loss_autoencoder = comp_loss
                else:
                    self.loss_autoencoder += comp_loss
            elif self.args.autoencoder:
                decoded = self.policy_net.decode()
                if self.args.recurrent:
                    # x_all = x[0].reshape(-1).expand_as(decoded)
                    x_all = x[0].sum(dim=1).expand(self.args.nagents, -1).reshape(decoded.shape)
                else:
                    # x_all = x.reshape(-1).expand_as(decoded)
                    x_all = x.sum(dim=1).expand(self.args.nagents, -1).reshape(decoded.shape)
                if self.loss_autoencoder == None:
                    self.loss_autoencoder = torch.nn.functional.mse_loss(decoded, x_all)
                else:
                    self.loss_autoencoder +=torch.nn.functional.mse_loss(decoded, x_all)
            # mask action if not available
            #print(action_out, '\n', self.env.env.get_avail_actions())
            if hasattr(self.env.env, 'get_avail_actions'):
                avail_actions = torch.from_numpy(np.array(self.env.env.get_avail_actions()))
                action_out= action_out + torch.clamp(torch.log(avail_actions), min=-1e10)
                action_out = torch.nn.functional.log_softmax(action_out, dim=-1)
            # this is actually giving you actions from logits
            action = select_action(self.args, action_out)

            # this is for the gating head penalty
            if not self.args.continuous and not self.args.comm_action_one:
                # log_p_a = action_out
                # p_a = [[z.exp() for z in x] for x in log_p_a]
                # gating_probs = p_a[1][0].detach().numpy()
                gating_probs = comm_prob.detach().numpy()

                # since we treat this as reward so probability of 0 being high is rewarded
                # gating_head_rew = np.array([p[1] for p in gating_probs])
                gating_head_rew = gating_probs
                if self.args.min_comm_loss:
                    # print("c prob", comm_prob)
                    episode_comm.append(comm_prob.double().reshape(1,-1))
                    # comm_prob = comm_prob.double()
                    # comm_losses = torch.zeros_like(comm_prob)
                    # ind_budget = np.ones(self.args.nagents) * self.args.max_steps * self.args.soft_budget
                    # ind_budget += np.ones(self.args.nagents) * self.policy_net.get_null_action()
                    # ind_budget = torch.tensor(ind_budget / self.args.max_steps)
                    # comm_losses[comm_prob < ind_budget] = (ind_budget[comm_prob < ind_budget] - comm_prob[comm_prob < ind_budget]) / ind_budget[comm_prob < ind_budget]
                    # comm_losses[comm_prob >= ind_budget] = (comm_prob[comm_prob >= ind_budget] - ind_budget[comm_prob >= ind_budget]) / (1. - ind_budget[comm_prob >= ind_budget])
                    # comm_losses = torch.abs(comm_losses).mean()
                    # if self.loss_min_comm == None:
                    #     self.loss_min_comm = comm_losses
                    # else:
                    #     self.loss_min_comm += comm_losses
                if self.args.gating_head_cost_factor != 0:
                    if self.args.gating_punish:
                        # encourage communication to be at thresh %
                        # thresh = 0.125
                        thresh = self.args.soft_budget
                        Kp = 1.
                        # Kd = 3.2
                        Kd = 1.6
                        Ki = 0.026
                        Kpdi = 1.
                        # 0.05 is the minimum comm rate to ensure success
                        # gating_head_rew[gating_head_rew < 0.05] = 10
                        # error = (gating_head_rew - (0.5*(thresh_top+thresh_bot))) ** 2
                        error = np.zeros_like(gating_head_rew)
                        error[gating_head_rew < thresh] = (thresh - gating_head_rew[gating_head_rew < thresh]) / thresh
                        error[gating_head_rew >= thresh] = (thresh - gating_head_rew[gating_head_rew >= thresh]) / (1. - thresh)
                        if self.last_error is None:
                            self.last_error = error
                        derivative = error - self.last_error
                        if self.total_error is None:
                            self.total_error = np.zeros_like(error)
                        gating_head_rew = Kpdi * np.abs(Kp * error + Kd * derivative + Ki * self.total_error)
                        self.last_error = error
                        self.total_error += error
                        self.total_error = np.clip(self.total_error, -50, 50)
                        # gating_head_rew[gating_head_rew < 0.05] = (gating_head_rew[gating_head_rew < 0.05] - (0.5*(thresh_top+thresh_bot))) ** 2
                        # gating_head_rew[np.logical_and((gating_head_rew <= thresh_top), (gating_head_rew >= thresh_bot))] = 0
                        # gating_head_rew[gating_head_rew > 0.05] = (gating_head_rew[gating_head_rew > 0.05] - (0.5*(thresh_top+thresh_bot))) ** 2
                        # print("here punish", gating_head_rew, gating_probs)
                    else:
                        # encourage communication to be high
                        # gating_head_rew = (gating_head_rew - 1) ** 2
                        # gating_head_rew = (gating_head_rew - self.policy_net.budget) ** 2
                        # punish trying to communicate over budget scaled to [0,1]
                        # print(gating_head_rew, stat['comm_action'] / stat['num_steps'], info['comm_budget'])
                        if self.policy_net.budget != 1:
                            # gating_head_rew = (np.abs(info['comm_action'] - info['comm_budget'])).astype(np.float64)
                            # gating_head_rew = (np.abs(info['comm_action'] - info['comm_budget']) / (1 - self.policy_net.''budget'')).astype(np.float64)
                            # punish excessive and strengthen current communication
                            # gating_head_rew = (np.abs(gating_head_rew - info['comm_budget'])).astype(np.float64)
                            # only punish excessive communication
                            # mask_rew = info['comm_action'] != info['comm_budget']
                            # error = np.zeros_like(gating_head_rew)
                            # error[mask_rew] = np.abs(gating_head_rew[mask_rew] - info['comm_budget'][mask_rew]).astype(np.float64)
                            gating_head_rew = np.abs(gating_head_rew - info['comm_budget']).astype(np.float64)
                            # gating_head_rew = error
                        else:
                            # max communication when budget is full
                            # gating_head_rew = np.abs(info['comm_action'] - 1).astype(np.float64)
                            gating_head_rew = np.abs(gating_head_rew - 1).astype(np.float64)
                        # punish communication under budget scaled to [0,1]
                        # gating_head_rew += np.abs(info['comm_budget'] - self.policy_net.budget) / (self.policy_net.budget)
                        # print("here", gating_head_rew, gating_probs)
                    gating_head_rew *= -1 * np.abs(self.args.gating_head_cost_factor)
                    # try these methods:
                        # A) negative reward when not rate expected
                        # B) positive reward when <= comm rate expected, negative reward when > comm rate
                        # C) adaptive
                            # like A/B but comm rate is adaptive based on success rate
                    stat['gating_reward'] = stat.get('gating_reward', 0) + gating_head_rew
                    # print(gating_head_rew)

            # this converts stuff to numpy
            action, actual = translate_action(self.args, self.env, action)
            # decode intent + observation autoencoder
            if self.args.autoencoder and self.args.autoencoder_action and not random:
                decoded = self.policy_net.decode()
                x_all = torch.zeros_like(decoded)
                if self.args.recurrent:
                    # x_all[:,:-self.args.nagents] = x[0].sum(dim=1).expand(self.args.nagents, -1)
                    x_all[0,:,:-self.args.nagents] = x[0].sum(dim=1).expand(1, self.args.nagents, -1)
                    x_all[0,:,-self.args.nagents:] = torch.tensor(actual[0])
                else:
                    # decoded = decoded.reshape(decoded.shape[0],decoded.shape[1])
                    x_all[:,:-self.args.nagents] = x.reshape(decoded.shape[0], decoded.shape[1]-self.args.nagents)
                    # x_all[0,:,:-self.args.nagents] = x.sum(dim=1).expand(self.args.nagents, -1)
                    x_all[:,-self.args.nagents:] = torch.tensor(actual[0])
                if self.loss_autoencoder == None:
                    self.loss_autoencoder = torch.nn.functional.mse_loss(decoded, x_all)
                else:
                    self.loss_autoencoder += torch.nn.functional.mse_loss(decoded, x_all)


            if self.args.supervised_comm:
                proto_comm= self.policy_net.proto_comm
                if self.args.env_name == "predator_prey":
                    offline_comm = self.policy_net.sample_offline_pp(self.env.env.predator_loc,self.env.env.prey_loc,actual,self.args.vision)
                elif self.args.env_name == "mini_dragon":
                    offline_comm = self.policy_net.sample_offline_dragon(self.env.env.unwrapped.agents,actual)
                if self.supervised_loss == None:
                    self.supervised_loss = torch.nn.functional.cosine_embedding_loss(proto_comm, offline_comm,torch.ones(3))
                else:
                    self.supervised_loss += torch.nn.functional.cosine_embedding_loss(proto_comm, offline_comm,torch.ones(3))

            # comm_budget = info_comm['comm_budget']
            next_state, reward, done, info = self.env.step(actual)

            stat['env_reward'] = stat.get('env_reward', 0) + reward[:self.args.nfriendly]
            if not self.args.continuous and self.args.gating_head_cost_factor != 0:
                if not self.args.variable_gate: # TODO: remove or True later
                    reward += gating_head_rew

            # store comm_action in info for next step
            if self.args.hard_attn and self.args.commnet and not random:
                # info_comm['comm_action'] = action[-1] if not self.args.comm_action_one else np.ones(self.args.nagents, dtype=int)
                info_comm['step_t'] = t
                if self.args.comm_action_zero:
                    info_comm['comm_action'] = np.zeros(self.args.nagents, dtype=int)
                stat['comm_action'] = stat.get('comm_action', 0) + info_comm['comm_action'][:self.args.nfriendly]
                # stat['comm_budget'] = stat.get('comm_budget', 0) + comm_budget[:self.args.nfriendly]
                if hasattr(self.args, 'enemy_comm') and self.args.enemy_comm:
                    stat['enemy_comm']  = stat.get('enemy_comm', 0)  + info_comm['comm_action'][self.args.nfriendly:]


            if 'alive_mask' in info_comm:
                misc['alive_mask'] = info_comm['alive_mask'].reshape(reward.shape)
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

            trans = Transition(state, action, action_out, value, episode_mask, episode_mini_mask,
                               next_state, reward, misc)
            episode.append(trans)
            state = next_state
            if done:
                break
        stat['num_steps'] = t + 1
        stat['steps_taken'] = stat['num_steps']
        if self.args.contrastive:
            contrast_rand_loss, contrast_future_loss = self.contrast_obj.get_contrastive_loss(self.policy_net)
            if self.contrastive_loss_rand == None:
                self.contrastive_loss_rand = contrast_rand_loss
                self.contrastive_loss_future = contrast_future_loss
            else:
                self.contrastive_loss_rand += contrast_rand_loss
                self.contrastive_loss_future += contrast_future_loss

        if self.args.min_comm_loss:
            episode_comm = torch.cat(episode_comm, 0).T
            episode_comm = episode_comm.mean(1)
            # print(episode_comm)
            comm_losses = torch.zeros_like(episode_comm)
            ind_budget = np.ones(self.args.nagents) * self.args.max_steps * self.args.soft_budget
            ind_budget += np.ones(self.args.nagents) * self.policy_net.get_null_action()
            ind_budget = torch.tensor(ind_budget / self.args.max_steps)
            comm_losses[episode_comm < ind_budget] = (ind_budget[episode_comm < ind_budget] - episode_comm[episode_comm < ind_budget]) / ind_budget[episode_comm < ind_budget]
            comm_losses[episode_comm >= ind_budget] = (episode_comm[episode_comm >= ind_budget] - ind_budget[episode_comm >= ind_budget]) / (1. - ind_budget[episode_comm >= ind_budget])
            comm_losses = stat['num_steps'] * torch.abs(comm_losses).mean()
            # print(episode_comm, ind_budget)
            # comm_losses = torch.nn.functional.mse_loss(episode_comm, ind_budget)
            if self.loss_min_comm == None:
                self.loss_min_comm = comm_losses
            else:
                self.loss_min_comm += comm_losses

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
        if random:
            return inputs

        # self.tracker.print_diff()
        return (episode, stat)

    def compute_grad(self, batch, other_stat=None):
        stat = dict()
        num_actions = self.args.num_actions
        # dim_actions = self.args.dim_actions
        dim_actions = 1

        n = self.args.nagents
        batch_size = len(batch.state)
        rewards = torch.Tensor(np.array(batch.reward)).to(self.device)
        episode_masks = torch.Tensor(np.array(batch.episode_mask)).to(self.device)
        episode_mini_masks = torch.Tensor(np.array(batch.episode_mini_mask)).to(self.device)
        actions = torch.Tensor(np.array(batch.action)).to(self.device)
        actions = actions.transpose(1, 2).view(-1, n, dim_actions)

        # old_actions = torch.Tensor(np.concatenate(batch.action, 0))
        # old_actions = old_actions.view(-1, n, dim_actions)
        # print(old_actions == actions)

        # can't do batch forward.
        values = torch.cat(batch.value, dim=0).to(self.device)
        action_out = list(zip(*batch.action_out))
        action_out = [torch.cat(a, dim=0).to(self.device) for a in action_out]

        alive_masks = torch.Tensor(np.concatenate([item['alive_mask'] for item in batch.misc])).view(-1).to(self.device)

        coop_returns = torch.Tensor(batch_size, n).to(self.device)
        ncoop_returns = torch.Tensor(batch_size, n).to(self.device)
        returns = torch.Tensor(batch_size, n).to(self.device)
        deltas = torch.Tensor(batch_size, n).to(self.device)
        advantages = torch.Tensor(batch_size, n).to(self.device)
        values = values.view(batch_size, n)

        prev_coop_return = 0
        prev_ncoop_return = 0
        prev_value = 0
        prev_advantage = 0
        for i in reversed(range(rewards.size(0))):
            coop_returns[i] = rewards[i] + self.args.gamma * prev_coop_return * episode_masks[i]
            ncoop_returns[i] = rewards[i] + self.args.gamma * prev_ncoop_return * episode_masks[i] * episode_mini_masks[i]

            prev_coop_return = coop_returns[i].clone()
            prev_ncoop_return = ncoop_returns[i].clone()

            returns[i] = (self.args.mean_ratio * coop_returns[i].mean()) \
                        + ((1 - self.args.mean_ratio) * ncoop_returns[i])

        advantages = returns - values.data
        # print(advantages, returns, values.data,"\n")
        if self.args.normalize_rewards:
            advantages = (advantages - advantages.mean()) / advantages.std()

        if self.args.continuous:
            action_means, action_log_stds, action_stds = action_out
            log_prob = normal_log_density(actions, action_means, action_log_stds, action_stds)
        else:
            log_p_a = [action_out[i].view(-1, num_actions[i]) for i in range(dim_actions)]
            actions = actions.contiguous().view(-1, dim_actions)

            if self.args.advantages_per_action:
                log_prob = multinomials_log_densities(actions, log_p_a)
            else:
                log_prob = multinomials_log_density(actions, log_p_a)

        if self.args.advantages_per_action:
            action_loss = -advantages.view(-1).unsqueeze(-1) * log_prob
            action_loss *= alive_masks.unsqueeze(-1)
        else:
            action_loss = -advantages.view(-1) * log_prob.squeeze()
            action_loss *= alive_masks

        action_loss = action_loss.sum()
        stat['action_loss'] = action_loss.item()

        # value loss term
        targets = returns
        value_loss = (values - targets).pow(2).view(-1)
        value_loss *= alive_masks
        value_loss = value_loss.sum()

        stat['value_loss'] = value_loss.item()
        # adding regularization term to minimize communication
        loss = action_loss + self.args.value_coeff * value_loss
        if self.args.max_info:
            loss += self.args.eta_info * 0   # TODO: add euclidean distance between memory cells

        if not self.args.continuous:
            # entropy regularization term
            entropy = 0
            for i in range(len(log_p_a)):
                entropy -= (log_p_a[i] * log_p_a[i].exp()).sum()
            stat['entropy'] = entropy.item()
            if self.args.entr > 0:
                loss -= self.args.entr * entropy

        if self.args.min_comm_loss:
            self.loss_min_comm *= self.args.eta_comm_loss
            stat['regularization_loss'] = self.loss_min_comm.item()
            loss += self.loss_min_comm
        if self.args.supervised_comm:
            stat['supervised_loss'] = self.supervised_loss.item()
            loss = loss + self.args.supervised_gamma * self.supervised_loss


        # print("compute_grad 2.c.")
        if self.args.autoencoder or self.args.vae or self.args.use_vqvib or self.args.use_compositional:
            stat['autoencoder_loss'] = self.loss_autoencoder.item()
            loss = 0.5 * loss + 0.5 * self.loss_autoencoder
        if self.args.contrastive:
            stat['contrastive_loss_rand'] = 0.1*self.contrastive_loss_rand.item()
            stat['contrastive_loss_future'] = 0.1*self.contrastive_loss_future.item()
            loss += 0.1*(self.contrastive_loss_rand + self.contrastive_loss_future)
        # print("compute_grad 2.d.")
        loss.backward()
        # print("compute_grad 2.e.")
        if self.args.autoencoder or self.args.vae or self.args.use_vqvib or self.args.use_compositional:
            self.loss_autoencoder = None
        if self.args.contrastive:
            self.contrastive_loss_rand = None
            self.contrastive_loss_future = None
        if self.args.min_comm_loss:
            self.loss_min_comm = None
        if self.args.supervised_comm:
            self.supervised_loss = None

        # print("compute_grad 3")

        # self.counter = 0
        # self.summer = 0
        # self.summer1 = 0
        return stat

    def run_batch(self, epoch):
        # self.reward_curriculum(epoch)
        if epoch >= 250 and self.args.use_tj_curric:
            self.begin_tj_curric = True
        self.epoch_num = epoch
        batch = []
        self.stats = dict()
        self.stats['num_episodes'] = 0
        while len(batch) < self.args.batch_size:
            if self.args.batch_size - len(batch) <= self.args.max_steps:
                self.last_step = True
            episode, episode_stat = self.get_episode(epoch)
            merge_stat(episode_stat, self.stats)
            self.stats['num_episodes'] += 1
            batch += episode

        self.last_step = False
        self.stats['num_steps'] = len(batch)
        self.stats['learning_rate'] = self.get_lr(self.optimizer)
        batch = Transition(*zip(*batch))
        return batch, self.stats

    # only used when nprocesses=1
    def train_batch(self, epoch):
        # run_st_time = time.time()
        batch, stat = self.run_batch(epoch)

        # print(f"time taken for data collection is {time.time() - run_st_time}")

        self.optimizer.zero_grad()

        # grad_st_time = time.time()
        s = self.compute_grad(batch, other_stat=stat)
        # print(f"time taken for grad computation {time.time() - grad_st_time}")

        merge_stat(s, stat)

        for p in self.params:
            if p._grad is not None:
                p._grad.data /= stat['num_steps']
        self.optimizer.step()
        if self.args.scheduleLR:
            # print("LR step")
            self.scheduler.step()
        return stat

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def set_lr(self):
        for param_group in self.optimizer.param_groups:
            # param_group['lr'] = self.args.lrate
            param_group['lr'] = self.args.lrate * 0.01

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state):
        self.optimizer.load_state_dict(state)

    def setup_var_reload(self):
        if self.args.variable_gate:
            self.args.comm_action_one = False
            self.args.variable_gate = False

    # def load_scheduler(self, start_epoch):
    #     print("load_scheduler",start_epoch)
    #     self.scheduler1 = optim.lr_scheduler.ConstantLR(self.optimizer, factor=1)
    #     self.scheduler2 = optim.lr_scheduler.StepLR(self.optimizer, 500*self.args.epoch_size, gamma=0.1)
    #     self.scheduler = optim.lr_scheduler.SequentialLR(self.optimizer, schedulers=[self.scheduler1, self.scheduler2], milestones=[2500*self.args.epoch_size])
