from inspect import getargspec
import data
import numpy as np
import torch

class Contrastive(object):
    '''
    Takes a policy trajectory and random trajectory for each agent and computes
    a contrastive binary NCE loss between the two.
    '''
    def __init__(self, args):
        self.args = args
        # env for random rollouts
        self.env = data.init(args.env_name, args, False)
        self.reset()

    def random_rollout(self):
        reset_args = getargspec(self.env.reset).args
        # print(reset_args, " trainer", self.env.reset)
        if 'epoch' in reset_args:
            state = self.env.reset(0, success=False)
        else:
            state = self.env.reset()
        self.random_trajectory.append(state)
        for t in range(self.args.max_steps):
            rr_actions = np.random.randint(0, self.env.env.naction, size=self.args.nagents) # discrete action spaces only
            next_state, _, _, _ = self.env.step([rr_actions])
            self.random_trajectory.append(next_state)
        return

    def update_policy_rollout(self, state, state_message_enc):
        self.policy_trajectory.append(state)
        self.y.append(state_message_enc)
        return

    def get_contrastive_loss(self, policy_net):
        contrast_rand_loss, contrast_future_loss = 0, 0
        for t in range(len(self.policy_trajectory)-1):
            critic_features = self.y[t].squeeze()

            r_obs = self.random_trajectory[t]
            r_enc = policy_net.embed(r_obs).squeeze()
            contrast_rand_loss += torch.log(1-torch.sigmoid(critic_features.T @ r_enc) + 1e-9).mean()

            # randomly selected future state
            t_f = np.random.randint(t, len(self.policy_trajectory)-1)
            f_obs = self.policy_trajectory[t_f]
            f_enc = policy_net.embed(f_obs).squeeze()
            contrast_future_loss += torch.log(torch.sigmoid(critic_features.T @ f_enc) + 1e-9).mean()
        self.reset()
        divisor = len(self.policy_trajectory)-1
        return contrast_rand_loss / divisor, contrast_future_loss / divisor

    def reset(self):
        self.policy_trajectory = []
        self.y = []
        self.random_trajectory = []
