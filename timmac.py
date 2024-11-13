import torch
import torch.nn.functional as F
from torch import nn
import time
from models import SelfAttention
from action_utils import select_action, translate_action
from networks import ProtoNetwork, ProtoLayer
from network_utils import gumbel_softmax
from noise import OUNoise
import numpy as np
import os

class TIMMAC(nn.Module):
    """
    Transformer Information Maximizing Multi-Agent Communication.
    Uses communication vector to communicate info between agents
    """
    def __init__(self, args, num_inputs, train_mode=True):
        """Initialization method for this class, setup various internal networks
        and weights

        Arguments:
            TIMMAC {object} -- Self
            args {Namespace} -- Parse args namespace
            num_inputs {number} -- Environment observation dimension for agents
        """

        super(TIMMAC, self).__init__()
        self.args = args
        self.nagents = args.nagents
        self.hid_size = args.hid_size
        self.comm_passes = args.comm_passes
        self.max_len = min(args.dim, 10)
        self.dropout = 0.

        # embedding for one hot encoding of inputs
        self.embed = nn.Linear(num_inputs, args.comm_dim)

        if self.args.vae:
            self.fc_mu = nn.Linear(args.hid_size, args.hid_size)
            self.fc_var = nn.Linear(args.hid_size, args.hid_size)

        # encode observation and action (intent)
        self.attend_obs_intent = SelfAttention(args.num_heads, args.hid_size, dropout=self.dropout)
        # self.obs_intent_head = nn.Linear(args.hid_size, args.hid_size)
        if args.recurrent:
            self.init_hidden(args.batch_size)
            self.f_module = nn.LSTMCell(args.comm_dim, args.hid_size)
        else:
            self.f_module = nn.Linear(args.hid_size, args.hid_size)

        # attend to communications to determine relevant information
        if args.mha_comm:
            self.attend_comm = SelfAttention(args.num_heads, args.comm_dim, dropout=self.dropout)
        self.comm_head = nn.Linear(args.comm_dim, args.comm_dim)

        # communication and obs/intent aggregation
        # self.aggregate_info = nn.Linear(args.comm_dim + args.hid_size, args.hid_size)

        # obs + intent decoder
        # self.attend_decoder = SelfAttention(args.num_heads, args.hid_size)

        # Decoder for information maximizing communication
        if self.args.autoencoder_action:
            self.decoder_head = nn.Linear(args.hid_size, num_inputs-3 + 4*self.args.nagents)
        elif self.args.autoencoder:
            self.decoder_head = nn.Linear(args.hid_size, num_inputs)
        elif self.args.vae:
            self.decoder_head = nn.Linear(args.hid_size, num_inputs)

        # attend to latent obs/intent/comm to produce action
        # self.attend_action = SelfAttention(args.num_heads, args.hid_size)
        print(args.num_actions)
        self.action_head = nn.Linear(self.hid_size, args.num_actions[0])

        # Gate sparse communication
        self.gating_head = nn.Linear(self.hid_size, 2)

        # Critic
        self.value_head = nn.Linear(self.hid_size, 1)

        # Mask for communication
        if self.args.comm_mask_zero:
            self.comm_mask = torch.zeros(self.nagents, self.nagents)
        else:
            # this just prohibits self communication
            self.comm_mask = torch.ones(self.nagents, self.nagents) \
                            - torch.eye(self.nagents, self.nagents)


        # sequence data
        self.obs_intent_seq = Sequence(self.max_len)

        # positional encoding
        self.pe = PositionalEncoding(args.hid_size, self.nagents, self.max_len, dropout=0)

        self.apply(self.init_weights)
        torch.nn.init.zeros_(self.embed.weight)
        # if self.args.vae:
        #     torch.nn.init.zeros_(self.fc_mu.weight)
        #     torch.nn.init.zeros_(self.fc_var.weight)

    def forward(self, x, info={}):
        """
        Forward function for TIMMAC
        """
        # embed observations
        x, h, cell = self.do_embed(x)
        n = self.nagents

        # update observation_intent sequence
        self.obs_intent_seq.step(x)
        x_oa = self.obs_intent_seq.get().transpose(1,0)

        # positional encoding
        x_oa = self.pe(x_oa)
        # latent observation space
        oa_mask = self.obs_intent_seq.mask()
        # print()
        # print(x_oa.shape, oa_mask.shape)
        # x_oa_skip = x_oa.clone()
        if self.args.recurrent:
            h, cell = self.f_module(x_oa.view(n, self.args.hid_size), (h, cell))
            x_oa = h
        else:
            x_oa = self.attend_obs_intent(x_oa,mask=oa_mask) + x
            if self.args.vae:
                mu = self.fc_mu(x_oa)
                log_var = self.fc_var(x_oa)
                #reparameterize
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                x_oa = eps * std + mu

                self.decoding_mu = mu.squeeze().clone()
                self.decoding_log_var = log_var.squeeze().clone()
            else:
                x_oa = self.f_module(x_oa.view(n, self.args.hid_size))

        # x_oa = self.obs_intent_head(x_oa.view(n, self.args.hid_size))
        x_oa = x_oa.reshape(1, n, self.args.hid_size)
        # print(x_oa.shape, x_oa_skip.shape)
        # x_oa = x_oa + x_oa_skip     # skip connection
        # self.encoded_info = x_oa.reshape(n, self.args.hid_size)

        # combine latent observations with communications
        comm, comm_prob, comm_mask = self.communicate(x_oa, info)
        if self.args.mha_comm:
            comm = self.attend_comm(comm.view(n,n,self.args.comm_dim).transpose(1,0), mask=comm_mask, is_comm=True)
            # comm = self.comm_head(comm)
            # comm = torch.tanh(comm)
        else:
            comm_sum = comm.sum(dim=0)
            comm = self.comm_head(comm_sum)
            # comm = torch.tanh(comm)
        if self.args.recurrent:
            inp = x_oa + comm
            inp = inp.view(n, self.args.comm_dim)
            x_oa_comm, cell = self.f_module(inp, (x_oa.view(n, self.args.hid_size), cell))
            h = x_oa_comm
        else:
            x_oa_comm = x + x_oa + comm # skip connection to aggregate oa with comm
            x_oa_comm = self.f_module(x_oa_comm)
            x_oa_comm = torch.tanh(x_oa_comm)
        # x = self.aggregate_info(torch.cat((x_oa, comm), -1))
        # self.obs_intent_seq.replace_step(x_oa_comm)
        self.encoded_info = x_oa_comm.reshape(n, self.args.hid_size).clone()
        if self.args.recurrent:
            self.encoded_info = self.encoded_info + comm

        # choose an action
        # a = self.attend_action(x_oa_comm)
        a = self.action_head(x_oa_comm).reshape(1, n, -1)
        if self.args.env_name != 'starcraft':
            a = F.log_softmax(a, dim=-1)
        # critic
        v = self.value_head(x_oa_comm + x_oa).reshape(1, n, -1)
        if self.args.recurrent:
            return a, v, (h.clone(), cell.clone()), comm_prob

        return a, v, comm_prob


    def communicate(self, comm, info):
        n = self.nagents
        '''Mask Communication'''
        # mask 1) input communication
        mask = self.comm_mask.view(n, n)

        # 2) Mask communcation from dead agents, 3) communication to dead agents
        num_agents_alive, agent_mask = self.get_agent_mask(self.max_len, info)
        # gating sparsity mask
        agent_mask, comm_action, comm_prob = self.get_gating_mask(comm, agent_mask)
        info['comm_action'] = comm_action.detach().numpy()

        # Mask 1) input communication 2) Mask communcation from dead agents, 3) communication to dead agents
        comm_out_mask = mask * agent_mask * agent_mask.transpose(0, 1)

        '''Perform communication'''
        # doing cts communication vectors only right now
        comm  = comm.view(n, self.args.comm_dim)
        comm = comm.unsqueeze(-2).expand(n, n, self.args.comm_dim)
        # print(comm_out_mask.unsqueeze(-1).shape, comm.shape)
        comm = comm * comm_out_mask.unsqueeze(-1).expand_as(comm)

        if not self.args.mha_comm:
            if hasattr(self.args, 'comm_mode') and self.args.comm_mode == 'avg' \
                and num_agents_alive > 1:
                comm = comm / (num_agents_alive - 1)

        return comm, comm_prob, comm_out_mask

    def do_embed(self, x):
        hidden_state, cell_state = None, None

        if self.args.recurrent:
            x, extras = x

            # In case of recurrent first take out the actual observation and then encode it.
            x = self.embed(x)

            if self.args.rnn_type == 'LSTM':
                # if you're using the extras would have both the hidden and the cell state.
                hidden_state, cell_state = extras
            else:
                hidden_state = extras
        else:
            # how to embed positional encoding
            x = self.embed(x)
            x = torch.tanh(x)
        return x, hidden_state, cell_state

    def decode(self):
        if self.args.vae:
            return self.decoder_head(self.encoded_info), self.decoding_mu, self.decoding_log_var
        return self.decoder_head(self.encoded_info)

    def get_gating_mask(self, x, agent_mask):
        # Gating
        # Hard Attention - action whether an agent communicates or not
        comm_prob = None
        if self.args.comm_action_one:
            comm_action = torch.ones(self.nagents)
        elif self.args.comm_action_zero:
            comm_action = torch.zeros(self.nagents)
        else:
            x = x.view(self.nagents, self.hid_size)
            comm_prob = F.log_softmax(self.gating_head(x), dim=-1)[0]
            comm_prob = gumbel_softmax(comm_prob, temperature=1, hard=True)
            comm_prob = comm_prob[:, 1].reshape(self.nagents)
            comm_action = comm_prob

        comm_action_mask = comm_action.expand(self.nagents, self.nagents)
        # action 1 is talk, 0 is silent i.e. act as dead for comm purposes.
        agent_mask = agent_mask * comm_action_mask.double()
        return agent_mask, comm_action, comm_prob

    def get_agent_mask(self, batch_size, info):
        n = self.nagents

        if 'alive_mask' in info:
            agent_mask = torch.from_numpy(info['alive_mask'])
            num_agents_alive = agent_mask.sum()
        else:
            agent_mask = torch.ones(n)
            num_agents_alive = n

        agent_mask = agent_mask.view(1, n)
        agent_mask = agent_mask.expand(n, n)

        return num_agents_alive, agent_mask

    def reset(self):
        self.obs_intent_seq.reset()

    def reset_layers(self):
        # reset output layers
        self.init_layer(self.action_head)
        # self.init_layer(self.value_head)
        # self.init_layer(self.attend_obs_intent.unifyheads)

    def init_layer(self, m):
        torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
        m.bias.data.fill_(0.01)

    def init_hidden(self, batch_size):
        # dim 0 = num of layers * num of direction
        return tuple(( torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True),
                       torch.zeros(batch_size * self.nagents, self.hid_size, requires_grad=True)))

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            self.init_layer(m)

class Sequence():
    def __init__(self, max_len=5):
        self.sequence = None
        self.max_len = max_len
        # self.position = 0

    def step(self, x):
        if self.sequence is None:
            self.sequence = [torch.zeros_like(x) for i in range(self.max_len)]
        # self.sequence[-1] = self.sequence[-1].xdetach()
        self.sequence.pop(0)
        self.sequence.append(x.clone())



    def replace_step(self, x):
        self.sequence[-1] = x.clone()

    def reset(self):
        self.sequence = None

    def get(self):
        return torch.cat(self.sequence)

    def mask(self):
        out = self.get().transpose(1,0)
        return (out.sum(-1) != 0).double()

class PositionalEncoding(nn.Module):
    def __init__(self, dim, nagents, max_len=1000, device='cpu', dropout=0):
        super(PositionalEncoding, self).__init__()
        """
        Inputs:
          dim: feature dimension of the positional encoding
        """
        self.P = torch.zeros((nagents, max_len, dim), device=device)
        X = torch.arange(0,max_len, device=device).reshape(-1,1) /\
            torch.pow(10000, torch.arange(0,dim,2,dtype=torch.float32, device=device) / dim)
        self.P[:,:,0::2] = torch.sin(X)
        self.P[:,:,1::2] = torch.cos(X)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X):
        """
        Inputs:
            X: tensor of size (N, T, D_in)
        Output:
            Y: tensor of the same size of X
        """
        return self.dropout(X + self.P[:,:X.shape[1],:].requires_grad_(False))
