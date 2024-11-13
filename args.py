import argparse

def get_args():

    # parser = argparse.ArgumentParser(description='PyTorch RL trainer')
    parser = argparse.ArgumentParser(description='PyTorch RL trainer')
    # training
    # note: number of steps per epoch = epoch_size X batch_size x nprocesses
    parser.add_argument('--num_epochs', default=100, type=int,
                     help='number of training epochs')
    parser.add_argument('--epoch_size', type=int, default=10,
                     help='number of update iterations in an epoch')
    parser.add_argument('--batch_size', type=int, default=500,
                     help='number of steps before each update (per thread)')
    parser.add_argument('--nprocesses', type=int, default=16,
                     help='How many processes to run')
    # model
    parser.add_argument('--hid_size', default=64, type=int,
                     help='hidden layer size')
    parser.add_argument('--recurrent', action='store_true', default=False,
                     help='make the model recurrent in time')
    # optimization
    parser.add_argument('--gamma', type=float, default=1.0,
                     help='discount factor')
    parser.add_argument('--tau', type=float, default=1.0,
                     help='gae (remove?)')
    parser.add_argument('--seed', type=int, default=-1,
                     help='random seed. Pass -1 for random seed')  # TODO: works in thread?
    parser.add_argument('--normalize_rewards', action='store_true', default=False,
                     help='normalize rewards in each batch')
    parser.add_argument('--lrate', type=float, default=0.001,
                     help='learning rate')
    parser.add_argument('--entr', type=float, default=0,
                     help='entropy regularization coeff')
    parser.add_argument('--value_coeff', type=float, default=0.01,
                     help='coeff for value loss term')
    # environment
    parser.add_argument('--env_name', default="Cartpole",
                     help='name of the environment to run')
    parser.add_argument('--max_steps', default=20, type=int,
                     help='force to end the game after this many steps')
    parser.add_argument('--nactions', default='1', type=str,
                     help='the number of agent actions (0 for continuous). Use N:M:K for multiple actions')
    parser.add_argument('--action_scale', default=1.0, type=float,
                     help='scale action output from model')
    # other
    parser.add_argument('--plot', action='store_true', default=False,
                     help='plot training progress')
    parser.add_argument('--plot_env', default='main', type=str,
                     help='plot env name')
    parser.add_argument('--save', default='trained_models', type=str,
                     help='save the model after training')
    parser.add_argument('--save_every', default=0, type=int,
                     help='save the model after every n_th epoch')
    parser.add_argument('--load', default='trained_models', type=str,
                     help='load the model')
    parser.add_argument('--display', action="store_true", default=False,
                     help='Display environment state')

    parser.add_argument('--random', action='store_true', default=False,
                     help="enable random model")

    # CommNet specific args
    parser.add_argument('--commnet', action='store_true', default=False,
                     help="enable commnet model")
    parser.add_argument('--ic3net', action='store_true', default=False,
                     help="enable commnet model")
    parser.add_argument('--mac', action='store_true', default=False,
                     help="enable multi-agent communication model")
    parser.add_argument('--nagents', type=int, default=1,
                     help="Number of agents (used in multiagent)")
    parser.add_argument('--comm_mode', type=str, default='avg',
                     help="Type of mode for communication tensor calculation [avg|sum]")
    parser.add_argument('--comm_passes', type=int, default=1,
                     help="Number of comm passes per step over the model")
    parser.add_argument('--comm_mask_zero', action='store_true', default=False,
                     help="Whether communication should be there")
    parser.add_argument('--mean_ratio', default=1.0, type=float,
                     help='how much coooperative to do? 1.0 means fully cooperative')
    parser.add_argument('--rnn_type', default='MLP', type=str,
                     help='type of rnn to use. [LSTM|MLP]')
    parser.add_argument('--detach_gap', default=10000, type=int,
                     help='detach hidden state and cell state for rnns at this interval.'
                          + ' Default 10000 (very high)')
    parser.add_argument('--comm_init', default='uniform', type=str,
                     help='how to initialise comm weights [uniform|zeros]')
    parser.add_argument('--hard_attn', default=False, action='store_true',
                     help='Whether to use hard attention: action - talk|silent')
    parser.add_argument('--comm_action_one', default=False, action='store_true',
                     help='Whether to always talk, sanity check for hard attention.')
    parser.add_argument('--comm_action_zero', default=False, action='store_true',
                        help='Whether to never talk.')
    parser.add_argument('--advantages_per_action', default=False, action='store_true',
                     help='Whether to multipy log porb for each chosen action with advantages')
    parser.add_argument('--share_weights', default=False, action='store_true',
                     help='Share weights for hops')
    parser.add_argument('--log_dir', default='tb_logs', type=str,
                     help='directory to save tensorboard logs')
    parser.add_argument('--exp_name', default='default_exp', type=str,
                     help='directory to save tensorboard logs')

    # TODO: Sanity check so as to make sure discrete and proto works for environments other than predator-prey.
    #  Currently the discrete and prototype based methods will only really take effect from inside the CommNet.
    parser.add_argument('--use_proto', default=False, action='store_true',
                     help='Whether to use prototype nets in the communication layer.')

    parser.add_argument('--discrete_comm', default=False, action='store_true',
                     help='Whether to use discrete_comm')
    parser.add_argument('--num_proto', type=int, default=6,
                     help="Number of prototypes to use")
    parser.add_argument('--add_comm_noise', default=False, action='store_true',
                    help='Whether to add noise to communication')

    parser.add_argument('--comm_dim', type=int, default=128,
                     help="Dimension of the communication vector")

    # TODO: Formalise this gating head penalty factor
    parser.add_argument('--gating_head_cost_factor', type=float, default=0.0,
                     help='discount factor')
    parser.add_argument('--restore', action='store_true', default=False,
                     help='plot training progress')

    parser.add_argument('--budget', type=float, default=1.0,
                     help='Communication budget')
    # objective function communication regularization terms
    parser.add_argument('--min_comm_loss', action='store_true', default=False,
                     help='minimize communication loss')
    parser.add_argument('--eta_comm_loss', type=float, default=1.0,
                     help='Communication budget hyperparameter')
    parser.add_argument('--max_info', action='store_true', default=False,
                     help='maximize mutual information')
    parser.add_argument('--eta_info', type=float, default=1.0,
                     help='maximize mutual information hyperparameter')

    parser.add_argument('--soft_budget', type=float, default=1.0,
                        help='Soft comm budget')
    # autoencoder
    parser.add_argument('--autoencoder', action='store_true', default=False,
                     help='use autoencoder to learn comms')
    parser.add_argument('--autoencoder_action', action='store_true', default=False,
                     help='use actions intent in autoencoder')
    # null comm removal
    parser.add_argument('--remove_null', action='store_true', default=False,
                     help='remove null communications from being communicated')
    parser.add_argument('--null_dict_dir', type=str, default='',
                     help='null dictionary directory')

    # multi-headed attention for communication receiving
    parser.add_argument('--mha_comm', action='store_true', default=False,
                        help='multi-headed attention for communication receiving')
    parser.add_argument('--num_heads', type=int, default=1,
                    help="Number of heads for attention")
    parser.add_argument('--preencode', action='store_true', default=False,
                    help='pretrain autoencoder')
    # use variational autoencoder
    parser.add_argument('--vae', action='store_true', default=False,
                    help='variational autoencoder')
    parser.add_argument('--use_vqvib', action='store_true', default=False,
                    help='variational autoencoder')
    parser.add_argument('--use_compositional', action='store_true', default=False,
                    help='compositional messages')
    parser.add_argument('--contrastive', action='store_true', default=False,
                    help='contrastive communication critic')
    parser.add_argument('--comp_beta', type=float, default=0.01,
                    help='compositional Communication budget hyperparameter')

    # use supervised loss to bias proto token learnt
    parser.add_argument('--supervised_comm', action='store_true', default=False,
                        help='use supervised learning')
    parser.add_argument('--norm_comm', action='store_true', default=False,
                        help='normalize comm vec')
    parser.add_argument('--data_path', type=str,
                        default='',
                        help='path of offline dataset for supervised learning')
    parser.add_argument('--supervised_gamma', type=float, default=0.1,
                        help='weight hyperparameter of supervised loss')
    parser.add_argument('--sampling_method', type=str, default='ind',
                        help='method used for sampling the offline dataset')
    parser.add_argument('--prey_loc_res', action="store_true", default=False,
                        help="if restrain the range of prey loc")
    parser.add_argument('--eval_mode', type=str, default='train',
                        help="train or eval mode for generalize exp")
    return parser
