import os, sys

#os.environ["OMP_NUM_THREADS"] = "1"
# TODO: Run proto version


env = "mini_dragon"

seeds = [13]
#
# methods = ['supervised_exact','supervised_exact_large','supervised_exact_large_10','supervised_exact_large_0.1']
methods = ['supervised_exact_norm_10_large','mac_mha_fixed_proto_autoencoder_vqvib']


#methods = ['ic3net_proto_autoencoder_supervised','ic3net_autoencoder_supervised','ic3net_fixed_autoencoder_supervised']

# for seed in seeds:
for method in methods:
    seed = 13
    exp_name = env + '_' + method
    num_epochs = 2000
    hid_size = 256
    save_every = 100
    lrate = 0.0001
    nagents = 3
    max_steps = 100

    nprocesses = 0

    num_proto = 58

    discrete_comm = False
    if "proto" in method:
        discrete_comm = True
    if 'fixed' in method:
        comm_action_one = True
    else:
        comm_action_one = False
    if 'noComm' in method:
        comm_action_zero = True
    else:
        comm_action_zero = False
    if '0.1' in method:
        supervised_gamma = 0.1
    elif '10' in method:
        supervised_gamma = 10
    else:
        supervised_gamma = 1
    if 'supervised' in method:
        supervised_comm = True
    else:
        supervised_comm = False
    if 'norm' in method:
        norm_comm = True
    else:
        norm_comm = False
    restore = False
    sampling_method = 'exact'
    if 'large' in method:
        data_path = '../../LLM/embedded_256_offline_llm_dataset_dragon_large.csv'
    else:
        data_path = '../../LLM/embedded_256_offline_llm_dataset_dragon.csv'

    reward_curriculum = False
    variable_gate = False

    # run_str = f"python evaluate_null_finder.py --env_name {env} --nagents {nagents} --nprocesses 0 "+\
    run_str = f"python ../../main.py --env_name {env} --nagents {nagents} --nprocesses {nprocesses} " + \
              f"--num_epochs {num_epochs}  --epoch_size 10 " + \
              f"--hid_size {hid_size} --lrate {lrate} " + \
              f" --detach_gap 10 --supervised_gamma {supervised_gamma} --sampling_method {sampling_method} --data_path {data_path} " + \
              f"--comm_dim {hid_size} " + \
              f"--max_steps {max_steps} " + \
              f"--exp_name {exp_name} --save_every {save_every} "
    # print(run_str)
    if restore:
        run_str += f"--restore "
    if norm_comm:
        run_str += f"--norm_comm "
    if supervised_comm:
        run_str += f"--supervised_comm "

    if discrete_comm:
        run_str += f"--discrete_comm --use_proto --num_proto {num_proto} "
    if comm_action_one:
        run_str += f"--comm_action_one "
    if comm_action_zero:
        run_str += f"--comm_action_zero "
    if reward_curriculum:
        run_str += f"--gate_reward_curriculum "

    if "minComm" in method:
        run_str += "--min_comm_loss --eta_comm_loss 1. "
    if "maxInfo" in method:
        run_str += "--max_info --eta_info 0.5 "
    if "autoencoder" in method:
        run_str += "--autoencoder "
    if "action" in method:
        run_str += "--autoencoder_action "
    if 'mha' in method:
        run_str += '--mha_comm '
    if 'timmac' in method:
        run_str += '--timmac '
    elif 'mac' in method:
        run_str += '--mac --recurrent --rnn_type GRU '
    else:
        run_str += '--ic3net --recurrent '

    if 'preencode' in method:
        run_str += '--preencode '
    if 'vae' in method:
        run_str += '--vae '
    if 'vqvib' in method:
        run_str += '--use_vqvib '
    if 'compositional' in method:
        run_str += '--use_compositional '
    if 'contrastive' in method:
        run_str += '--contrastive '

    os.system(run_str + f"--seed {seed}")
