import os, sys

# os.environ["OMP_NUM_THREADS"] = "1"
# TODO: Run proto version


env = "predator_prey"

seeds = [2,3]

# methods = ['supervised_v0_half','supervised_v1_half','reproduce_ic3net_v0_half','reproduce_ic3net_v1_half','reproduce_proto_v0_half','reproduce_proto_v1_half']
# methods = ['ic3net_fixed_v0','noComm_v0','proto_v0','ic3net_v0','ic3net_autoencoder_v0','mac_mha_fixed_proto_autoencoder_vqvib_v0']
methods = ['supervised_v1_exact_norm_NEW']
for seed in seeds:
    for method in methods:

        # if method in ['reproduce_noComm_v0','reproduce_ic3net_v0']:
        #     restore = True
        # else:
        #     restore = False

        exp_name = method
        num_epochs = 500
        hid_size = 256
        save_every = 100
        lrate = 0.001
        nagents = 3
        max_steps = 20

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

        if 'supervised' in method:
            supervised_comm = True
        else:
            supervised_comm = False


        if 'res' in method:

            prey_loc_res = True
        else:

            prey_loc_res = False

        supervised_gamma = 1


        if 'norm' in method:
            norm_comm = True
        else:
            norm_comm = False

        if 'team' in method:
            sampling_method = 'team'
            if 'v0' in method:
                data_path = '../../LLM/embedded_256_offline_llm_dataset_v0_team.csv'
                vision = 0
            elif 'v1' in method:
                data_path = '../../LLM/embedded_256_offline_llm_dataset_v1_team.csv'
                vision = 1
        elif 'exact' in method:
            sampling_method = 'exact'
            if 'v0' in method:
                data_path = '../../LLM/embedded_256_offline_llm_dataset_v0.csv'
                vision = 0
            elif 'v1' in method:
                data_path = '../../LLM/embedded_256_offline_llm_dataset_v1_exact.csv'
                vision = 1

        else:
            sampling_method = 'ind'
            if 'v0' in method:
                data_path = '../../LLM/embedded_256_offline_llm_dataset_v0.csv'
                vision = 0
            elif 'v1' in method:
                data_path = '../../LLM/embedded_256_offline_llm_dataset_v1.csv'
                vision = 1

        reward_curriculum = False
        variable_gate = False

        # run_str = f"python evaluate_null_finder.py --env_name {env} --nagents {nagents} --nprocesses 0 "+\
        run_str = f"python ../../main.py --env_name {env} --nagents {nagents} --nprocesses {nprocesses} " + \
                  f"--num_epochs {num_epochs}  --epoch_size 10 " + \
                  f"--hid_size {hid_size} --lrate {lrate} " + \
                  f"--detach_gap 10 --supervised_gamma {supervised_gamma} --sampling_method {sampling_method} --data_path {data_path} " + \
                  f"--comm_dim {hid_size} " + \
                  f"--max_steps {max_steps} --vision {vision} --dim 5 " + \
                  f"--exp_name {exp_name} --save_every {save_every} "
        # print(run_str)
        # if restore:
        #     run_str += f"--restore "

        if norm_comm:
            run_str += f"--norm_comm "
        if supervised_comm:
            run_str += f"--supervised_comm "
        if prey_loc_res:
            run_str += f"--prey_loc_res "
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
        print(run_str)
        os.system(run_str + f"--seed {seed}")
