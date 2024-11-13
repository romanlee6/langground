import os, sys

# os.environ["OMP_NUM_THREADS"] = "1"
# TODO: Run proto version


env = "predator_prey"

seeds = [256]

methods = ['models/supervised_v0_exact_half']
# methods = ['supervised_256_adaEmbeddings_cos_1_cont']
for seed in seeds:
    for method in methods:

        exp_name = method
        num_epochs = 2000
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
        comm_action_one = False
        if 'noComm' in method:
            comm_action_zero = True
        else:
            comm_action_zero = False

        if 'supervised' in method:
            supervised_comm = True
        else:
            supervised_comm = False

        if 'half' in method:
            prey_loc_res = True
        else:
            prey_loc_res = False
        prey_loc_res = False

        supervised_gamma = 1
        restore = True

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
        else:
            if 'exact' in method:
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


        # supervised_comm = True
        # data_path = '../../LLM/embedded_256_offline_llm_dataset_v0.csv'
        # vision = 0
        # sampling_method = 'exact'

        reward_curriculum = False
        variable_gate = False
        # data_path = '../../LLM/embedded_256_offline_llm_dataset_v1.csv'
        # vision = 1
        run_str = f"python generalizable_new.py --env_name {env} --nagents {nagents} --nprocesses {nprocesses} " + \
                  f"--num_epochs {num_epochs}  --epoch_size 10 " + \
                  f"--hid_size {hid_size} --lrate {lrate} " + \
                  f"--detach_gap 10 --supervised_gamma {supervised_gamma} --sampling_method {sampling_method} --data_path {data_path} " + \
                  f"--comm_dim {hid_size} " + \
                  f"--max_steps {max_steps} --vision {vision} --dim 5 " + \
                  f"--exp_name {exp_name} --save_every {save_every} "
        # print(run_str)
        if restore:
            run_str += f"--restore "

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
