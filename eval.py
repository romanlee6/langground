import os, sys
os.environ["OMP_NUM_THREADS"] = "1"
# TODO: Run proto version

env = "traffic_junction"
# seed = 1
# method = 'easy_fixed_proto'
seeds = [0,1,2,3,4,5,6,7,8,9]
# seeds = [0]
# methods = ['easy_fixed_proto', 'easy_fixed_proto_autoencoder', 'easy_proto_soft_minComm_autoencoder']
methods = ['easy_proto_soft_minComm_autoencoder', 'easy_soft_minComm_autoencoder', 'hard_soft_minComm_autoencoder']
# methods = ['easy_proto_soft_minComm_autoencoder']
for method in methods:
    if 'hard' in method:
        seeds = [1,2,3,4,6,7]
    # method = 'easy_fixed_autoencoder'
    exp_name = "tj_" + method
    discrete_comm = False
    if "proto" in method:
        discrete_comm = True
    vision = 0
    num_epochs = 1
    hid_size= 64
    save_every = 100
    # g=1. If this is set to true agents will communicate at every step.
    comm_action_one = False
    comm_action_zero = False
    if "fixed" in method or "baseline" in method:
        comm_action_one = True

    if "medium" in method:
        nagents = 10
        max_steps = 40
        dim = 14
        add_rate_min = 0.05
        add_rate_max = 0.2
        difficulty = 'medium'
        num_proto = 112
    elif "hard" in method:
        nagents = 20
        max_steps = 80
        dim = 18
        add_rate_min = 0.02
        add_rate_max = 0.05
        difficulty = 'hard'
        num_proto = 256
    else:
        # easy
        nagents = 5
        max_steps = 20
        dim = 6
        add_rate_min = 0.1
        add_rate_max = 0.3
        difficulty = 'easy'
        num_proto = 56



    # run_str = f"python evaluate_null_finder.py --env_name {env} --nagents {nagents} --nprocesses 0 "+\
    run_str = f"python evaluate_comm_action.py --env_name {env} --nagents {nagents} --nprocesses 0 "+\
              f"--load /Users/seth/Documents/research/neurips/paper_models "+\
              f"--num_epochs {num_epochs}  --epoch_size 10 "+\
              f"--hid_size {hid_size} "+\
              f" --detach_gap 10 --ic3net --vision {vision} "+\
              f"--recurrent --comm_dim {hid_size} "+\
              f"--max_steps {max_steps} --dim {dim} --nagents {nagents} --add_rate_min {add_rate_min} --add_rate_max {add_rate_max} --curr_epochs 1000 --difficulty {difficulty} "+\
              f"--exp_name {exp_name} --save_every {save_every} "
    if discrete_comm:
        run_str += f"--discrete_comm --use_proto --num_proto {num_proto} "
    if comm_action_one:
        run_str += f"--comm_action_one  "
    if comm_action_zero:
        run_str += f"--comm_action_zero "
    if "autoencoder" in method:
        run_str += "--autoencoder "
    # print(run_str)
    for seed in seeds:
        os.system(run_str + f"--seed {seed}")
