import os, sys

env = 'dragon'

seeds = [x for x in range(20,50)]

methods = ['pp_ez_v0']
# methods = ['dragon']
for seed in seeds:

    for method in methods:
        save_path = 'data/'+method+'/'+'/'

        run_str = f"python pp_exp.py --allow_comm --belief --save_path {save_path} "


        os.system(run_str + f"--seed {seed}")
