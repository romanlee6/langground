# LangGround

## Installation

First, clone the repo and install ic3net-envs which contains implementation for Predator-Prey and Traffic-Junction

```
# downgrade setuptools for gym=0.21
pip install setuptools==65.5.0 "wheel<0.40.0"
cd gym-dragon
pip install -e .

cd IC3Net/ic3net-envs
python setup.py develop
pip install tensorboardX

```

Next, we need to install dependencies for IC3Net including PyTorch. For doing that run:

```
pip install -r requirements.txt
```
## Running

Once everything is installed, we can run the using these example commands

Note: We performed our experiments on `nprocesses` set to 1, you can change it according to your machine, but the plots may vary.

Note: Use `OMP_NUM_THREADS=1` or `torch.set_num_threads(1)` to limit the number of threads spawned if you want to run multiple processes.

### Urban Search and Rescue (i.e. gym_dragon)
- IC3Net on mini_dragon with 5 nodes
```
python main.py --env_name mini_dragon --exp_name ic3net --nagents 3 --hid_size 128 --nprocesses 1 --num_epochs 2000 --epoch_size 10 --detach_gap 10 --lrate 0.0003 --max_steps 100 --ic3net --comm_dim 128 --recurrent
```
- IC3Net w/o comm on mini_dragon with 5 nodes
```
python main.py --env_name mini_dragon --exp_name no_comm --nagents 3 --hid_size 128 --nprocesses 1 --num_epochs 2000 --epoch_size 10 --detach_gap 10 --lrate 0.0003 --max_steps 100 --ic3net --comm_dim 128 --recurrent --comm_action_zero
```

- Prototype-based communication on mini_dragon with 5 nodes
```
python main.py --env_name mini_dragon --exp_name no_comm --nagents 3 --hid_size 128 --nprocesses 1 --num_epochs 2000 --epoch_size 10 --detach_gap 10 --lrate 0.0003 --max_steps 100 --ic3net --comm_dim 128 --recurrent --discrete_comm --use_proto --num_proto 10
```

### Predator-Prey

- IC3Net on easy version

```
python main.py --env_name predator_prey --nagents 3 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 5 --max_steps 20 --ic3net --vision 0 --recurrent
```

- CommNet on easy version

```
python main.py --env_name predator_prey --nagents 3 --nprocesses 16 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 5 --max_steps 20 --commnet --vision 0 --recurrent
```
