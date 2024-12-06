# LangGround

This repository contains reference implementation for LangGround paper (accepted to NeurIPS 2024), **Language Grounded Multi-agent Reinforcement Learning with Human-interpretable Communication**, available at [https://arxiv.org/abs/2409.17348](https://arxiv.org/abs/2409.17348)

## Installation

First, clone the repo and install ic3net-envs which contains implementation for Predator-Prey and Traffic-Junction

```
# downgrade setuptools for gym=0.21
pip install setuptools==65.5.0 "wheel<0.40.0"
cd gym-dragon
pip install -e .

cd ic3net-envs
python setup.py develop
pip install tensorboardX

```
Then, install gym-dragon which contains the implementation of the Urban Search and Rescue (USAR) environment.

```
cd gym-dragon
pip install -e .
```

Finally, we need to install dependencies including PyTorch. For doing that run:

```
pip install -r requirements.txt
```
## Train baseline comm-MARL agents

Once everything is installed, we can train MARL-comm agents using these example commands

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

- IC3Net on predator prey (vision = 0)

```
python main.py --env_name predator_prey --nagents 3 --nprocesses 1 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 5 --max_steps 20 --ic3net --vision 0 --recurrent
```

- CommNet on predadator prey (vision = 0)

```
python main.py --env_name predator_prey --nagents 3 --nprocesses 1 --num_epochs 2000 --hid_size 128 --detach_gap 10 --lrate 0.001 --dim 5 --max_steps 20 --commnet --vision 0 --recurrent
```

## Train LangGround agents

### Collect offline dataset from LLM agents

To train LangGround agents using the pipeline proposed in the paper, you will need to first collect the offline communication dataset from LLM agents.

Navigate to the LLM directory and install dependencies for LLM agents including openai API package.

```
cd LLM
pip install -r requirements.txt
```

Then set up the openai API key in your system environment variables as instructed in this [guideline](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety).

```
echo "export OPENAI_API_KEY='yourkey'" >> ~/.bash_profile
source ~/.bash_profile
echo $OPENAI_API_KEY
```

Once everything is ready, we can run the LLM agent simulation using these example commands:

- 3 GPT-4-turbo agents on pp_v0 with communication
```
python pp_exp.py --model gpt-4-turbo-preview --exp_name gpt-4 --allow_comm --dim 5 --vision 0 
```
- 3 GPT-4-turbo agents on pp_v0 without communication
```
python pp_exp.py --model gpt-4-turbo-preview --exp_name gpt-4 --dim 5 --vision 0 
```

Note the above commands only collect team trajactory for one episode. To collect and process data in batchm run:

```
python offline_data_collection.py
python offline_data_process.py
```
### Train LangGround agents

To train LangGround agents with customized configurations, we recommend use the example shell scripts to call the main training function.

```
cd ic3net-env/scripts
python train_supervised.py
```

## Contacts

Please contact Huao Li ([@romanlee6](https://github.com/romanlee6)) for any questions about this repo.

## Acknowledgements

### LangGround authors
- Huao Li
- Hossein Nourkhiz Mahjoub
- Behdad Chalaki
- Vaishnav Tadiparthi
- Kwonjoon Lee
- Ehsan Moradi-Pari
- Michael Lewis
- Katia Sycara

### Prototype Communication authors
- Mycal Tucker
- Seth Karten
- Siva Kailas

### Gym-Dragon authors
- Ini Oguntola
- 
### IC3Net authors
- Amanpreet Singh
- Tushar Jain
- Sainbayar Sukhbaatar
