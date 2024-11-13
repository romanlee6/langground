import argparse
import json
import os
from dragon_llm import DragonTextEnv, ChatAgent
import numpy as np
import time

def save_dataset(args, obs,action,reward,done,comm,info):
    data_path = os.path.join(args.save_path, args.model, args.exp_name,'offline_data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    timestr = time.strftime("-%m-%d-%H-%M-%S", time.localtime())
    data = {}
    data['args'] = vars(args)
    data['obs'] = obs
    data['action'] = {x:int(action[x]) for x in action.keys()}
    data['reward'] = reward
    data['done'] = done
    data['comm'] = comm
    data['room'] = info['room']
    data['bomb'] = info['bomb']
    data['sequence'] = info['sequence']
    path = data_path + '/seed' + str(args.seed)+timestr + '.npy'
    np.save(path,data)

parser = argparse.ArgumentParser(description='Text interface for LLM agent')
# training
# note: number of steps per epoch = epoch_size X batch_size x nprocesses
parser.add_argument('--save_path', default='data/', type=str,
                    help='path of saved log files')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed for environment initialization')
parser.add_argument('--tom', action='store_true', default=False,
                    help='conduct belief inference measurements')
parser.add_argument('--belief', action='store_true', default=False,
                    help='maintain a belief state about the env')
parser.add_argument('--allow_comm', action='store_true', default=False,
                    help='allow communication between team members')
parser.add_argument('--model', type=str, default='gpt-4-turbo-preview',
                    help='base LM to use')
parser.add_argument('--include_agent_action', action='store_true', default=False,
                    help='present other agents action in obs')
# parser.add_argument('--act_and_comm', action='store_true', default=False,
#                     help='allow action and communication in the same round')
parser.add_argument('--tool_per_agent', type = int, default=2,
                    help='kinds of tool each agent has')
parser.add_argument('--temperature', type = float, default=0,
                    help='temperature used by base LM')
parser.add_argument('--max_step', type = int, default=30,
                    help='maximum steps allowed')
parser.add_argument('--exp_name', type = str, default='default_exp',
                    help='exp name to save')
# model
args = parser.parse_args()

print(args)


DATA_PATH = os.path.join(args.save_path ,args.model ,args.exp_name,'seed' + str(args.seed))+'/'
# DATA_PATH += 'GPT4-turbo-comm-seed24/'
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# print(obs_text)
seed = args.seed
ToM = args.tom
belief = args.belief
allow_comm = args.allow_comm
model_name = args.model
act_and_comm = True

env = DragonTextEnv(seed = seed,include_agent_action = args.include_agent_action,allow_comm = allow_comm,act_and_comm = act_and_comm,tool_per_agent = args.tool_per_agent)
Action = env.env.action_enum
obs = env.env._get_obs()
info = {}
initial_node = str(env.env.agents['alpha'].node.id)
initial_bomb = str(env.env.agents['alpha'].bomb.id)
chat_agents = {'alpha':ChatAgent(agent_id='alpha',model = model_name,temperature=args.temperature,belief = belief,allow_comm = allow_comm, initial_bomb = initial_bomb, initial_node = initial_node),
               'bravo':ChatAgent(agent_id='bravo',model = model_name,temperature=args.temperature,belief = belief,allow_comm = allow_comm,initial_bomb = initial_bomb, initial_node = initial_node),
               'charlie':ChatAgent(agent_id='charlie',model = model_name,temperature=args.temperature,belief = belief,allow_comm = allow_comm,initial_bomb = initial_bomb, initial_node = initial_node)}
initial_actions = {'alpha':Action.go_to(int(initial_node)),'bravo':Action.go_to(int(initial_node)),'charlie':Action.go_to(int(initial_node))}
communications = {'alpha':'None','bravo':'None','charlie':'None'}

chat_output = {}
actions = {}
done = {'__all__':False}
round = 1

cols = ['round', 'agent_id',
 'action', 'comm', 'crop_comm','room','bomb','sequence']

with open(DATA_PATH + 'summary.csv', 'w+', encoding='utf-8') as f:
    for k in cols:
        f.write(k)
        f.write(',')
    f.write('\n')



who_has_inspected_what = {'alpha':set(),'bravo':set(),'charlie':set()}

while not done['__all__'] and round <= args.max_step:
    for agent_id in chat_agents.keys():
        chat_agent = chat_agents[agent_id]
        if round == 1 and not belief:
            obs, reward, done, info, obs_text = env.step(agent_id, 0,initial_actions, communications)

            chat_agent.update_history(obs_text)

        # if agent_id not in initial_actions.keys():
        chat_output[agent_id] = chat_agent.step()

        initial_actions[agent_id], communications[agent_id] = env.decode_action(chat_output[agent_id])
        # initial_actions[agent_id], communications[agent_id] = env.decode_action_API(chat_output[agent_id])

        agent = env.env.agents[agent_id]

        room = agent.node.id
        if agent.bomb is not None:
            bomb = agent.bomb.id
            sequence = agent.bomb._full_sequence[agent.bomb._current_step:]
        else:
            bomb = -1
            sequence = []
        if initial_actions[agent_id] is not None:
            act = initial_actions[agent_id]
        else:
            act = -1
        actions[agent_id] = act
        if 'room' not in info.keys():
            info['room'] = {}
        info['room'][agent_id]= room
        if 'bomb' not in info.keys():
            info['bomb'] = {}
        info['bomb'][agent_id]= bomb
        if 'sequence' not in info.keys():
            info['sequence'] = {}
        info['sequence'][agent_id]= np.array(sequence)

        summary = {'round': round, 'agent_id': agent_id, 'action':int(act),'comm':communications[agent_id],'crop_comm':communications[agent_id].split('.')[0],'room':int(room),'bomb':int(bomb),'sequence':np.array(sequence)}
        print(summary)
        with open(DATA_PATH + 'summary.csv', 'a+', encoding='utf-8') as f:
            for k,v in summary.items():
                f.write(str(v).replace(',',';').replace('\n',''))
                f.write(',')
            f.write('\n')


        obs, reward, done, _, obs_text = env.step(agent_id, round,initial_actions, communications)

        new_belief = chat_agent.update_history(obs_text)





        ground_truth = None
        ToM1st = None
        ToM2nd = None
        ToM3rd = None

        if ToM:
            target_id = np.random.choice([x for x in chat_agents.keys() if x != agent_id])
            if initial_actions[agent_id].node() is not None:
                if agent.node.id != initial_actions[agent_id].node().id:
                    ground_truth = False
                else:
                    ground_truth = True

                # first-order ToM / introspective
                ToM1st = chat_agent.ask(obs_text+
                    'Do you know the current contents of room {room_id}?'.format(player_id=target_id,
                                                                                                  room_id=initial_actions[
                                                                                                      agent_id].node().id))
                # second-order ToM
                ToM2nd = chat_agent.ask(obs_text+
                    'Does player {player_id} know the current contents of room {room_id}?'.format(player_id=target_id,
                                                                                                  room_id=initial_actions[
                                                                                                      agent_id].node().id))
                # third-order ToM
                ToM3rd = chat_agent.ask(obs_text+
                    'Based on the observation and previous history, is player {player_id} aware of the fact that you know the current contents of room {room_id}?'.format(player_id=target_id,
                                                                                                  room_id=initial_actions[
                                                                                                      agent_id].node().id))
            elif initial_actions[agent_id].tool() is not None:
                ground_truth = False
                if agent.bomb:
                    bomb_id = agent.bomb.id
                    # if agent.bomb.state =='defused':
                    #     chat_agent.ask('Does player {player_id} know bomb {bomb_id} has been defused?'.format(player_id=target_id,bomb_id=bomb_id))
                    # else:

                    # first-order ToM / introspective
                    # ToM1st = chat_agent.ask(obs_text+
                    #     'Do you know the current state and remaining sequence of bomb {bomb_id}?'.format(
                    #         player_id=target_id, bomb_id=bomb_id))

                    ToM1st = chat_agent.ask(obs_text+
                        'Do you know the state and remaining sequence of bomb {bomb_id} has been changed?'.format(
                            player_id=target_id, bomb_id=bomb_id))

                    # second-order ToM
                    ToM2nd = chat_agent.ask(obs_text+
                        'Does player {player_id} know the state and remaining sequence of bomb {bomb_id} has been changed?'.format(player_id=target_id, bomb_id=bomb_id))
                    # third-order ToM
                    ToM3rd = chat_agent.ask(obs_text+
                        'Based on the observation and previous history, is player {player_id} aware of the fact that you have changed the state and remaining sequence of bomb {bomb_id}?'.format(
                            player_id=target_id,
                            bomb_id=bomb_id))
                elif isinstance(reward, int):
                    if reward >= 0:
                        ToM1st = chat_agent.ask(obs_text +
                                                'Do you know a bomb phase has just been defused?')

                        # second-order ToM
                        ToM2nd = chat_agent.ask(obs_text +
                                                'Does player {player_id} know a bomb phase has just been defused?'.format(player_id = target_id))
                        # third-order ToM
                        ToM3rd = chat_agent.ask(obs_text +
                                                'Based on the observation and previous history, is player {player_id} aware of the fact that you know a bomb phase has just been defused?'.format(player_id = target_id))


            elif initial_actions[agent_id] == Action.inspect_bomb:
                if agent.bomb:
                    bomb_id = agent.bomb.id
                    who_has_inspected_what[agent_id].add(bomb_id)
                    ground_truth = bomb_id in who_has_inspected_what[target_id]
                    # first-order ToM / introspective
                    ToM1st = chat_agent.ask(obs_text+
                        'Do you know the sequence of bomb {bomb_id}?'.format(bomb_id=bomb_id))
                    # second-order ToM
                    ToM2nd = chat_agent.ask(obs_text+
                        'Does player {player_id} know the sequence of bomb {bomb_id}?'.format(player_id=target_id,
                                                                                              bomb_id=bomb_id))
                    # third-order ToM
                    ToM3rd = chat_agent.ask(obs_text+
                        'Based on the observation and previous history, is player {player_id} aware of the fact that you know the sequence of bomb {bomb_id}?'.format(player_id=target_id,
                                                                                              bomb_id=bomb_id))
                else:
                    ground_truth = None
                    ToM1st = None
                    ToM2nd = None
                    ToM3rd = None
            else:
                ground_truth =None
                ToM1st = None
                ToM2nd = None
                ToM3rd = None





        chat_agent.save(DATA_PATH)



        # record = {'round': round, 'agent_id': agent_id, 'chat_output': chat_output[agent_id], 'action':act,'comm':communications[agent_id],'obs_text': obs_text}
        # with open(DATA_PATH + 'record.json', 'a+', encoding='utf-8') as f:
        #     json.dump(record, f)
        #
        # summary = {'round': round, 'agent_id': agent_id, 'chat_output': chat_output[agent_id], 'action':act,'comm':communications[agent_id],'obs_text': obs_text,"new_belief":new_belief, 'ground_truth':ground_truth, 'ToM1st':ToM1st, 'ToM2nd':ToM2nd, 'ToM3rd':ToM3rd}
        # print(summary)
        # with open(DATA_PATH + 'summary.csv', 'a+', encoding='utf-8') as f:
        #     for k,v in summary.items():
        #         f.write(str(v).replace(',',';').replace('\n',''))
        #         f.write(',')
        #     f.write('\n')
    save_dataset(args, obs, actions, reward, done, communications,info)
    round +=1
