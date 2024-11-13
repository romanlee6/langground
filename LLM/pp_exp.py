import os

from ic3net_envs.predator_prey_env import PredatorPreyEnv
from pp_llm import *
import argparse

def save_dataset(args, obs,action,reward,done,comm,info):
    data_path = os.path.join(args.save_path, args.model, args.exp_name,'offline_data')
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    timestr = time.strftime("-%m-%d-%H-%M-%S", time.localtime())
    data = {}
    data['args'] = vars(args)
    data['obs'] = obs
    data['action'] = action
    data['reward'] = reward
    data['done'] = done
    data['comm'] = comm
    data['predator_locs'] = info['predator_locs']
    data['prey_locs'] = info['prey_locs']
    path = data_path + '/seed' + str(args.seed)+timestr + '.npy'
    np.save(path,data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Example GCCNet environment random agent')
    parser.add_argument('--nagents', type=int, default=3, help="Number of agents")
    parser.add_argument('--display', action="store_true", default=True,
                        help="Use to display environment")

    parser.add_argument('--allow_comm', action="store_true", default=False,
                        help="allow communication among agents")
    parser.add_argument('--belief', action="store_true", default=False,
                        help="maintain belief representations")
    parser.add_argument('--seed', type = int, default=0,
                        help="random seed")
    parser.add_argument('--exp_name', type=str, default='default',
                        help="exp_name")
    parser.add_argument('--save_path', type=str, default='data/pp/',
                        help="experiment name")
    env = PredatorPreyTextEnv()
    env.init_curses()
    env.init_args(parser)

    args = parser.parse_args()

    args.dim = 5
    # args.vision = 1
    # args.seed = 244
    args.mode = 'cooperative'
    # args.save_path = 'data/pp_prompt/'
    args.model = 'gpt-4-turbo-preview'
    args.temperature = 0

    # args.exp_name = 'comm'
    # args.allow_comm = True
    # args.belief = False

    args.nfriendly = args.nagents



    DATA_PATH = os.path.join(args.save_path, args.model, args.exp_name, 'seed' + str(args.seed)) + '/'
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)



    env.multi_agent_init(args)

    BASE = env.BASE
    PREY_CLASS = env.PREY_CLASS
    PREDATOR_CLASS = env.PREDATOR_CLASS
    OUTSIDE_CLASS = env.OUTSIDE_CLASS


    agents = []
    for i in range(args.nagents):
        agent = GPTAgent(args,i)
        agents.append(agent)

    episodes = 0

    while episodes < 1:
        obs = env.reset()
        for i in range(args.nagents):
            agents[i].reset()
        done = False
        rewards = []
        comms = ['','','']
        step = 0
        env.render()
        last_loc = env.predator_loc.copy()
        info = {'predator_locs':env.predator_loc.copy(),'prey_locs':env.prey_loc.copy()}
        # idle_count = 0
        while not done and step < 20:
            actions = []
            action_selections = []
            # explanations = []
            beliefs = []
            text_obs = env.translate_observation_to_description(obs,args.allow_comm,comms)
            comms = []
            for i in range(args.nagents):

                action, comm = agents[i].step(text_obs[i])

                comms.append(comm)
                # explanations.append(exp)
                beliefs.append(agents[i].belief_state)
                action_selections.append(action)
                actions.append(agents[i].decode_action(action))
                agents[i].save(DATA_PATH)



            new_obs, reward, done, new_info = env.step(actions)

            for i in range(len(env.reached_prey)):
                if env.reached_prey[i]:
                    agents[i].reached_prey = True

            # if np.array_equal(last_loc,env.predator_loc):
            #     idle_count+=1
            # last_loc = env.predator_loc

            if np.all(env.reached_prey):
                done = True

            save_dataset(args, obs, actions, reward,done,comms,info)



            rewards.append(reward)
            print(reward)

            for i in range(args.nagents):
                summary = {'step': step, 'agent_id': i, 'obs': text_obs[i],'action':action_selections[i],'comm':comms[i],'belief':beliefs[i],'reward':reward[i],'prey_loc':env.prey_loc[0],'predator_loc':last_loc[i]}
                # print(summary)
                with open(DATA_PATH + 'summary.csv', 'a+', encoding='utf-8') as f:
                    for k, v in summary.items():
                        f.write(str(v).replace(',', ';').replace('\n', ''))
                        f.write(',')
                    f.write('\n')

            obs = new_obs.copy()
            last_loc = env.predator_loc.copy()
            info = new_info.copy()

            if args.display:
                env.render()
            step += 1

        print('Episode: {episode}, Reward: {reward}'.format(episode = str(episodes), reward = str(np.array(rewards).sum(axis=1).mean(axis=0))))

        episodes += 1
    for agent in agents:
        print(agent.usage)
    env.close()
