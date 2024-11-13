import time
import json
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import numpy as np

import os

from ic3net_envs.predator_prey_env import PredatorPreyEnv

from openai import OpenAI

client = OpenAI(api_key = 'na')


import anthropic

a_client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key="na",
)

class GPTAgent(object):
    def __init__(self, args, index):

        self.args = args
        self.model = args.model
        self.temperature = args.temperature
        self.index = index
        self.allow_comm = args.allow_comm
        self.belief = args.belief


        self.usage = 0
        self.reached_prey = False
        self.belief_state = ''
        # with open('../ic3net-envs/ic3net_envs/predator_prey_env.py',"r") as f:
        #     env_code = f.read()
        #     # print(env_code)
        # self.messages = [
        #     {"role": "system", "content": 'You are playing a text game with the user.'},
        #     {"role": "user", "content": INITIAL_PROMPT},
        #     # {"role": "user", "content": INSTRUCT_PROMPT},
        # ]
        # self.messages.append({"role": "user", "content": env_code})
        # explanations = self.makeAPIcall()
        # self.messages.pop()
        # self.messages.pop()
        # self.messages.append({"role": "user", "content": explanations})
        # self.messages.append({"role": "user", "content": EMBODIED_PROMPT})
        ## agent.messages.append({"role": "user", "content": PARSER_PROMPT})
        self.reset()
    def reset(self):
        self.messages = [
            {"role": "system", "content": 'You are playing a text game with the user.'},
            {"role": "user", "content": HUMAN_EXP},
            {"role": "user", "content": EMBODIED_PROMPT.format(index = self.index, vision = self.args.vision, nagents = self.args.nagents, dim = self.args.dim, mode = self.args.mode)}
            # {"role": "user", "content": INSTRUCT_PROMPT},
        ]
        if self.belief:
            self.messages.append({"role": "user", "content": BELIEF_PROMPT + INITIAL_BELIEF})
            self.belief_state = INITIAL_BELIEF

        self.reached_prey = False


    def update_belief(self,obs):
        self.messages.append({"role": "user",
                              "content": 'Your current observation is: ' + obs + ' Update your belief about the contents of explored locations. Return only the new content list in the same format as above, with no additional information. '})
        self.belief_state = self.makeAPIcall()

        self.messages.pop()

        self.messages.append({"role": "user",
                              "content": 'Your current observation is: ' + obs })
        if self.allow_comm:
            self.messages.append({"role": "user",
                                  "content": BELIEF_PROMPT + self.belief_state + ' To reach the prey location, what is your next action and communication messages you want to share with other predators? Response with the following format: "Action Selection: Up/Right/Down/Left/Stay; Messages: *STR*."'})
        else:
            self.messages.append({"role": "user","content": BELIEF_PROMPT + self.belief_state + ' To reach the prey location, what is your next action? Response with the following format: "Action Selection: Up/Right/Down/Left/Stay."'})

    def step(self, obs,action = None):
        # if self.reached_prey:
        #     return '4', ''
        if action == None:
            if self.belief:
                self.update_belief(obs)
            else:
                if self.allow_comm:
                    self.messages.append({"role": "user",
                                          "content": 'Your current observation is: ' + obs + ' To reach the prey location, what is your next action and communication messages you want to share with other predators? Response with the following format: "Action Selection: Up/Right/Down/Left/Stay; Messages: *STR*."'})
                else:
                    self.messages.append({"role": "user",
                                          "content": 'Your current observation is: ' + obs + ' To reach the prey location, what is your next action? Response with the following format: "Action Selection: Up/Right/Down/Left/Stay."'})

            output = self.makeAPIcall()
            self.messages.append({"role": "assistant","content": output.split(';')[0]})

            action = output.split(';')[0]
            if len(output.split('Messages:'))>1:
                message = output.split('Messages:')[1]
            else:
                message = ''
            # if len(output.split('Explanations:')) > 1:
            #     exp = output.split('Explanations:')[1].split('Messages:')[0]
            # else:
            #     exp = ''

            return action,message
        else:
            if action == 0:
                text_action = 'Move Up.'
            if action == 1:
                text_action = 'Move Right.'
            if action == 2:
                text_action = 'Move Down.'
            if action == 3:
                text_action = 'Move Left.'
            if action == 4:
                text_action = 'Stay.'
            self.messages.append({"role": "user",
                                  "content": 'Your current observation is: ' + obs + ' To reach the prey location, you decide to '+text_action+' What are the communication messages you want to share with other predators? Response with the following format: "Messages: *STR*."'})
            output = self.makeAPIcall()
            self.messages.append({"role": "assistant", "content": 'Action Selection: '+text_action})

            if len(output.split('Messages:')) > 1:
                message = output.split('Messages:')[1]
            else:
                message = ''

            return action, message

        # return output

    def decode_action(self, act):
        act = act.lower()
        # act = chat.split(';')[0]
        if '0' in act or 'up' in act:
            return 0
        elif '1' in act or 'right' in act:
            return 1
        elif '2' in act or 'down' in act:
            return 2
        elif '3' in act or 'left' in act:
            return 3
        else:
            return 4


    # @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def makeAPIcall(self):

        response = client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature
        )
        self.usage += response.usage.total_tokens
        return response.choices[0].message.content

    def save(self,data_path):
        data = {}
        data['agent_id'] = self.index
        data['model'] = self.model
        data['temperature'] = self.temperature
        data['messages'] = self.messages
        timestr = time.strftime("%m-%d-%H-%M-%S", time.localtime())
        path = data_path + '{model}_{temperature}_{agent_id}_{timestr}.json'.format(timestr=timestr,agent_id = self.index, model = self.model,temperature = self.temperature)
        with open(path, 'w+', encoding='utf-8') as f:
            json.dump(data, f)

from transformers import AutoTokenizer
import requests





class PredatorPreyTextEnv(PredatorPreyEnv):

    def translate_observation_to_description(self,observation,allow_comm,comms):
        vision_range = self.vision
        grid_size = self.dim
        descriptions = []

        for agent_index, agent_obs in enumerate(observation):
            agent_desc = f"Agent {agent_index}: "
            agent_desc += f"The map is a grid world with {grid_size} * {grid_size} cells. You can only see objects within the {2*vision_range+1}*{2*vision_range+1} square surrounding your current location."
            curr_loc = self.predator_loc[agent_index]
            agent_desc += f"Your current location is {curr_loc}. "

            no_obj_flag = True
            cell_desc = ""

            for i, p in enumerate(self.predator_loc):
                if i == agent_index:
                    continue
                if abs(p[0]-curr_loc[0]) <= vision_range and abs(p[1]-curr_loc[1]) <= vision_range:
                    predator_id = i
                    agent_desc += f"You can see another predator, Agent {predator_id}, located at coordinates {p}. "
                    no_obj_flag = False

            for i, p in enumerate(self.prey_loc):
                if abs(p[0] - curr_loc[0]) <= vision_range and abs(p[1] - curr_loc[1]) <= vision_range:
                    agent_desc += f"You can see the prey located at coordinates {p}. "
                    no_obj_flag = False
                if p[0] == curr_loc[0] and p[1] == curr_loc[1]:
                    agent_desc += f"You have reached the prey location and can not move anymore. "

            if no_obj_flag:
                agent_desc += f"You do not see any predator or prey in your field of view. "

            if allow_comm:
                agent_desc += f"Communication messages from other predators: "
                for i, comm in enumerate(comms):
                    agent_desc += f"Agent {i}: '{comm}'; "

            descriptions.append(agent_desc.strip())
        return descriptions

    # def translate_observation_to_description(self,observation,allow_comm,comms):
    #     vision_range = self.vision
    #     grid_size = self.dim
    #     descriptions = []
    #     for agent_index, agent_obs in enumerate(observation):
    #         agent_desc = f"Agent {agent_index}: "
    #         agent_desc += f"The map is a grid world with {grid_size} * {grid_size} cells. You can only see objects within the {2*vision_range+1}*{2*vision_range+1} square surrounding your current location."
    #         curr_loc = self.decode_global_coordinates(agent_obs[vision_range, vision_range, :], grid_size)
    #         agent_desc += f"Your current location is {curr_loc}. "
    #
    #         no_obj_flag = True
    #         for y in range(agent_obs.shape[0]):
    #             for x in range(agent_obs.shape[1]):
    #                 cell_desc = ""
    #                 # Decode global coordinates from the observation
    #                 global_coord = self.decode_global_coordinates(agent_obs[y, x, :self.BASE], grid_size)
    #                 if agent_obs[y, x, self.PREDATOR_CLASS] == 1:
    #                     if (y, x) != (vision_range, vision_range):  # Exclude the agent itself
    #                         # predators_seen.append((x, y))
    #                         predator_id = 'UNKNOWN'
    #                         for i, loc in enumerate(self.predator_loc):
    #                             if loc[0] == global_coord[0] and loc[1] ==global_coord[1]:
    #                                 predator_id = i
    #                         cell_desc += f"You can see another predator, Agent {predator_id}, located at coordinates {global_coord}. "
    #                         no_obj_flag = False
    #
    #                 if agent_obs[y, x, self.PREY_CLASS] == 1:
    #                     # preys_seen.append((x, y))
    #
    #                     if curr_loc[0] == global_coord[0] and curr_loc[1] ==global_coord[1]:
    #                         cell_desc += f"You have reached the prey location at coordinates {global_coord}. You are not allowed to move anymore. "
    #                     else:
    #                         cell_desc += f"You can see the prey located at coordinates {global_coord}. "
    #                     no_obj_flag = False
    #
    #                 #         elif agent_obs[y, x, OUTSIDE_CLASS] == 1:
    #                 #             relative_wall_loc.append(np.array(global_coord)-np.array(curr_loc))
    #
    #                 if cell_desc:
    #                     agent_desc += cell_desc + " "
    #         if no_obj_flag:
    #             agent_desc += f"You do not see any predator or prey in your field of view. "
    #
    #         #
    #         # if len(relative_wall_loc)>0:
    #         #     print(relative_wall_loc)
    #         #     vector = np.zeros(2)
    #         #     for loc in relative_wall_loc:
    #         #         vector = vector + loc
    #         #     agent_desc += f"You can see the map boundary on your {get_relative_dire(vector,(0,0))}. "
    #         if allow_comm:
    #             agent_desc += f"Communication messages from other predators: "
    #             for i, comm in enumerate(comms):
    #                 agent_desc += f"Agent {i}: '{comm}'; "
    #
    #         descriptions.append(agent_desc.strip())
    #     return descriptions

    # def decode_global_coordinates(self,one_hot_vector, grid_size):
    #     # Assuming one_hot_vector includes the one-hot encoded global coordinates
    #     # Decode this to return the global (x, y) coordinates
    #     coord_index = np.argmax(one_hot_vector)  # Simplified for illustration
    #     x = coord_index % grid_size
    #     y = coord_index // grid_size
    #     return (y, x)

    # def get_relative_dire(global_coord,curr_loc):
    #     print(global_coord)
    #     print(curr_loc)
    #     dx = global_coord[1] - curr_loc[1]
    #     dy = global_coord[0] - curr_loc[0]
    #     direction = ""
    #     if dy <= 0:
    #         if dx <= 0:
    #             direction += f"northwest"
    #         elif dx > 0:
    #             direction += f"southwest"
    #     elif dy > 0:
    #         if dx <= 0:
    #             direction += f"northeast"
    #         elif dx > 0:
    #             direction += f"southeast"
    #     return direction.strip()


INITIAL_PROMPT = 'Given the below code wirtten in Python, explain what the gym environment is about to an embodied agent controlled by large language models. Focus on the representations of observation, action and reward.'
EMBODIED_PROMPT = 'Given the above explanation, try to play as one of the predator (i.e. Agent {index}) in the environment. The environment is initialized with grid dimension = {dim}, number of predators = {nagents}, vision range = {vision}, collaboration mode = {mode}. You will receive your observation in natural language. Your goal is to help all predators (including yourself) to reach the static prey location as soon as possible. '
BELIEF_PROMPT = 'During the task, you will need to keep track of the contents of explored areas to facilitate searching. Here is your current belief with coordinates index start from 0. \n'
INITIAL_BELIEF = "Your location: [], Empty locations: [], Predator locations: [], Prey locations: []."
PARSER_PROMPT = 'Given the above python code and explanation, write a observation parser in Python to translate vector observation into natural language descriptions.'

# with open('resource/gpt4_exp.txt', 'r') as f:
#     GPT4_EXP = f.read()
module_dir = os.path.dirname(os.path.abspath(__file__))

# with open(os.path.join(module_dir, "resource", "human_exp.txt"), 'r') as f:
#     HUMAN_EXP = f.read()
with open(os.path.join(module_dir, "resource", "human_exp_improved.txt"), 'r') as f:
    HUMAN_EXP = f.read()
