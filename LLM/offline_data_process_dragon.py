import numpy as np
import os
import pandas as pd
# offline_path = 'archived_data/pp_prompt/gpt-4-turbo-preview/comm/offline_data'
import nltk
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import itertools
from ast import literal_eval
'''
The saved npy files contain a dictionary with the following keys
args: arguments for experimental and env settings
obs: a list of observations for all agents, in shape (n_agent, vision,vision,cell_features). in the case of easy pp, it is (3,3,3,29)
# data['action'] = action
# data['reward'] = reward
# data['done'] = done
# data['comm'] = comm
# data['predator_locs'] = info['predator_locs']
# data['prey_locs'] = info['prey_locs']
'''

''
from openai import OpenAI
client = OpenAI(api_key='na')


def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model, dimensions=256).data[
        0].embedding

sourse_dataset = pd.read_csv('embedded_256_offline_llm_dataset_dragon_large.csv')
#sourse_dataset["embedding"] = sourse_dataset.ada_embedding.apply(literal_eval).apply(np.array)

def retr_embedding(text):
    text = text.replace("\n", " ")
    x = sourse_dataset[sourse_dataset["crop_comm"] == text]
    if x.shape[0]>0:
        return x.iloc[0]["ada_embedding"]
    else:
        return client.embeddings.create(input=[text], model="text-embedding-3-large", dimensions=256).data[
        0].embedding

null_embedding = get_embedding(' ')

# for subdir, dirs, files in os.walk(path):
#     for file in files:
#         #print os.path.join(subdir, file)
#         filepath = subdir + os.sep + file
#         if filepath.endswith(".npy"):
#             data = np.load(filepath,allow_pickle = True)
#             for i in range(data.item()['args']['nagents']):
#                 obs = data.item()['obs'][i]
#                 action = data.item()['action'][i]
#                 comm = data.item()['comm'][i]
#                 predator_loc = data.item()['predator_locs'][i]
#                 prey_loc = data.item()['prey_locs'][0]
def decode_action(str):
    if 'up' in str.lower():
        return 0
    elif 'right' in str.lower():
        return 1
    elif 'down' in str.lower():
        return 2
    elif 'left' in str.lower():
        return 3
    else:
        return 5



for step_cap in [1000,1500,2000]:
    loc_to_comm = {}
    loc_to_comm_crop = {}
    path = 'data/dragon/gpt-4-turbo-preview/default_exp'
    valid_episodes = 0
    valid_steps = 0
    dataset = []
    for subdir, dirs, files in os.walk(path):
        for file in files:
            if valid_steps > step_cap:
                break
            #print os.path.join(subdir, file)
            filepath = subdir + os.sep + file
            if filepath.endswith("summary.csv"):
                data = pd.read_csv(filepath,index_col=False)

                valid_episodes+=1
                for i, row in data.iterrows():
                    if not isinstance(row['comm'],str):
                        continue
                    full_comm = row['comm']
                    crop_comm = row['crop_comm']
                    room = int(row['room'])
                    bomb = int(row['bomb'])
                    sequence = [int(x) for x in row['sequence'].strip('[]').split()]
                    while len(sequence) <=3:
                        sequence.append(-1)


                    action = int(row['action'])

                    dataset.append([room,bomb,sequence[0],sequence[1],sequence[2],action,full_comm,crop_comm])

                    valid_steps+=1

    dataset = pd.DataFrame(dataset,columns=['room','bomb','seq0','seq1','seq2','action','full_comm','crop_comm'])
    #dataset.to_csv('offline_llm_dataset_dragon_large.csv',index = False)
    print(len(loc_to_comm.keys()))
    print(valid_steps)
    print(valid_episodes)
    dataset['ada_embedding'] = dataset.crop_comm.apply(lambda x: retr_embedding(x))
    dataset.to_csv('embedded_256_offline_llm_dataset_dragon_{step_cap}.csv'.format(step_cap = step_cap),index = False)
