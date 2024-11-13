import numpy as np
import os
import pandas as pd
offline_path = 'archived_data/pp_prompt/gpt-4-turbo-preview/comm/offline_data'
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


# vision = 1
#
# loc_to_comm = {}
# loc_to_comm_crop = {}
# path = 'data/pp_ez_v1/gpt-4-turbo-preview/default'
# valid_episodes = 0
# valid_steps = 0
# dataset = []
# for subdir, dirs, files in os.walk(path):
#     for file in files:
#         #print os.path.join(subdir, file)
#         filepath = subdir + os.sep + file
#         if filepath.endswith("summary.csv"):
#             data = pd.read_csv(filepath,names = ['step','agent_id','obs ','action','comm','exp','reward','prey_loc','predator_loc'],index_col=False)
#             if data.shape[0]>=59:
#                 continue
#             valid_episodes+=1
#             for i, row in data.iterrows():
#                 if not isinstance(row['comm'],str):
#                     continue
#                 full_comm = row['comm']
#                 crop_comm = row['comm'].split('.')[0]
#                 predator_loc = [int(x) for x in row['predator_loc'].strip('[]').split()]
#                 prey_loc = [int(x) for x in row['prey_loc'].strip('[]').split()]
#                 prey_in_fov = abs(predator_loc[0]-prey_loc[0]) <= vision and abs(predator_loc[1]-prey_loc[1]) <= vision
#                 loc_tuple = (predator_loc[0],predator_loc[1],int(prey_in_fov))
#
#                 action = decode_action(row['action'])
#
#                 dataset.append([predator_loc[0],predator_loc[1],int(prey_in_fov),action,full_comm,crop_comm])
#                 # if loc_tuple not in loc_to_comm.keys():
#                 #     loc_to_comm[loc_tuple] = []
#                 # loc_to_comm[loc_tuple].append(full_comm)
#                 # if loc_tuple not in loc_to_comm_crop.keys():
#                 #     loc_to_comm_crop[loc_tuple] = []
#                 # loc_to_comm_crop[loc_tuple].append(crop_comm)
#                 valid_steps+=1
#
# dataset = pd.DataFrame(dataset,columns=['predator_y','predator_x','prey_in_fov','action','full_comm','crop_comm'])
# dataset.to_csv('offline_llm_dataset_v1.csv',index = False)
# print(len(loc_to_comm.keys()))
# print(valid_steps)
# print(valid_episodes)

#
#
#
# from ast import literal_eval
#
# state_to_comm = {}
# data_path = 'embedded_256_offline_llm_dataset.csv'
# offline_data = pd.read_csv(data_path)
# offline_data["embedding"] = offline_data.ada_embedding.apply(literal_eval).apply(np.array)
# for i, row in offline_data.iterrows():
#     key_tuple = (int(row['predator_y']),int(row['predator_x']),int(row['prey_in_fov']),int(row['action']))
#     if key_tuple not in state_to_comm.keys():
#         state_to_comm[key_tuple] = []
#     state_to_comm[key_tuple].append(np.array(row["crop_comm"]))
# print(state_to_comm)


# text_wPrey = ''
# text_woPrey = ''
# for loc, comms in loc_to_comm_crop.items():
#     if loc[2] == 0:
#         for comm in comms:
#             text_woPrey+=comm
#             text_woPrey += ' '
#     else:
#         for comm in comms:
#             text_wPrey+=comm
#             text_wPrey += ' '
#
# # Download NLTK stopwords
# nltk.download('stopwords')
# nltk.download('punkt')
#
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# def create_wordcloud(text):
#     # Tokenize the text
#     words = word_tokenize(text)
#
#     # Filter out stopwords and punctuation
#     stop_words = set(stopwords.words('english'))
#     words_filtered = [word for word in words if word.casefold() not in stop_words and word.isalpha()]
#     cleaned_text = ' '.join(words_filtered)
#
#     # Generate a word cloud image
#     wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=400).generate(cleaned_text)
#
#     # Display the generated image:
#     plt.figure(figsize=(10, 5))
#     plt.imshow(wordcloud, interpolation='bilinear')
#     plt.axis("off")
#     plt.show()
#
# create_wordcloud(text_wPrey)
# create_wordcloud(text_woPrey)


#
# vision = 0
#
# loc_to_comm = {}
# loc_to_comm_crop = {}
# path = 'data/pp_ez_v0/gpt-4-turbo-preview/default'
# valid_episodes = 0
# valid_steps = 0
# dataset = []
#
# embeddings = pd.read_csv('embedded_256_offline_llm_dataset_v0.csv')
# # embeddings["embedding"] = embeddings.ada_embedding.apply(literal_eval).apply(np.array)
#
#
# for subdir, dirs, files in os.walk(path):
#     for file in files:
#         #print os.path.join(subdir, file)
#         filepath = subdir + os.sep + file
#         if filepath.endswith("summary.csv"):
#             data = pd.read_csv(filepath,names = ['step','agent_id','obs ','action','comm','exp','reward','prey_loc','predator_loc'],index_col=False)
#
#             if data.shape[0]>=59:
#                 continue
#             valid_episodes+=1
#             current_step = 0
#             temp = []
#             for i, row in data.iterrows():
#                 if int(row['step'])!=current_step:
#                     permutations = itertools.permutations(temp)
#                     permutations_list = list(permutations)
#                     for perm in permutations_list:
#                         d = [x['state'] for x in perm]
#                         for x in perm:
#                             e = embeddings[embeddings["crop_comm"] == x['comm']]["ada_embedding"]
#
#                             if len(e) > 0:
#                                 d.append(e.values[0])
#                             else:
#                                 d.append(null_embedding)
#                         dataset.append(d)
#                     current_step += 1
#                     temp = []
#
#                 if not isinstance(row['comm'],str):
#                     full_comm = ''
#                     crop_comm = ''
#                 else:
#                     full_comm = row['comm']
#                     crop_comm = row['comm'].split('.')[0]
#                 predator_loc = [int(x) for x in row['predator_loc'].strip('[]').split()]
#                 prey_loc = [int(x) for x in row['prey_loc'].strip('[]').split()]
#                 prey_in_fov = abs(predator_loc[0]-prey_loc[0]) <= vision and abs(predator_loc[1]-prey_loc[1]) <= vision
#                 action = decode_action(row['action'])
#                 loc_tuple = (predator_loc[0],predator_loc[1],int(prey_in_fov),int(action))
#                 agent_id = int(row['agent_id'])
#
#                 temp.append({'state': loc_tuple,'comm' : crop_comm})
#
#                 # if loc_tuple not in loc_to_comm.keys():
#                 #     loc_to_comm[loc_tuple] = []
#                 # loc_to_comm[loc_tuple].append(full_comm)
#                 # if loc_tuple not in loc_to_comm_crop.keys():
#                 #     loc_to_comm_crop[loc_tuple] = []
#                 # loc_to_comm_crop[loc_tuple].append(crop_comm)
#                 valid_steps+=1
#
# dataset = pd.DataFrame(dataset,columns=['state0','state1','state2','comm0','comm1','comm2'])
# dataset.to_csv('embedded_256_offline_llm_dataset_v0_team.csv',index = False)
# # print(len(loc_to_comm.keys()))
# print(valid_steps)
# print(valid_episodes)
