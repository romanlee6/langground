import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

lg_paths= []

len = []
num_episode = 0
success = 0
timestamp_list = []
entropy_list = []
fig, ax = plt.subplots()
# for path in paths:
for path in lg_paths:

    for root, dirs, files in os.walk(path):
        # Get the full path of the file
        for filename in files:
            file_path = os.path.join(root, filename)

            # Check if the path is a file (not a directory)
            if file_path.endswith('.csv'):
                num_episode +=1
                data = pd.read_csv(file_path,names = ['step', 'agent_id', 'action', 'comm','entropy','crop_comm', 'room', 'bomb', 'sequence', 'score'])
                # data = pd.read_csv(file_path,names=['step', 'agent_id', 'obs','action', 'comm', 'entropy', 'reached_prey', 'reward', 'prey_loc','predator_loc'],index_col=False)
                # first_reached_agent = data[data['reached_prey']==1].iloc[0]['agent_id']
                # anchor = data[data['reached_prey'] == 1].iloc[0]['step']
                # if first_reached_agent == 0:
                #     timestamps = data[data['agent_id']!=0]['step'] - anchor -1
                #     entropy = data[data['agent_id'] != 0]['entropy']
                #     timestamp_list+=timestamps.to_list()
                #     entropy_list+=entropy.to_list()
                length = data.shape[0] / 3
                if length <50:
                    success += 1
                len.append(length)

# sns.lineplot(x=timestamp_list, y=entropy_list, errorbar = 'se',c = 'blue' ,ax =ax)
    #plt.show()
print(success/num_episode)
print(np.mean(len))
print(np.std(len))

#
# len = []
# num_episode = 0
# success = 0
# timestamp_list = []
# entropy_list = []
# for path in ae_paths:
#
#     for root, dirs, files in os.walk(path):
#         # Get the full path of the file
#         for filename in files:
#             file_path = os.path.join(root, filename)
#
#             # Check if the path is a file (not a directory)
#             if file_path.endswith('.csv'):
#                 num_episode +=1
#                 #data = pd.read_csv(file_path,names = ['step', 'agent_id', 'action', 'comm','entropy','crop_comm', 'room', 'bomb', 'sequence', 'score'])
#                 data = pd.read_csv(file_path,names=['step', 'agent_id', 'obs','action', 'comm', 'entropy', 'reached_prey', 'reward', 'prey_loc','predator_loc'],index_col=False)
#                 if data[data['reached_prey']==1].shape[0]>0:
#                     first_reached_agent = data[data['reached_prey']==1].iloc[0]['agent_id']
#                     anchor = data[data['reached_prey'] == 1].iloc[0]['step']
#                     if first_reached_agent == 0:
#                         timestamps = data[data['agent_id']!=0]['step'] - anchor -1
#                         entropy = data[data['agent_id'] != 0]['entropy']
#                         timestamp_list+=timestamps.to_list()
#                         entropy_list+=entropy.to_list()
#                 length = data.shape[0] / 3
#                 if length <20:
#                     success += 1
#                 len.append(length)
#
# sns.lineplot(x=timestamp_list, y=entropy_list, errorbar = 'se',c = 'red',ax =ax)
# plt.show()
# print(success/num_episode)
# print(np.mean(len))
# print(np.std(len))



