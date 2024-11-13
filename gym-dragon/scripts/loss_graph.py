import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = '16'
base_dir = 'gym-dragon/scripts/'

fig, (ax, ax1) = plt.subplots(2,1)

env = "mini_dragon"



# methods = ['supervised_1','ic3net','supervised_1_fixed','ic3net_fixed']
# model_names = ['Supervised_ic3net','ic3net','Supervised_CommNet','CommNet']
# methods = ['ic3net_proto_autoencoder','ic3net_fixed_autoencoder','ic3net_autoencoder']
methods = ['supervised_1','ic3net','ic3net_autoencoder','proto_58','ic3net_fixed','noComm']
model_names = ['supervised','icenet','ae_comm','proto','commNet','noComm']
metrics = ['Score', 'steps_taken']
seeds = [12,13,14]
mins = [0, 0]
maxs = [90, 100]
epochs = 2000
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# n = 16
linestyles = ['solid', 'dotted', 'dashdot']
for method, model_name in zip(methods, model_names):
    for metric, _min, _max  in zip(metrics, mins, maxs):
        model_data = []
        if method == 'ic3net_autoencoder':
            seeds = [12,14]
        else:
            seeds = [12,13,14]
        sum = []
        for seed in seeds:
            s = str(seed)
            data_success = np.load(base_dir + 'trained_models/mini_dragon/'+env+'_'+method+'/seed'+s+'/logs/'+metric+'.npy')
            if len(data_success) > epochs:
                data_success = data_success[:epochs]
            data_success = np.convolve(data_success, np.ones((10,))/10, mode='valid')
            sum.append(data_success)
        sum = np.array(sum)
        data = np.mean(sum, axis = 0)
        std = np.std(sum, axis=0) / np.sqrt(3)
        X = np.arange(data.shape[0])*10*500
        lbl = model_name

        if metric == 'Score':
            ax.plot(X, data.reshape(-1), label=lbl, linestyle='solid')
            ax.fill_between(X, data.reshape(-1) - std, data.reshape(-1) + std, alpha=0.2)

            print(lbl + ':' + str(data.max()))
        else:
            ax1.plot(X, data.reshape(-1), label=lbl, linestyle='solid')
            ax1.fill_between(X, data.reshape(-1) - std, data.reshape(-1) + std, alpha=0.2)
            print(lbl + ':' + str(data.min()))



ax1.set_xlim(0,1e7)
ax1.set_ylim(0,100)
ax1.legend(loc = 'upper right')
ax1.set_ylabel('Episode Length')
ax1.grid()

ax.set_xlim(0,1e7)
ax.set_ylim(0,100)
ax.set_title('Team Performance')

ax1.set_xlabel('Training Timesteps')
ax.set_ylabel('Score')
fig.set_size_inches(8, 12)
# fig.text(0.5, 0.04, 'Steps', ha='center')
# fig.text(0.02, 0.5, 'Normalized Value', va='center', rotation='vertical')
ax.grid()
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.savefig('compositional_beta.png', bbox_inches='tight')
plt.savefig('figs/log_new.png', bbox_inches='tight')

plt.show()
