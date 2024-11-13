import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = '16'
base_dir = '/home/hmahjoub/PycharmProjects/USAR/comm_MARL_USAR/ic3net-envs/scripts/'

fig, (ax, ax1) = plt.subplots(2,1)





# methods = ['supervised_256_adaEmbeddings_cos_0.1_cont','supervised_256_adaEmbeddings_cos_1_cont','reproduce_ic3net_256']
# model_names = ['supervised_0.1','supervised_1','ic3net']
methods = ['supervised_256_adaEmbeddings_cos_1_cont','reproduce_ic3net_256','ic3net_autoencoder_v1','mac_mha_fixed_proto_autoencoder_vqvib_v1','proto_v1','noComm_v1']
model_names = ['supervised','ic3net','ae_comm','vqvib','proto','noComm']
# methods = ['supervised_256_adaEmbeddings_cos_1_cont','supervised_v1_exact_norm','supervised_v1_norm','supervised_v1_team_norm']
# model_names = ['supervised','supervised_exact_norm','supervised_norm','supervised_team_norm']
metrics = ['success', 'steps_taken']
mins = [0, 0]
maxs = [1, 20]
epochs = 500
ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# n = 16
# linestyles = ['solid', 'dotted', 'dashdot']
for method, model_name in zip(methods, model_names):
    for metric, _min, _max  in zip(metrics, mins, maxs):
        model_data = []
        if method == 'supervised_v1_team_norm':
            seeds = [11]
        else:
            seeds = [1,2,3]
        sum = []
        for seed in seeds:
            s = str(seed)
            data_success = np.load(base_dir + 'trained_models/predator_prey/'+method+'/seed'+s+'/logs/'+metric+'.npy')
            if len(data_success) > epochs:
                data_success = data_success[:epochs]
            data_success = np.convolve(data_success, np.ones((10,))/10, mode='valid')
            sum.append(data_success)
        sum = np.array(sum)
        data = np.mean(sum, axis = 0)
        std = np.std(sum, axis=0) / np.sqrt(3)
        X = np.arange(data.shape[0])*10*500
        lbl = model_name

        if metric == 'success':
            ax.plot(X, data.reshape(-1), label=lbl, linestyle="solid")
            ax.fill_between(X, data.reshape(-1) - std, data.reshape(-1) + std, alpha=0.2)
            print(lbl + ':' + str(data.max()))
        else:
            ax1.plot(X, data.reshape(-1), label=lbl, linestyle="solid")
            ax1.fill_between(X, data.reshape(-1) - std, data.reshape(-1) + std, alpha=0.2)
            print(lbl + ':' + str(data.min()))




ax1.set_ylim(0,20)
ax1.legend()
ax1.set_ylabel('Episode length')
ax1.grid()


ax.set_ylim(0,1)
ax.set_title('Team Performance')

ax.set_xlabel('Training Timesteps')
ax.set_ylabel('Success rate')
fig.set_size_inches(16, 12)
# fig.text(0.5, 0.04, 'Steps', ha='center')
# fig.text(0.02, 0.5, 'Normalized Value', va='center', rotation='vertical')
ax.grid()
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.savefig('compositional_beta.png', bbox_inches='tight')
plt.savefig('log_new.png', bbox_inches='tight')
# plt.savefig('/Users/seth/Documents/research/IC3Net/TIMMAC_figs/timmac_easytj',bbox_inches='tight')
plt.show()
