import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams['font.size'] = '16'
base_dir = '/home/hmahjoub/PycharmProjects/USAR/comm_MARL_USAR/ic3net-envs/scripts/'

fig, ax1 = plt.subplots(1,1)
#
#
#
#
#
# methods = ['supervised_256_adaEmbeddings_cos_0.1_cont','supervised_256_adaEmbeddings_cos_1_cont','reproduce_ic3net_256']
# model_names = ['supervised_0.1','supervised_1','ic3net']
methods = ['supervised_v1_exact_norm','reproduce_ic3net_256','ic3net_autoencoder_v1','mac_mha_fixed_proto_autoencoder_vqvib_v1','proto_v1','noComm_v1']
model_names = ['LangGround','IC3Net','aeComm','VQ-VIB','protoComm','noComm']
# methods = ['supervised_v0_exact_norm','ic3net_v0','ic3net_autoencoder_v0','mac_mha_fixed_proto_autoencoder_vqvib_v0','proto_v0','noComm_v0']
# model_names = ['LangGround','IC3Net','aeComm','VQ-VIB','protoComm','noComm']
# methods = ['supervised_256_adaEmbeddings_cos_1_cont','supervised_v1_exact_norm','supervised_v1_norm','supervised_v1_team_norm']
# model_names = ['supervised','supervised_exact_norm','supervised_norm','supervised_team_norm']
metrics = ['steps_taken']
mins = [0]
maxs = [25]
# seeds = [1,2,3]
epochs = 2000

# n = 16
# linestyles = ['solid', 'dotted', 'dashdot']
for method, model_name in zip(methods, model_names):
    for metric, _min, _max  in zip(metrics, mins, maxs):
        model_data = []
        sum = []
        seeds = [1, 2, 3]
        # if method == 'mac_mha_fixed_proto_autoencoder_vqvib_v0':
        #     seeds = [1,3]
        # else:
        #     seeds = [1, 2, 3]
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

        ax1.plot(X, data.reshape(-1), label=lbl, linestyle="solid")
        ax1.fill_between(X, data.reshape(-1) - std, data.reshape(-1) + std, alpha=0.2)
        print(lbl + ':' + str(data.min()))


ax1.set_xlim(0,2.51e6)
# ax1.set_xlim(0,1e7)
ax1.set_ylim(0,22)
ax1.legend()
ax1.set_ylabel('Episode length')
ax1.set_xlabel('Training timestamps')
ax1.set_yticks([0, 5, 10, 15, 20])
ax1.grid()
ax1.set_title('Predatory Prey (vision=1)')
fig.set_size_inches(8, 8)
# fig.text(0.5, 0.04, 'Steps', ha='center')
# fig.text(0.02, 0.5, 'Normalized Value', va='center', rotation='vertical')

plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))

# plt.savefig('compositional_beta.png', bbox_inches='tight')
plt.savefig('figs/v1_team_performance.png', bbox_inches='tight')
# plt.savefig('/Users/seth/Documents/research/IC3Net/TIMMAC_figs/timmac_easytj',bbox_inches='tight')
plt.show()
#
alignment_data = pd.DataFrame([['pp(v1)','w/ alignment',0.825810851081924,0.661613363415467],
['pp(v1)','w/ alignment',0.829323948165506,0.680962406888039],
['pp(v1)','w/ alignment',0.824202881699272,0.629044145333874],
['pp(v0)','w/ alignment',0.811321190692335,0.665635318578854],
['pp(v0)','w/ alignment',0.791332941251981,0.554838940259629],
['pp(v0)','w/ alignment',0.804038534799917,0.772754749848332],
['pp(v1)','w/o alignment',0.014464577778911,0.153335226589946],
['pp(v1)','w/o alignment',-0.014515456365431,0.125946824432761],
['pp(v1)','w/o alignment',0.014927368428439,0.128609882350421],
['pp(v0)','w/o alignment',0.004968959040781,0.204305736258175],
['pp(v0)','w/o alignment',-0.056076423765564,0.2454510268436],
['pp(v0)','w/o alignment',0.021727530083823,0.218358217241289]],columns=['env','condition','cos sim','bleu score'])

alignment_data = pd.melt(alignment_data, id_vars=['env', 'condition'], var_name='measurement', value_name='value')


fig, ax1 = plt.subplots(1,1)
sns.barplot(alignment_data[alignment_data['env']=='pp(v0)'],x = "measurement", y = 'value', hue = 'condition',errorbar='se',legend = False)
ax1.set_ylim(0,1)
ax1.set_xlabel("")
ax1.set_title("Predator Prey (vision=0)")
plt.tight_layout()
fig.set_size_inches(8, 8)
plt.savefig('figs/v0_alignment.png', bbox_inches='tight')
fig, ax1 = plt.subplots(1,1)
sns.barplot(alignment_data[alignment_data['env']=='pp(v1)'],x = "measurement", y = 'value', hue = 'condition',errorbar='se')
ax1.set_ylim(0,1)
ax1.set_xlabel("")
ax1.set_title("Predator Prey (vision=1)")
plt.tight_layout()
fig.set_size_inches(8, 8)
plt.savefig('figs/v1_alignment.png', bbox_inches='tight')
