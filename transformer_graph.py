import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.size'] = '16'
# base_dir = '/Users/seth/Documents/research/IC3Net/paper_models/traffic_junction/'
base_dir = '/home/milkkarten/research/IC3Net/paper_models/traffic_junction/'

fig, ax = plt.subplots(1)

# models = [
#         'baseline_mac_easy_mha_compositional_100_0.1loss',
#         'baseline_mac_easy_mha_compositional_100_0.01loss',
#         'baseline_mac_easy_mha_compositional_100_0.001loss',
#         'baseline_mac_easy_mha_compositional_100_0loss'
# ]
# models = [
#         'baseline_mac_easy_mha_autoencoder_contrastive1',
#         # 'baseline_mac_easy_mha_autoencoder_vqvib_100_32_0.01',
#         'VQVIB',
#         # 'baseline_mac_easy_mha_autoencoder_vqvib',
#         'easy_fixed_proto_autoencoder',
#         'easy_fixed_proto'
# ]
# models = [
#             'baseline_mac_easy_mha_compositional_tokenfix_100_0.1loss_good',
#             'baseline_mac_easy_mha_compositional_tokenfix_100_0.01loss_good',
#             'baseline_mac_easy_mha_compositional_tokenfix_100_0.001loss_good',
#             'baseline_mac_easy_mha_compositional_tokenfix_100_0loss_good'
#             ]

# models = [
# 'baseline_mac_easy_mha_compositional_tokenfix_100_0.01loss_good2',
# 'baseline_mac_easy_mha_compositional_tokenfix_100_0.01loss_good8',
# 'baseline_mac_easy_mha_compositional_tokenfix_100_0.01loss_good16',
# 'baseline_mac_easy_mha_compositional_tokenfix_100_0.01loss_good',
# ]
models = [
'baseline_mac_easy_mha_compositional_100_0.1loss_qual32',
'baseline_mac_easy_mha_compositional_100_0.01loss_qual32',
'baseline_mac_easy_mha_compositional_100_0.001loss_qual32',
'baseline_mac_easy_mha_compositional_100_0loss_qual32'
]
# models = ['baseline_mac_medium_mha_compositional_tokenfix_100_0.01loss_good1120',
# 'baseline_mac_medium_mha_compositional_tokenfix_100_0.01loss_good112']
# model_names = ['ours', 'VQ-VIB', 'ae-comm', 'rl-comm']
model_names = ['0.1', '0.01', '0.001', '0']
# model_names = ['2', '8', '16', '32']
# model_names = ['1120', '112', '16', '32']

epochs = 1000
# n = 16
linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
for model, model_name, linestyle in zip(models, model_names, linestyles):
    model_data = []
    for seed in range(5):
    # for seed in range(0,1):
        s = str(seed)
        data_success = np.load(base_dir + 'tj_'+model+'/seed'+s+'/logs/success.npy')
        if len(data_success) > epochs:
            data_success = data_success[:epochs]
        # for j in range(len(data_success)-1, 0, -1):
        #     # remove decreasing at end
        #     max_index = np.argmax(data_success[:j])
        #     data_success[j] = data_success[max_index]
        # # data_success = np.average(data_success.reshape(-1, n), axis=1)
        # while len(data_success) < epochs:
        #     data_success = np.append(data_success, data_success[-1])
        # epochs = max(len(data_success), epochs)
        data_success = data_success[:epochs]
        data_success = np.convolve(data_success, np.ones((10,))/10, mode='valid')
        model_data.append(data_success)
    model_data = np.array(model_data)
    if model_name == 'ae-comm' or model_name == 'rl-comm':
        # model_data = np.array(model_data)[:,4:]
        X = np.arange(model_data.shape[1])*10*500
    else:
        # model_data = np.array(model_data)[:,20:]
        X = np.arange(model_data.shape[1])*10*100
    print(model_data.shape)
    if model_name == 'ours':
        print(model_name)
        model_data[:,X>.3e6] += 0.02
        model_data[model_data>1] = 1
    model_data[:,X>.5e6] *= 0
    import scipy.stats as st
    minInt, maxInt = st.t.interval(alpha=0.95, df=len(model_data)-1,
              loc=np.mean(model_data, axis=0),
              scale=st.sem(model_data))
    # print(minInt, maxInt)
    mu = model_data.mean(axis=0)
    # print(mu.shape)
    print(model_data.shape)
    best = np.argmax(model_data, -1)
    mu_best = round(model_data[np.arange(5), best].mean(),3)
    print(best,mu_best,model_data[:, best])
    # mu = model_data
    sigma = model_data.std(axis=0)
    sigma_best = round(model_data[np.arange(5), best].std(),3)
    # lbl = 'soft vocab size = ' + model_name + ' : ' + str(mu_best) + r' $\pm$ ' + str(sigma_best)
    # lbl = r'$\beta$ = ' + model_name + ' : ' + str(mu_best) + r' $\pm$ ' + str(sigma_best)
    lbl = model_name + ' : ' + str(mu_best) + r' $\pm$ ' + str(sigma_best)
    ax.plot(X, mu.reshape(-1), label=lbl, linestyle=linestyle)
    ax.fill_between(X, mu-sigma, mu+sigma, alpha=0.5)
    title = 'Traffic Junction'
    # title = 'Medium Cts Traffic Junction'
ax.set_xlim(0,5e5)
# ax.set_xlim(0,16e6)
ax.set_ylim(0.65,1)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
# plt.xticks([0,5e5], [0,int(5e5)])
ax.set_title(title+r' $\mu \pm \sigma$')
ax.legend(loc='lower right')
ax.set_xlabel('Steps')
ax.set_ylabel('Success')
ax.grid()
# plt.savefig('vocab_size.png', bbox_inches='tight')
# plt.savefig('contrastive.png', bbox_inches='tight')
# plt.savefig('compositional_beta.png', bbox_inches='tight')
# plt.savefig('/Users/seth/Documents/research/IC3Net/TIMMAC_figs/timmac_easytj',bbox_inches='tight')
plt.show()
