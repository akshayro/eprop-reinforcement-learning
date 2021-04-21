import numpy as np
import matplotlib.pyplot as plt
import spikes4 as spk4
import snn_models

#%% parameters
time_step = 1 # 1ms time resolution

lr = 0.002 # learning rate 0.005
thr = 0.6 # v threshold
tau_out = 3 # output decay time constant /// unit: ms
kappa = np.exp(-time_step/tau_out) # output decay
tau_mem = 20 # v decay time constant /// unit: ms
alpha = np.exp(-time_step/tau_mem) # v decay
beta = 0.2 # adaptation increment
tau_a = 200 # adaptation decay time constant /// unit: ms
rho = np.exp(-time_step/(tau_a+1e-12)) # adaptation decay
t_ref = 2 # refractory period /// unit: ms

decays = np.array([lr, thr, alpha, beta, kappa, rho, t_ref]) # save as a vector

#%% input data generation
nb_inputs, nb_hidden, nb_outputs  = 40, 100, 2 # number of neurons
batch_size = 256
nb_batch = 10 # number of batches
nb_data = batch_size*nb_batch # =256*10
n_train = batch_size*(nb_batch-1) # =256*9
n_test = batch_size # =256*1
nb_steps = 500 # simulation duration (ms)
dur_stim = 100 # stimulation duration (ms)

x = np.zeros((nb_batch,batch_size,nb_steps,40)) # input dta train
y1hot = np.zeros((nb_batch,batch_size,nb_steps,2)) # target data (1hot encoding)
y = np.zeros((nb_batch,batch_size,nb_steps,)) # target data (0 for left, 1 for right, 0.5 otherwise)

for batch_idx in range(nb_batch):
    for b in range(batch_size):
        y_data = 0.5*np.ones((nb_steps))
        y_data1hot = np.zeros((nb_steps, 2))
        cue_on = np.zeros((nb_steps),dtype=bool) # learning cue, True when learing cue is on.

        prob_noise = 1*(time_step/1000) #!!! noise has 1Hz spike frequency
        x_data = np.random.choice(2, size=(nb_steps,nb_inputs),p=[1-prob_noise,prob_noise])

        t_stim = np.random.randint(0,350) # stimuli given random time interval [0,350]ms
        t_learn = 400 # learning cue starts at 400ms
        prob_stim = 40*(time_step/1000) #!!! input has 40Hz spike frequency
        x_stim = np.random.choice(2, size=(dur_stim,int(nb_inputs/4)),p=[1-prob_stim,prob_stim]).astype(float)
        l_or_r = np.random.choice(2) # left or right?
        x_data[t_stim:t_stim+dur_stim, int(nb_inputs/4)*l_or_r:int(nb_inputs/4)*l_or_r+int(nb_inputs/4)] = x_stim

        x_learn = np.random.choice(2, size=(dur_stim,int(nb_inputs/4)),p=[1-prob_stim,prob_stim]).astype(float)
        x_data[t_learn:t_learn+dur_stim, int(nb_inputs/4)*2:int(nb_inputs/4)*2+int(nb_inputs/4)] = x_learn

        y_data1hot[t_learn:t_learn+dur_stim, l_or_r] = 1
        cue_on[t_learn:t_learn+dur_stim] = True
        y_data[t_learn:t_learn+dur_stim] = l_or_r

        x[batch_idx,b] = x_data
        y[batch_idx,b] = y_data
        y1hot[batch_idx,b] = y_data1hot
print('data generated.')
stim = 2*y-1 # same as y, but -1 for left, 1 for right, and 0 otherwise

print(x.shape)
print(y.shape)
print(y1hot.shape)
print(cue_on.shape)

#%% initialize weight
weight_scale = 10*(1.0-alpha) #!!!
w1 = np.random.normal(size=(nb_inputs,nb_hidden), loc=0.0, scale=weight_scale/np.sqrt(nb_inputs)) # input-->hidden
w2 = np.random.normal(size=(nb_hidden,nb_outputs), loc=0.0, scale=weight_scale/np.sqrt(nb_hidden)) # hidden-->hidden
bias = np.random.normal(size=(nb_outputs), loc=0.0, scale=weight_scale/np.sqrt(nb_outputs)) # output bias
#bias = np.zeros(nb_outputs) # output bias
B = np.random.normal(size=(nb_outputs,nb_hidden), loc=0.0, scale=weight_scale/np.sqrt(nb_outputs)) #!!! random e-prop

#%% recurrent weights (choose only one)
############### (1) random network ###############
# wr = np.random.normal(size=(nb_hidden,nb_hidden), loc=0.0, scale=weight_scale/np.sqrt(nb_hidden)) # hidden-->output
# np.fill_diagonal(wr,0) # no self connectivity

############### (2) E/I network ###############
import graph_analysis
nb_e = int(nb_hidden*0.8)
nb_i = nb_hidden - nb_e
ind_fromi = np.zeros((nb_hidden,nb_hidden),dtype=bool)
ind_fromi[nb_e:,:] = True # index of inhibitory neurons

wr, xy, cdens = graph_analysis.aij_distance2d([1000,1000], [250,150], nb_e, nb_i, np.array([[1,-1],[1,-1]]), cself=False, plot=False, randomseed=None)
wr = wr.T
wr *= 0.05 # scaling

plt.imshow(wr) # visualize initial weights

#%% compile (1st run numba is slow)
x_tmp, y1hot_tmp, cue_on_tmp = x[0,:,:3,:].copy(),y1hot[0,:,:3,:].copy(),cue_on[:3].copy()
loss, out_rec, dw1, dwr, dw2, dbias, v_rec, z_rec, a_rec = snn_models.lif_eprop(w1,wr,w2,bias,B,x_tmp, y1hot_tmp, cue_on_tmp,decays)
print('compile done.')

#%% run train
dw1_past = np.zeros((nb_inputs,nb_hidden))
dw2_past = np.zeros((nb_hidden,nb_outputs))
dwr_past = np.zeros((nb_hidden,nb_hidden))
dbias_past = np.zeros((nb_outputs,))
loss_train, fr_train, acc_train, spktr, wrs, atr = [],[],[],[],[],[]
loss_valid, fr_valid, acc_valid = [],[],[]
n_epochs = 16

# Adam
m_t_w1 = np.zeros((nb_inputs,nb_hidden))
m_t_w2 = np.zeros((nb_hidden,nb_outputs))
m_t_wr = np.zeros((nb_hidden,nb_hidden))
m_t_bias = np.zeros((nb_outputs,))

v_t_w1 = np.zeros((nb_inputs,nb_hidden))
v_t_w2 = np.zeros((nb_hidden,nb_outputs))
v_t_wr = np.zeros((nb_hidden,nb_hidden))
v_t_bias = np.zeros((nb_outputs,))

b1 = 0.9
b2 = 0.999
alpha_lr = 0.002
eps = 1e-9


for epoch in range(n_epochs): # 100
    ######################### train #########################
    loss_batch,acc_batch,fr_batch = np.zeros((nb_batch-1,)), np.zeros((nb_batch-1,)), np.zeros((nb_batch-1,))
    for batch_idx in range(nb_batch-1): # batch_idx = 0
        x_batch,y1hot_batch,y_batch = x[batch_idx],y1hot[batch_idx],y[batch_idx]
        loss, out_rec, dw1, dwr, dw2, dbias, v_rec, z_rec, a_rec = snn_models.lif_eprop(w1,wr,w2,bias,B,x_batch,y1hot_batch,cue_on,decays)

        # Implementation Adam
        m_t_w1 = b1 * m_t_w1 + (1 - b1) / (-lr) * np.mean(dw1, 0)
        m_t_w2 = b1 * m_t_w2 + (1 - b1) / (-lr) * np.mean(dw2, 0)
        m_t_wr = b1 * m_t_wr + (1 - b1) / (-lr) * np.mean(dwr, 0)
        m_t_bias = b1 * m_t_bias + (1 - b1) / (-lr) * np.mean(dbias, 0)

        v_t_w1 = b2 * v_t_w1 + (1 - b2) / (lr**2) * np.power(np.mean(dw1, 0), 2)
        v_t_w2 = b2 * v_t_w2 + (1 - b2) / (lr**2) * np.power(np.mean(dw2, 0), 2)
        v_t_wr = b2 * v_t_wr + (1 - b2) / (lr**2) * np.power(np.mean(dwr, 0), 2)
        v_t_bias = b2 * v_t_bias + (1 - b2) / (lr**2) * np.power(np.mean(dbias, 0), 2)

        t = (epoch + 1)

        m_w1_corr = m_t_w1 / (1. - b1**t)
        m_w2_corr = m_t_w2 / (1. - b1**t)
        m_wr_corr = m_t_wr / (1. - b1**t)
        m_bias_corr = m_t_bias / (1. - b1**t)

        v_w1_corr = v_t_w1 / (1. - b2**t)
        v_w2_corr = v_t_w2 / (1. - b2**t)
        v_wr_corr = v_t_wr / (1. - b2**t)
        v_bias_corr = v_t_bias / (1. - b2**t)

        div_w1 = -alpha_lr * m_w1_corr / (np.sqrt(v_w1_corr) + eps * np.ones_like(v_w1_corr))
        div_w2 = -alpha_lr * m_w2_corr / (np.sqrt(v_w2_corr) + eps * np.ones_like(v_w2_corr))
        div_wr = -alpha_lr * m_wr_corr / (np.sqrt(v_wr_corr) + eps * np.ones_like(v_wr_corr))
        div_bias = -alpha_lr * m_bias_corr / (np.sqrt(v_bias_corr) + eps * np.ones_like(v_bias_corr))

        w1 += div_w1 # input-->hidden update
        w2 += div_w2 # hidden-->hidden update
        wr += div_wr # hidden-->output update
        bias += div_bias # bias update

        # momentum update
        # dw1_past = 0.9*dw1_past + np.mean(dw1,0) #!!! momentum=0.9
        # dw2_past = 0.9*dw2_past + np.mean(dw2,0) #!!! momentum=0.9
        # dwr_past = 0.9*dwr_past + np.mean(dwr,0) #!!! momentum=0.9
        # dbias_past = 0.9*dbias_past + np.mean(dbias,0) #!!! momentum=0.9 #AKSHAY

        # w1 += dw1_past # input-->hidden update
        # w2 += dw2_past # hidden-->hidden update
        # wr += dwr_past # hidden-->output update
        # bias += dbias_past # bias update #AKSHAY

        # w1 += np.mean(dw1,0) # these are for no-momentum update (slow learning)
        # w2 += np.mean(dw2,0)
        # wr += np.mean(dwr,0)
        # bias += np.mean(dbias,0)
        # np.fill_diagonal(wr,0) # no-self connectivity

        #!!! below 5 lines are valid only for E/I network (uncomment when using random network)
        ind_negative = wr<0
        ind = ind_negative*(~ind_fromi) # find negative & exciatory neurons
        wr[ind] = 0. # force the negative weights to 0
        ind = (~ind_negative)*ind_fromi # find positive & inhibitory neurons
        wr[ind] = 0. # force the positive weights to 0

        pi = np.exp(out_rec[:,cue_on,:])/np.exp(out_rec[:,cue_on,:]).sum(2).reshape(batch_size,dur_stim,1) # sfotmax /// [batch_size,dur_stim,2]
        pi_m = pi.mean(1)
        acc_batch[batch_idx] = (np.argmax(pi_m,axis=1) == y_batch[:,cue_on][:,-1]).mean() # save cc

        loss_batch[batch_idx] = loss.mean() # save loss
        fr_batch[batch_idx] = 1000*z_rec.mean() # save firing rate
    loss_train.append(loss_batch.mean()) # save loss
    acc_train.append(acc_batch.mean()) # save acc
    fr_train.append(fr_batch.mean()) # save firing rate
    print('EPOCH %d TRAIN) loss: %0.4f, acc: %0.4f, fr: %0.4f Hz' %(epoch, loss_train[epoch], acc_train[epoch], fr_train[epoch]))

    ######################### validate #########################
    x_batch,y1hot_batch,y_batch = x[-1],y1hot[-1],y[-1] # the last batch is for valid data
    loss, out_rec, dw1, dwr, dw2, dbias, v_rec, z_rec, a_rec = snn_models.lif_eprop(w1,wr,w2,bias,B,x_batch,y1hot_batch,cue_on,decays)


    pi = np.exp(out_rec[:,cue_on,:])/np.exp(out_rec[:,cue_on,:]).sum(2).reshape(batch_size,dur_stim,1) # softmax /// [batch_size,dur_stim,2]
    pi_m = pi.mean(1)
    acc_valid.append((np.argmax(pi_m,axis=1) == y_batch[:,cue_on][:,-1]).mean()) # save acc

    loss_valid.append(loss.mean()) # save loss
    fr_valid.append(1000*z_rec.mean()) # save firing rate
    print('EPOCH %d VALID) loss: %0.4f, acc: %0.4f, fr: %0.4f Hz' %(epoch, loss_valid[epoch], acc_valid[epoch], fr_valid[epoch]))

    spktr.append(z_rec[0]) # save spike train /// save one batch data only to save memory
    atr.append(a_rec[0]) # save a trace
    wrs.append(wr.copy()) # save weights

    ######################### save best weights #########################
    if loss_valid[-1] == min(loss_valid): # or you can use acc criteria instead
        w1_save,w2_save,wr_save,bias_save = w1.copy(),w2.copy(),wr.copy(),bias.copy()
        epoch_best = epoch
spktr = np.stack(spktr,0)
atr = np.stack(atr,0)
wrs = np.stack(wrs,0)
N = int(n_epochs/8) # epoch interval for plotting
