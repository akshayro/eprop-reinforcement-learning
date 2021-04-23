import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import spikes4 as spk4
import snn_models

import scipy.io as sio #allows for importing of .mat files

from elephant.spike_train_generation import homogeneous_poisson_process
from quantities import Hz, s, ms




rootDir = ''
fn = 'contdata95.mat'

# conv_size = 3 # size of time history

# load the mat file
mat = sio.loadmat(rootDir+fn)

# Get each variable from the mat file
# (Flipping X and Y so that X = neural data, Y = kinematics)
x = mat['Y']
y = mat['X'][:,0:4]

x = x.astype(float)
y = y.astype(float)



# this takes about 1 hr to run (there is better a way to do it, but will do it later)

nsamp = x.shape[0]        # number of sample
N_neuron = x.shape[1]     # number of neuron
#N_neuron = 1
spike_time = [[] for x in range(N_neuron)]
x_Hz = x*10

for n in range(N_neuron):
    print(n)
    tmp = []
    for t in range(nsamp):
        rate_ = x_Hz[t,n]
        if rate_ > 0:
            Spktrain = homogeneous_poisson_process(rate=rate_*Hz, t_start=0.0*ms, t_stop=100.0*ms)
            tmp = np.concatenate((tmp, np.array(Spktrain) + t*100.))        
    spike_time[n] = np.rint(tmp).astype(int)
    
    

    
    
spike_data = np.zeros((nsamp*100,N_neuron))
for n in range(N_neuron):
    sp_t = spike_time[n]
    spike_data[sp_t,n] = 1
    
    
    
# saving the output, spike_data95.npy has already been generated, so no need to do it again,
# becareful not to overwrite the existing one

#with open('spike_data95.npy', 'wb') as f:
#    np.save(g, spike_data)



# interpolation of y 

from scipy.interpolate import interp1d

x = np.linspace(1,y.shape[0],num=y.shape[0])      # [ms]
x = x - 1  

f1 = interp1d(x, y[:,0], kind='cubic')
f2 = interp1d(x, y[:,1], kind='cubic')
f3 = interp1d(x, y[:,2], kind='cubic')
f4 = interp1d(x, y[:,3], kind='cubic')

xnew = np.zeros(y.shape[0]*100)
for i in range(xnew.shape[0]-100):
    xnew[i] = 0.01*i

y1 = f1(xnew)
y2 = f2(xnew)
y3 = f3(xnew)
y4 = f4(xnew)

interpolated_y = np.column_stack((y1,y2,y3,y4))
print(interpolated_y.shape)



with open('interpolated_y95.npy', 'wb') as h:
    np.save(h, interpolated_y)