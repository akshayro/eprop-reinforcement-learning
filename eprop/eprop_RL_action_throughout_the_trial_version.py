# this is the eprop update rule for the RL case (action throughout the trial version (fig4 & 5) )
# See bottom of P.25 to P.27 of the supp info
import numpy as np
import numba as nb

#Part1%% parameters
nb_inputs, nb_hidden, nb_outputs  = 40, 100, 2 # number of neurons
time_step = 1 # 1ms time resolution

lr = 0.002 # learning rate 0.005           #  learning rate = eta
thr = 0.6 # v threshold                   
tau_out = 3 # output decay time constant /// unit: ms     
kappa = np.exp(-time_step/tau_out) # output decay
tau_mem = 20 # v decay time constant /// unit: ms                  
alpha = np.exp(-time_step/tau_mem) # v decay                       
beta = 0.2 # adaptation increment                                  
tau_a = 200 # adaptation decay time constant /// unit: ms           
rho = np.exp(-time_step/(tau_a+1e-12)) # adaptation decay           
t_ref = 2 # refractory period /// unit: ms

gamma = 0.99 # discount factor
c_V = 0.5    # trade off btn E_pi and E_V (see eqn 32 of eprop paper)
params = np.array([lr, thr, alpha, beta, kappa, rho, t_ref, gamma, c_V]) # save as a vector


#Part2%% initialize weight
weight_scale = 10*(1.0-alpha) #!!!
w1 = np.random.normal(size=(nb_inputs,nb_hidden), loc=0.0, scale=weight_scale/np.sqrt(nb_inputs)) # input-->hidden
w2 = np.random.normal(size=(nb_hidden,nb_outputs), loc=0.0, scale=weight_scale/np.sqrt(nb_hidden)) # hidden--> output
w3 = np.random.normal(size=(nb_hidden,nb_outputs), loc=0.0, scale=weight_scale/np.sqrt(nb_hidden)) # hidden--> output
bias1 = np.random.normal(size=(nb_outputs), loc=0.0, scale=weight_scale/np.sqrt(nb_outputs)) # output bias
bias2 = np.random.normal(size=(nb_outputs), loc=0.0, scale=weight_scale/np.sqrt(nb_outputs)) # output bias
B_V_j = np.random.normal(size=(nb_outputs), loc=0.0, scale=weight_scale/np.sqrt(nb_outputs)) #!!! random e-prop
B_pi_jk = np.random.normal(size=(nb_outputs,nb_hidden), loc=0.0, scale=weight_scale/np.sqrt(nb_outputs)) #!!! random e-prop


# t = 0
z = np.zeros((nb_hidden,)) # spike or not (1 or 0)
a = np.zeros((nb_hidden,))               # adaptation  
v = np.zeros((nb_hidden,))               # voltage
y = np.dot(z,w2)           # y = np.zeros((nb_outputs,)) # output
eps_ijv = np.zeros((nb_hidden,1)) # eligibility vector for v(has only an "i" index) /// = z_i
eps_ija = np.zeros((nb_hidden, nb_hidden)) # eligibility vector for a
epsin_ijv = np.zeros((nb_inputs,1)) # eligibility vector for input v
eps_jkv = np.zeros((nb_hidden,)) # eligibility vector for output v
pi = np.exp(y)/np.exp(y).sum() # softmax
del_y = pi - target_1hot  #!!! delta_y /// [nb_outputs] 
L_j =  - c_V*B_V_j + np.dot(del_y, B_pi_jk)    
eij = eps_ijv*phi_j - beta*eps_ija*phi_j  
e_bar_ij = eps_ijv*phi_j - beta*eps_ija*phi_j
e_ij_in = epsin_ijv*phi_j
F_kappa_z_j_t = z
del_y_F_kappa_z_j_t = del_y*F_kappa_z_j_t
    
#%% lif eprop
@nb.jit(nopython=True, parallel=True)
def lif_eprop(w1_old, wr_old, w2_old, w3_old, bias1_old, bias2_old, a_old, v_old, y_old, z_old, \
              eps_ijv_old, eps_ija_old, eps_jkv_old, epsin_ijv_old, L_j_old, eij_old, e_bar_ij_old, e_ij_in_old, del_y_old, \
              F_kappa_z_j_t_old, del_y_F_kappa_z_j_t_old,  \
              B, input_data, target_1hot, r_t, V_t, V_tplus1, params):
    # "old" means from the last time step (t-1) 

    nb_inputs = input_data.shape[0] 
    nb_hidden,nb_outputs = w2_old.shape
    lr,thr,alpha,beta,kappa,rho,t_ref,gamma,c_V = params[0], params[1], params[2], params[3], params[4], params[5], params[6], decays[7], params[8] # get params
     
    syn_from_input = np.dot(input_data, w1_old) # synaptic current from input
    
    z_bool = np.zeros((nb_hidden,),dtype=nb.boolean) # spike or not (True or False)
    z_counts = np.zeros((nb_hidden,),dtype=nb.int16) # spike count for refractory period implementation
    
    dw1 = np.zeros((nb_inputs, nb_hidden))   # input->hidden weight change
    dwr = np.zeros((nb_hidden, nb_hidden))   # hidden weight change
    dw2 = np.zeros((nb_hidden, nb_outputs))  # hidden->output weight change
    dbias = np.zeros((nb_outputs,)) # bias change
    loss = 0 # loss value    

    phi_j = np.zeros((nb_hidden,)) # pseudo derivative
    
    # adaptation update (t-1)    
    a = rho*a_old + z_old       # adaptation update   
    A = thr + beta*a    # threshold update     
    
    # spike update (t-1)    
    z_bool = v_old > A      # find spikes
    z_counts[z_bool] = z_counts[z_bool] + t_ref #!!! refractory period
    nrn_ready = np.where(z_counts==0)[0]  # neurons that ready to fire (no refractory period)  
    
    # pseudo derivate update (only for non-refractory period neurons) (t)
    phi_j[:] = 0.
    phi_j[nrn_ready] = (0.3/thr)*np.maximum(0, 1-np.abs((v[nrn_ready]-A[nrn_ready])/thr)) #  /// [nb_hidden]    
    phi_j = phi_j.reshape(1,-1)
    
    # voltage update (t)
    z = z_bool.astype(nb.float64) # change data type
    syn = syn_from_input[t] + np.dot(z,wr_old) # total synaptic currents /// [nb_hidden]
    v_old[z_bool] = A[z_bool] #!!! voltage is not larger than threshold
    v = alpha*v_old - A*z + syn # voltage update
 
    # output update (t)
    y = kappa*y_old + np.dot(z,w2_old) #!!! + bias # 

    # eligibility trace for eij (t)
    eps_ijv = alpha*eps_ijv_old + z.reshape(-1,1) # notice that, "eps_ijv = eps_iv"
    eps_ija = eps_ijv* phi_j + rho*eps_ija_old #!!! faster than using "outer" function
    eij = eps_ijv*phi_j - beta*eps_ija*phi_j #!!! faster than using "outer" function
            
    # eligibility trace for eij for input->output (t)
    epsin_ijv = alpha*epsin_ijv + input_data.reshape(-1,1)
    eij_in = epsin_ijv*phi_j
            
    # eiligibility trace for eij for output (t)
    eps_jkv = kappa*eps_jkv + z 
            
    # Temporal Difference (TD) error
    V_tplus1 = 0 #????
    delta_t = r_t + gamma*V_tplus1 - V_t

    pi = np.exp(y)/np.exp(y).sum() # softmax
    del_y = pi - target_1hot  #!!! delta_y /// [nb_outputs] 

    # this assume t > 0
    e_bar_ij = kappa*e_ij_old + eij
#   lsig = np.dot(del_y, B) # learning signal /// [nb_outputs],[nb_outputs,nb_hidden]-->[nb_hidden]    
    L_j =  - c_V*B_V_j + np.dot(del_y, B_pi_jk)    # eqn (37) in eprop paper
    F_gamma_Lj_e_bar_ij = gamma*L_j_old.reshape(1,nb_hidden)*(e_bar_ij_old) + L_j.reshape(1,nb_hidden)*e_bar_ij                

    # (1) update recurrents
#   dwr[b] += -lr*lsig.reshape(1,nb_hidden)*eij # [1,nb_hidden],[nb_hidden,nb_hidden]-->[nb_hidden,nb_hidden]
    dwr = -lr*delta_t*F_gamma_Lj_e_bar_ij    # [1,nb_hidden],[nb_hidden,nb_hidden]-->[nb_hidden,nb_hidden]  
    
    # (2) update inputs-->recurrent
    # this assume t > 0
    e_bar_ij_in = kappa*e_ij_in_old + eij_in
 
    F_gamma_Lj_e_bar_ij_in = gamma*L_j_old.reshape(1,nb_hidden)*(e_bar_ij_in_old) + L_j.reshape(1,nb_hidden)*e_bar_ij_in         
    dw1 = -lr*delta_t*F_gamma_Lj_e_bar_ij_in
#    dw1 = -lr*lsig.reshape(1,nb_hidden)*eij_in # [1,nb_hidden],[nb_inputs,nb_hidden]-->[nb_inputs,nb_hidden]
                
    # (3) update of recurrent --> output neurons  
    # (3.1) W_kj^(pi,out) eqn (47) of supp info
    del_y_F_kappa_z_j_t = del_y*F_kappa_z_j_t
    F_gamma_del_y_F_kappa_z_j_t = gamma*del_y_F_kappa_z_j_t_old + del_y_F_kappa_z_j_t
    dw2 = -lr*delta_t*F_gamma_del_y_F_kappa_z_j_t
#    dw2 = -lr*eps_jkv.reshape(-1,1)*del_y.reshape(1,-1)
    
    # (3.2) W_j^(V,out)  eqn (48) of supp info
    F_kappa_z_j_t = kappa*z_old + z
    F_gamma_F_kappa_z_j_t = gamma*F_kappa_z_j_t_old + F_kappa_z_j_t
    dw3 = lr*c_V*delta_t*F_gamma_F_kappa_z_j_t
    
    # (4) bias update
       # dbias += -lr*del_y
    F_gamma_pi_k_I_k = gamma*del_y_old + del_y  
    dbias1 = -lr*delta_t*F_gamma_pi_k_I_k     # db_k_pi_out
    dbias2 = lr*c_V*delta_t                   # db_V_out
    
    loss = -np.sum(target_1hot*np.log(pi+1e-10))
    
    return loss, y, dw1, dwr, dw2, dw3, dbias1, dbias2, a, v, y, z, \
            eps_ijv, eps_ija, eps_jkv, epsin_ijv, L_j, eij, e_bar_ij, e_ij_in, del_y, \
            F_kappa_z_j_t, del_y_F_kappa_z_j_t