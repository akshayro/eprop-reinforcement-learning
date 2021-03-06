import numpy as np
import numba as nb

#%% lif eprop
@nb.jit(nopython=True, parallel=True)
def lif_eprop(w1,wr,w2,bias,B,input_data,target_1hot,cue_on,decays):
    # classification version
    batch_size,nb_steps,nb_inputs = input_data.shape
    nb_hidden,nb_outputs = w2.shape
    lr,thr,alpha,beta,kappa,rho,t_ref = decays[0], decays[1], decays[2], decays[3], decays[4], decays[5], decays[6] # get params
    out_rec = np.zeros((batch_size, nb_steps, nb_outputs)) # output record
    v_rec = np.zeros((batch_size, nb_steps, nb_hidden)) # hidden v record
    z_rec = np.zeros((batch_size, nb_steps, nb_hidden)) # hidden z record
    a_rec = np.zeros((batch_size, nb_steps, nb_hidden)) # hidden a record
    dw1 = np.zeros((batch_size, nb_inputs, nb_hidden)) # input->hidden weight change
    dwr = np.zeros((batch_size, nb_hidden, nb_hidden)) # hidden weight change
    dw2 = np.zeros((batch_size, nb_hidden, nb_outputs)) # hidden->output weight change
    dbias = np.zeros((batch_size, nb_outputs)) # bias change
    loss = np.zeros((batch_size)) # loss value
    for b in nb.prange(int(batch_size)): # prarallel processing, change "nb.prange" to "range" when not using parallel
        syn_from_input = np.dot(input_data[b], w1) # synaptic current from input
        z = np.zeros((nb_hidden,)) # spike or not (1 or 0)
        z_bool = np.zeros((nb_hidden,),dtype=nb.boolean) # spike or not (True or False)
        z_counts = np.zeros((nb_hidden,),dtype=nb.int16) # spike count for refractory period implementation
        v = np.zeros((nb_hidden,)) # voltage
        y = np.zeros((nb_outputs,)) # output
        a = np.zeros((nb_hidden,)) # adaptation
        eps_ijv = np.zeros((nb_hidden,1)) # eligibility vector for v(has only an "i" index) /// = z_i
        eps_ija = np.zeros((nb_hidden, nb_hidden)) # eligibility vector for a
        epsin_ijv = np.zeros((nb_inputs,1)) # eligibility vector for input v
        eps_jkv = np.zeros((nb_hidden,)) # eligibility vector for output v
        phi_j = np.zeros((nb_hidden,)) # pseudo derivative
        for t in range(nb_steps): # t=0
            # adaptation update (t-1)
            a = rho*a + z # adaptation update
            A = thr + beta*a # threshold update
            
            # spike update (t-1)
            z_bool = v > A # find spikes
            z_counts[z_bool] = z_counts[z_bool] + t_ref #!!! refractory period
            nrn_ready = np.where(z_counts==0)[0] # neurons that ready to fire (no refractory period)
            
            # pseudo derivate update (only for non-refractory period neurons) (t)
            phi_j[:] = 0.
            phi_j[nrn_ready] = (0.3/thr)*np.maximum(0, 1-np.abs((v[nrn_ready]-A[nrn_ready])/thr)) #  /// [nb_hidden]
            
            # voltage update (t)
            z = z_bool.astype(nb.float64) # change data type
            syn = syn_from_input[t] + np.dot(z,wr) # total synaptic currents /// [nb_hidden]
            v[z_bool] = A[z_bool] #!!! voltage is not larger than threshold
            v_new = alpha*v - A*z + syn # voltage update
            v = v_new 
            
            # output update (t)
            y_new = kappa*y + np.dot(z,w2) + bias  #!!! + bias
            y = y_new 
            
            # eligibility trace for eij (t)
            eps_ijv = alpha*eps_ijv + z.reshape(-1,1) # notice that, "eps_ijv = eps_iv"
            eps_ija = eps_ijv*phi_j.reshape(1,-1) + rho*eps_ija #!!! faster than using "outer" function
            eij = eps_ijv*phi_j.reshape(1,-1) - beta*eps_ija*phi_j.reshape(1,-1) #!!! faster than using "outer" function
            
            # eligibility trace for eij for input->output (t)
            epsin_ijv = alpha*epsin_ijv + input_data[b,t].reshape(-1,1)
            eij_in = epsin_ijv*phi_j.reshape(1,-1)
            
            # eiligibility trace for eij for output (t)
            eps_jkv = kappa*eps_jkv + z
            
            # when learning cue is on, and weight update
            if cue_on[t]!=0:
                pi = np.exp(y)/np.exp(y).sum() # softmax
                del_y = pi - target_1hot[b,t] #!!! delta_y /// [nb_outputs]
                lsig = np.dot(del_y, B) # learning signal /// [nb_outputs],[nb_outputs,nb_hidden]-->[nb_hidden]
                
                # (1) update recurrents
                # eq 27 of Methods
                dwr[b] += -lr*lsig.reshape(1,nb_hidden)*eij # [1,nb_hidden],[nb_hidden,nb_hidden]-->[nb_hidden,nb_hidden]
                
                # (2) update inputs-->recurrent
                dw1[b] += -lr*lsig.reshape(1,nb_hidden)*eij_in # [1,nb_hidden],[nb_inputs,nb_hidden]-->[nb_inputs,nb_hidden]
                
                # (3) update of recurrent-->output neurons (below eq.20 in supplementary3)
                dw2[b] += -lr*eps_jkv.reshape(-1,1)*del_y.reshape(1,-1)
                
                # (4) bias update
                dbias[b] += -lr*del_y 
                
                loss[b] += -np.sum(target_1hot[b,t]*np.log(pi+1e-10)) # cross entropy
            z_counts[z_counts>=1] = z_counts[z_counts>=1]-1 # spike count decay
            v_rec[b,t] = v # save v
            z_rec[b,t] = z # save z
            out_rec[b,t] = y # save y
            a_rec[b,t] = a # save a
    return loss, out_rec, dw1, dwr, dw2, dbias, v_rec, z_rec, a_rec


@nb.jit(nopython=True, parallel=True)
def lif_eprop2(w1,wr,w2,bias,B,input_data,target_y,decays):
    # regression version (see supp info P.17)
    nb_batch,nb_steps,nb_inputs = input_data.shape
    nb_hidden,nb_outputs = w2.shape
    lr,thr,alpha,beta,kappa,rho,t_ref = decays[0], decays[1], decays[2], decays[3], decays[4], decays[5], decays[6] # get params
    out_rec = np.zeros((nb_batch, nb_steps, nb_outputs)) # output record
    v_rec = np.zeros((nb_batch, nb_steps, nb_hidden)) # hidden v record
    z_rec = np.zeros((nb_batch, nb_steps, nb_hidden)) # hidden z record
    a_rec = np.zeros((nb_batch, nb_steps, nb_hidden)) # hidden a record
    dw1 = np.zeros((nb_batch, nb_inputs, nb_hidden)) # input->hidden weight change
    dwr = np.zeros((nb_batch, nb_hidden, nb_hidden)) # hidden weight change
    dw2 = np.zeros((nb_batch, nb_hidden, nb_outputs)) # hidden->output weight change
    dbias = np.zeros((nb_batch, nb_outputs)) # bias change
    loss = np.zeros(nb_batch) # loss value
    print("nb_batch=", nb_batch)
    for b in nb.prange(int(nb_batch)): # prarallel processing, change "nb.prange" to "range" when not using parallel
        syn_from_input = np.dot(input_data[b], w1) # synaptic current from input
        z = np.zeros((nb_hidden,)) # spike or not (1 or 0)
        z_bool = np.zeros((nb_hidden,),dtype=nb.boolean) # spike or not (True or False)
        z_counts = np.zeros((nb_hidden,),dtype=nb.int16) # spike count for refractory period implementation
        v = np.zeros((nb_hidden,)) # voltage
        y = np.zeros((nb_outputs,)) # output
        a = np.zeros((nb_hidden,)) # adaptation
        eps_ijv = np.zeros((nb_hidden,1)) # eligibility vector for v(has only an "i" index) /// = z_i
        eps_ija = np.zeros((nb_hidden, nb_hidden)) # eligibility vector for a
        epsin_ijv = np.zeros((nb_inputs,1)) # eligibility vector for input v
        eps_jkv = np.zeros((nb_hidden,)) # eligibility vector for output v
        phi_j = np.zeros((nb_hidden,)) # pseudo derivative
        for t in range(nb_steps): # t=0
            # adaptation update (t-1)
            a = rho*a + z # adaptation update
            A = thr + beta*a # threshold update
            
            # spike update (t-1)
            z_bool = v > A # find spikes
            z_counts[z_bool] = z_counts[z_bool] + t_ref #!!! refractory period
            nrn_ready = np.where(z_counts==0)[0] # neurons that ready to fire (no refractory period)
            
            # pseudo derivate update (only for non-refractory period neurons) (t)
            phi_j[:] = 0.
            phi_j[nrn_ready] = (0.3/thr)*np.maximum(0, 1-np.abs((v[nrn_ready]-A[nrn_ready])/thr)) #  /// [nb_hidden]
            
            # voltage update (t)
            z = z_bool.astype(nb.float64) # change data type
            syn = syn_from_input[t] + np.dot(z,wr) # total synaptic currents /// [nb_hidden]
            v[z_bool] = A[z_bool] #!!! voltage is not larger than threshold
            v_new = alpha*v - A*z + syn # voltage update
            v = v_new 
            
            # output update (t)
            y_new = kappa*y + np.dot(z,w2) + bias #!!! + bias
            y = y_new 
            
            # eligibility trace for eij (t)
            eps_ijv = alpha*eps_ijv + z.reshape(-1,1) # notice that, "eps_ijv = eps_iv"
            eps_ija = eps_ijv*phi_j.reshape(1,-1) + rho*eps_ija #!!! faster than using "outer" function
            eij = eps_ijv*phi_j.reshape(1,-1) - beta*eps_ija*phi_j.reshape(1,-1) #!!! faster than using "outer" function
            
            # eligibility trace for eij for input->output (t)
            epsin_ijv = alpha*epsin_ijv + input_data[b,t].reshape(-1,1)
            eij_in = epsin_ijv*phi_j.reshape(1,-1)
            
            # eiligibility trace for eij for output (t)
            eps_jkv = kappa*eps_jkv + z

            del_y = y - target_y[b,t]   #!!! delta_y /// [nb_outputs]    
#            loss[b] += 0.5*np.sum((del_y)**2)   # SE
            loss[b] += np.mean((del_y)**2)   # MSE           
#            print('loss=',loss[b])
            lsig = np.dot(del_y, B) # learning signal /// [nb_outputs],[nb_outputs,nb_hidden]-->[nb_hidden]
                
            # (1) update recurrents
            dwr[b] += -lr*lsig.reshape(1,nb_hidden)*eij # [1,nb_hidden],[nb_hidden,nb_hidden]-->[nb_hidden,nb_hidden]
                
            # (2) update inputs-->recurrent
            dw1[b] += -lr*lsig.reshape(1,nb_hidden)*eij_in # [1,nb_hidden],[nb_inputs,nb_hidden]-->[nb_inputs,nb_hidden]
                
            # (3) update of recurrent-->output neurons (below eq.20 in supplementary3)
            dw2[b] += -lr*eps_jkv.reshape(-1,1)*del_y.reshape(1,-1)
                
            # (4) bias update
            dbias[b] += -lr*del_y 
                
            z_counts[z_counts>=1] = z_counts[z_counts>=1]-1 # spike count decay
            v_rec[b,t] = v # save v
            z_rec[b,t] = z # save z
            out_rec[b,t] = y # save y
            a_rec[b,t] = a # save a
    return loss, out_rec, dw1, dwr, dw2, dbias, v_rec, z_rec, a_rec

@nb.jit(nopython=True, parallel=True)
def lif_eprop3(w1,wr,w2,bias,B,input_data,target,cue_on,decays):
    # classification version
    batch_size,nb_steps,nb_inputs = input_data.shape
    nb_hidden,nb_outputs = w2.shape
    lr,thr,alpha,beta,kappa,rho,t_ref = decays[0], decays[1], decays[2], decays[3], decays[4], decays[5], decays[6] # get params
    out_rec = np.zeros((batch_size, nb_steps, nb_outputs)) # output record
    v_rec = np.zeros((batch_size, nb_steps, nb_hidden)) # hidden v record
    z_rec = np.zeros((batch_size, nb_steps, nb_hidden)) # hidden z record
    a_rec = np.zeros((batch_size, nb_steps, nb_hidden)) # hidden a record
    dw1 = np.zeros((batch_size, nb_inputs, nb_hidden)) # input->hidden weight change
    dwr = np.zeros((batch_size, nb_hidden, nb_hidden)) # hidden weight change
    dw2 = np.zeros((batch_size, nb_hidden, nb_outputs)) # hidden->output weight change
    dbias = np.zeros((batch_size, nb_outputs)) # bias change
    loss = np.zeros((batch_size)) # loss value
    for b in nb.prange(int(batch_size)): # prarallel processing, change "nb.prange" to "range" when not using parallel
        syn_from_input = np.dot(input_data[b], w1) # synaptic current from input
        z = np.zeros((nb_hidden,)) # spike or not (1 or 0)
        z_bool = np.zeros((nb_hidden,),dtype=nb.boolean) # spike or not (True or False)
        z_counts = np.zeros((nb_hidden,),dtype=nb.int16) # spike count for refractory period implementation
        v = np.zeros((nb_hidden,)) # voltage
        y = np.zeros((nb_outputs,)) # output
        a = np.zeros((nb_hidden,)) # adaptation
        eps_ijv = np.zeros((nb_hidden,1)) # eligibility vector for v(has only an "i" index) /// = z_i
        eps_ija = np.zeros((nb_hidden, nb_hidden)) # eligibility vector for a
        epsin_ijv = np.zeros((nb_inputs,1)) # eligibility vector for input v
        eps_jkv = np.zeros((nb_hidden,)) # eligibility vector for output v
        phi_j = np.zeros((nb_hidden,)) # pseudo derivative
        for t in range(nb_steps): # t=0
            # adaptation update (t-1)
            a = rho*a + z # adaptation update
            A = thr + beta*a # threshold update
            
            # spike update (t-1)
            z_bool = v > A # find spikes
            z_counts[z_bool] = z_counts[z_bool] + t_ref #!!! refractory period
            nrn_ready = np.where(z_counts==0)[0] # neurons that ready to fire (no refractory period)
            
            # pseudo derivate update (only for non-refractory period neurons) (t)
            phi_j[:] = 0.
            phi_j[nrn_ready] = (0.3/thr)*np.maximum(0, 1-np.abs((v[nrn_ready]-A[nrn_ready])/thr)) #  /// [nb_hidden]
            
            # voltage update (t)
            z = z_bool.astype(nb.float64) # change data type
            syn = syn_from_input[t] + np.dot(z,wr) # total synaptic currents /// [nb_hidden]
            v[z_bool] = A[z_bool] #!!! voltage is not larger than threshold
            v_new = alpha*v - A*z + syn # voltage update
            v = v_new 
            
            # output update (t)
            y_new = kappa*y + np.dot(z,w2)  #!!! + bias
            y = y_new 
            
            # eligibility trace for eij (t)
            eps_ijv = alpha*eps_ijv + z.reshape(-1,1) # notice that, "eps_ijv = eps_iv"
            eps_ija = eps_ijv*phi_j.reshape(1,-1) + rho*eps_ija #!!! faster than using "outer" function
            eij = eps_ijv*phi_j.reshape(1,-1) - beta*eps_ija*phi_j.reshape(1,-1) #!!! faster than using "outer" function
            
            # eligibility trace for eij for input->output (t)
            epsin_ijv = alpha*epsin_ijv + input_data[b,t].reshape(-1,1)
            eij_in = epsin_ijv*phi_j.reshape(1,-1)
            
            # eiligibility trace for eij for output (t)
            eps_jkv = kappa*eps_jkv + z
            
            # when learning cue is on, and weight update
            if cue_on[t]!=0:
                del_y = y - target[b,t] #!!! delta_y /// [nb_outputs]
                lsig = np.dot(del_y, B) # learning signal /// [nb_outputs],[nb_outputs,nb_hidden]-->[nb_hidden]
                
                # (1) update recurrents
                # eq 27 of Methods
                dwr[b] += -lr*lsig.reshape(1,nb_hidden)*eij # [1,nb_hidden],[nb_hidden,nb_hidden]-->[nb_hidden,nb_hidden]
                
                # (2) update inputs-->recurrent
                dw1[b] += -lr*lsig.reshape(1,nb_hidden)*eij_in # [1,nb_hidden],[nb_inputs,nb_hidden]-->[nb_inputs,nb_hidden]
                
                # (3) update of recurrent-->output neurons (below eq.20 in supplementary3)
                dw2[b] += -lr*eps_jkv.reshape(-1,1)*del_y.reshape(1,-1)
                
                # (4) bias update
                dbias[b] += -lr*del_y 
                
                # loss[b] += -np.sum(target[b,t]*np.log(pi+1e-10)) # cross entropy
                loss[b] += np.mean((del_y)**2)   # MSE 
            z_counts[z_counts>=1] = z_counts[z_counts>=1]-1 # spike count decay
            v_rec[b,t] = v # save v
            z_rec[b,t] = z # save z
            out_rec[b,t] = y # save y
            a_rec[b,t] = a # save a
    return loss, out_rec, dw1, dwr, dw2, dbias, v_rec, z_rec, a_rec




@nb.jit(nopython=True, parallel=True)
def lif_eprop4(w1,wr,w2,bias,B,input_data,target_y,decays):
    # regression version (see supp info P.17)
#    nb_batch,nb_steps,nb_inputs = input_data.shape
    nb_batch = 1
    nb_steps = input_data.shape[0]
    nb_inputs = input_data.shape[1]
    nb_hidden,nb_outputs = w2.shape
    lr,thr,alpha,beta,kappa,rho,t_ref = decays[0], decays[1], decays[2], decays[3], decays[4], decays[5], decays[6] # get params
    out_rec = np.zeros((nb_batch, nb_steps, nb_outputs)) # output record
    v_rec = np.zeros((nb_batch, nb_steps, nb_hidden)) # hidden v record
    z_rec = np.zeros((nb_batch, nb_steps, nb_hidden)) # hidden z record
    a_rec = np.zeros((nb_batch, nb_steps, nb_hidden)) # hidden a record
    dw1 = np.zeros((nb_batch, nb_inputs, nb_hidden)) # input->hidden weight change
    dwr = np.zeros((nb_batch, nb_hidden, nb_hidden)) # hidden weight change
    dw2 = np.zeros((nb_batch, nb_hidden, nb_outputs)) # hidden->output weight change
    dbias = np.zeros((nb_batch, nb_outputs)) # bias change
    loss = np.zeros(nb_batch) # loss value

    for b in nb.prange(int(nb_batch)): # prarallel processing, change "nb.prange" to "range" when not using parallel
        syn_from_input = np.dot(input_data[b], w1) # synaptic current from input
        z = np.zeros((nb_hidden,)) # spike or not (1 or 0)
        z_bool = np.zeros((nb_hidden,),dtype=nb.boolean) # spike or not (True or False)
        z_counts = np.zeros((nb_hidden,),dtype=nb.int16) # spike count for refractory period implementation
        v = np.zeros((nb_hidden,)) # voltage
        y = np.zeros((nb_outputs,)) # output
        a = np.zeros((nb_hidden,)) # adaptation
        eps_ijv = np.zeros((nb_hidden,1)) # eligibility vector for v(has only an "i" index) /// = z_i
        eps_ija = np.zeros((nb_hidden, nb_hidden)) # eligibility vector for a
        epsin_ijv = np.zeros((nb_inputs,1)) # eligibility vector for input v
        eps_jkv = np.zeros((nb_hidden,)) # eligibility vector for output v
        phi_j = np.zeros((nb_hidden,)) # pseudo derivative
        for t in range(nb_steps): # t=0
            # adaptation update (t-1)
            a = rho*a + z # adaptation update
            A = thr + beta*a # threshold update
            
            # spike update (t-1)
            z_bool = v > A # find spikes
            z_counts[z_bool] = z_counts[z_bool] + t_ref #!!! refractory period
            nrn_ready = np.where(z_counts==0)[0] # neurons that ready to fire (no refractory period)
            
            # pseudo derivate update (only for non-refractory period neurons) (t)
            phi_j[:] = 0.
            phi_j[nrn_ready] = (0.3/thr)*np.maximum(0, 1-np.abs((v[nrn_ready]-A[nrn_ready])/thr)) #  /// [nb_hidden]
            
            # voltage update (t)
            z = z_bool.astype(nb.float64) # change data type
            syn = syn_from_input[t] + np.dot(z,wr) # total synaptic currents /// [nb_hidden]
            v[z_bool] = A[z_bool] #!!! voltage is not larger than threshold
            v_new = alpha*v - A*z + syn # voltage update
            v = v_new 
            
            # output update (t)
            y_new = kappa*y + np.dot(z,w2) + bias #!!! + bias
            y = y_new 
            
            # eligibility trace for eij (t)
            eps_ijv = alpha*eps_ijv + z.reshape(-1,1) # notice that, "eps_ijv = eps_iv"
            eps_ija = eps_ijv*phi_j.reshape(1,-1) + rho*eps_ija #!!! faster than using "outer" function
            eij = eps_ijv*phi_j.reshape(1,-1) - beta*eps_ija*phi_j.reshape(1,-1) #!!! faster than using "outer" function
            
            # eligibility trace for eij for input->output (t)
#            epsin_ijv = alpha*epsin_ijv + input_data[b,t].reshape(-1,1)
            epsin_ijv = alpha*epsin_ijv + input_data[t].reshape(-1,1)
            eij_in = epsin_ijv*phi_j.reshape(1,-1)
            
            # eiligibility trace for eij for output (t)
            eps_jkv = kappa*eps_jkv + z
            
#            del_y = y - target_y[b,t]   #!!! delta_y /// [nb_outputs]    
            del_y = y - target_y[t]   #!!! delta_y /// [nb_outputs]    

#            loss[b] += 0.5*np.sum((del_y)**2)   # SE
            loss[b] += np.mean((del_y)**2)   # MSE           
#            print('loss=',loss[b])
            lsig = np.dot(del_y, B) # learning signal /// [nb_outputs],[nb_outputs,nb_hidden]-->[nb_hidden]
                
            # (1) update recurrents
            dwr[b] += -lr*lsig.reshape(1,nb_hidden)*eij # [1,nb_hidden],[nb_hidden,nb_hidden]-->[nb_hidden,nb_hidden]
                
            # (2) update inputs-->recurrent
            dw1[b] += -lr*lsig.reshape(1,nb_hidden)*eij_in # [1,nb_hidden],[nb_inputs,nb_hidden]-->[nb_inputs,nb_hidden]
                
            # (3) update of recurrent-->output neurons (below eq.20 in supplementary3)
            dw2[b] += -lr*eps_jkv.reshape(-1,1)*del_y.reshape(1,-1)
                
            # (4) bias update
            dbias[b] += -lr*del_y 
                
            z_counts[z_counts>=1] = z_counts[z_counts>=1]-1 # spike count decay
            v_rec[b,t] = v # save v
            z_rec[b,t] = z # save z
            out_rec[b,t] = y # save y
            a_rec[b,t] = a # save a
    return loss, out_rec, dw1, dwr, dw2, dbias, v_rec, z_rec, a_rec




def lif_eprop5(w1,wr,w2,bias,B,input_data,target_y,decays):
    #reduce to simplest case
    #input_data size = (timestep, nb_neuron)
    #target_y size = (timestep, 1)
    # regression version (see supp info P.17)
    # added firing rate regularization
    
#    nb_batch,nb_steps,nb_inputs = input_data.shape
    nb_batch = 1
    nb_steps = input_data.shape[0]
    nb_inputs = input_data.shape[1]
    nb_hidden,nb_outputs = w2.shape
    lr,thr,alpha,beta,kappa,rho,t_ref,c_reg,f_target = decays[0], decays[1], decays[2], decays[3], decays[4], decays[5], decays[6],decays[7],decays[8] # get params
    out_rec = np.zeros((nb_batch, nb_steps, nb_outputs)) # output record
    v_rec = np.zeros((nb_batch, nb_steps, nb_hidden)) # hidden v record
    z_rec = np.zeros((nb_batch, nb_steps, nb_hidden)) # hidden z record
    a_rec = np.zeros((nb_batch, nb_steps, nb_hidden)) # hidden a record
    dw1 = np.zeros((nb_batch, nb_inputs, nb_hidden)) # input->hidden weight change
    dwr = np.zeros((nb_batch, nb_hidden, nb_hidden)) # hidden weight change
    dw2 = np.zeros((nb_batch, nb_hidden, nb_outputs)) # hidden->output weight change
    dbias = np.zeros((nb_batch, nb_outputs)) # bias change
    loss = np.zeros(nb_batch) # loss value
    sum_z = 0
    f_ave = 0
    
#    for b in nb.prange(int(nb_batch)): # prarallel processing, change "nb.prange" to "range" when not using parallel
    for b in range(int(nb_batch)): # prarallel processing, change "nb.prange" to "range" when not using parallel
        syn_from_input = np.dot(input_data, w1) # synaptic current from input
        z = np.zeros((nb_hidden,)) # spike or not (1 or 0)
#        z_bool = np.zeros((nb_hidden,),dtype=np.boolean) # spike or not (True or False)
        z_bool = np.zeros((nb_hidden,),dtype=bool) # spike or not (True or False)        
#        z_counts = np.zeros((nb_hidden,),dtype=np.int16) # spike count for refractory period implementation
        z_counts = np.zeros((nb_hidden,),dtype=int) # spike count for refractory period implementation
        v = np.zeros((nb_hidden,)) # voltage
        y = np.zeros((nb_outputs,)) # output
        a = np.zeros((nb_hidden,)) # adaptation
        eps_ijv = np.zeros((nb_hidden,1)) # eligibility vector for v(has only an "i" index) /// = z_i
        eps_ija = np.zeros((nb_hidden, nb_hidden)) # eligibility vector for a
        epsin_ijv = np.zeros((nb_inputs,1)) # eligibility vector for input v
        eps_jkv = np.zeros((nb_hidden,)) # eligibility vector for output v
        phi_j = np.zeros((nb_hidden,)) # pseudo derivative
        
        
        for t in range(nb_steps): # t=0
            # adaptation update (t-1)
            a = rho*a + z # adaptation update
            A = thr + beta*a # threshold update
            
            # spike update (t-1)
            z_bool = v > A # find spikes
            z_counts[z_bool] = z_counts[z_bool] + t_ref #!!! refractory period
            nrn_ready = np.where(z_counts==0)[0] # neurons that ready to fire (no refractory period)
            
            # pseudo derivate update (only for non-refractory period neurons) (t)
            phi_j[:] = 0.
            phi_j[nrn_ready] = (0.3/thr)*np.maximum(0, 1-np.abs((v[nrn_ready]-A[nrn_ready])/thr)) #  /// [nb_hidden]
            
            # voltage update (t)
#            z = z_bool.astype(nb.float64) # change data type
            z = z_bool.astype(float) # change data type            
            syn = syn_from_input[t] + np.dot(z,wr) # total synaptic currents /// [nb_hidden]
            v[z_bool] = A[z_bool] #!!! voltage is not larger than threshold
            v_new = alpha*v - A*z + syn # voltage update
            v = v_new 
            
            # output update (t)
            y_new = kappa*y + np.dot(z,w2) + bias #!!! + bias
            y = y_new 
            
            # eligibility trace for eij (t)
            eps_ijv = alpha*eps_ijv + z.reshape(-1,1) # notice that, "eps_ijv = eps_iv"
            eps_ija = eps_ijv*phi_j.reshape(1,-1) + rho*eps_ija #!!! faster than using "outer" function
            eij = eps_ijv*phi_j.reshape(1,-1) - beta*eps_ija*phi_j.reshape(1,-1) #!!! faster than using "outer" function
            
            # eligibility trace for eij for input->output (t)
#            epsin_ijv = alpha*epsin_ijv + input_data[b,t].reshape(-1,1)
            epsin_ijv = alpha*epsin_ijv + input_data[t].reshape(-1,1)
            eij_in = epsin_ijv*phi_j.reshape(1,-1)
            
            # eiligibility trace for eij for output (t)
            eps_jkv = kappa*eps_jkv + z
            
#            del_y = y - target_y[b,t]   #!!! delta_y /// [nb_outputs]    
            del_y = y - target_y[t]   #!!! delta_y /// [nb_outputs]    

            loss[b] += 0.5*np.sum((del_y)**2)   # SE
#            loss[b] += np.mean((del_y)**2)   # MSE
    
#            print('loss=',loss[b])
            lsig = np.dot(del_y, B) # learning signal /// [nb_outputs],[nb_outputs,nb_hidden]-->[nb_hidden]
                
            # (1) update recurrents
            dwr[b] += -lr*lsig.reshape(1,nb_hidden)*eij # [1,nb_hidden],[nb_hidden,nb_hidden]-->[nb_hidden,nb_hidden]
            
            # firing rate regularization
            if t>0:
                sum_z += z
                f_ave = sum_z/(t)
                dwr[b] += lr* c_reg *(f_target - f_ave)*eij /t              
            
            # (2) update inputs-->recurrent
            dw1[b] += -lr*lsig.reshape(1,nb_hidden)*eij_in # [1,nb_hidden],[nb_inputs,nb_hidden]-->[nb_inputs,nb_hidden]
                
            # (3) update of recurrent-->output neurons (below eq.20 in supplementary3)
            dw2[b] += -lr*eps_jkv.reshape(-1,1)*del_y.reshape(1,-1)
                
            # (4) bias update
            dbias[b] += -lr*del_y 
                
            z_counts[z_counts>=1] = z_counts[z_counts>=1]-1 # spike count decay
            v_rec[b,t] = v # save v
            z_rec[b,t] = z # save z
            out_rec[b,t] = y # save y
            a_rec[b,t] = a # save a
    return loss, out_rec, dw1, dwr, dw2, dbias, v_rec, z_rec, a_rec






def lif_eprop6(w1,wr,w2,bias,B,input_data,target_y,decays):
    #reduce to simplest case
    #input_data size = (timestep, nb_neuron)
    #target_y size = (timestep, 1)
    # regression version (see supp info P.17)
    # added firing rate regularization
    
#    nb_batch,nb_steps,nb_inputs = input_data.shape
    nb_batch = 1
    nb_steps = input_data.shape[0]
    nb_inputs = input_data.shape[1]
    nb_hidden,nb_outputs = w2.shape
    lr,thr,alpha,beta,kappa,rho,t_ref,c_reg,f_target = decays[0], decays[1], decays[2], decays[3], decays[4], decays[5], decays[6],decays[7],decays[8] # get params
    out_rec = np.zeros((nb_batch, nb_steps, nb_outputs)) # output record
    v_rec = np.zeros((nb_batch, nb_steps, nb_hidden)) # hidden v record
    z_rec = np.zeros((nb_batch, nb_steps, nb_hidden)) # hidden z record
    a_rec = np.zeros((nb_batch, nb_steps, nb_hidden)) # hidden a record
    dw1 = np.zeros((nb_batch, nb_inputs, nb_hidden)) # input->hidden weight change
    dwr = np.zeros((nb_batch, nb_hidden, nb_hidden)) # hidden weight change
    dw2 = np.zeros((nb_batch, nb_hidden, nb_outputs)) # hidden->output weight change
    dbias = np.zeros((nb_batch, nb_outputs)) # bias change
    loss = np.zeros(nb_batch) # loss value
    sum_z = 0
    f_ave = 0
    
#    for b in nb.prange(int(nb_batch)): # prarallel processing, change "nb.prange" to "range" when not using parallel
    for b in range(int(nb_batch)): # prarallel processing, change "nb.prange" to "range" when not using parallel
        syn_from_input = np.dot(input_data, w1) # synaptic current from input
        z = np.zeros((nb_hidden,)) # spike or not (1 or 0)
#        z_bool = np.zeros((nb_hidden,),dtype=np.boolean) # spike or not (True or False)
        z_bool = np.zeros((nb_hidden,),dtype=bool) # spike or not (True or False)        
#        z_counts = np.zeros((nb_hidden,),dtype=np.int16) # spike count for refractory period implementation
        z_counts = np.zeros((nb_hidden,),dtype=int) # spike count for refractory period implementation
        v = np.zeros((nb_hidden,)) # voltage
        y = np.zeros((nb_outputs,)) # output
        a = np.zeros((nb_hidden,)) # adaptation
        eps_ijv = np.zeros((nb_hidden,1)) # eligibility vector for v(has only an "i" index) /// = z_i
        eps_ija = np.zeros((nb_hidden, nb_hidden)) # eligibility vector for a
        epsin_ijv = np.zeros((nb_inputs,1)) # eligibility vector for input v
        eps_jkv = np.zeros((nb_hidden,)) # eligibility vector for output v
        phi_j = np.zeros((nb_hidden,)) # pseudo derivative
        
        
        for t in range(nb_steps): # t=0
            # adaptation update (t-1)
            a = rho*a + z # adaptation update
            A = thr + beta*a # threshold update
            
            # spike update (t-1)
            z_bool = v > A # find spikes
#            print("z_bool=", z_bool)
            z_counts[z_bool] = z_counts[z_bool] + t_ref #!!! refractory period
            nrn_ready = np.where(z_counts==0)[0] # neurons that ready to fire (no refractory period)
            
            # pseudo derivate update (only for non-refractory period neurons) (t)
            phi_j[:] = 0.
            phi_j[nrn_ready] = (0.3/thr)*np.maximum(0, 1-np.abs((v[nrn_ready]-A[nrn_ready])/thr)) #  /// [nb_hidden]
            
            # voltage update (t)
#            z = z_bool.astype(nb.float64) # change data type
            z = z_bool.astype(float) # change data type            
#            print(np.sum(z))
            syn = syn_from_input[t] + np.dot(z,wr) # total synaptic currents /// [nb_hidden]
            v[z_bool] = A[z_bool] #!!! voltage is not larger than threshold
            v_new = alpha*v - A*z + syn # voltage update
            v = v_new 
            
            # output update (t)
            y_new = kappa*y + np.dot(z,w2) + bias #!!! + bias
            y = y_new 
            
            # eligibility trace for eij (t)
            eps_ijv = alpha*eps_ijv + z.reshape(-1,1) # notice that, "eps_ijv = eps_iv"
            eps_ija = eps_ijv*phi_j.reshape(1,-1) + rho*eps_ija #!!! faster than using "outer" function
            eij = eps_ijv*phi_j.reshape(1,-1) - beta*eps_ija*phi_j.reshape(1,-1) #!!! faster than using "outer" function
            
            # eligibility trace for eij for input->output (t)
#            epsin_ijv = alpha*epsin_ijv + input_data[b,t].reshape(-1,1)
            epsin_ijv = alpha*epsin_ijv + input_data[t].reshape(-1,1)
            eij_in = epsin_ijv*phi_j.reshape(1,-1)
            
            # eiligibility trace for eij for output (t)
            eps_jkv = kappa*eps_jkv + z
            
            if np.mod((t+1),100) == 0:
#            del_y = y - target_y[b,t]   #!!! delta_y /// [nb_outputs]    
                del_y = y - target_y[t]   #!!! delta_y /// [nb_outputs]    
#            print("del_y=", del_y)
                loss[b] += 0.5*np.sum((del_y)**2)   # SE
#            loss[b] += np.mean((del_y)**2)   # MSE

#            print('loss=',loss[b])
                lsig = np.dot(del_y, B) # learning signal /// [nb_outputs],[nb_outputs,nb_hidden]-->[nb_hidden]
                
            # (1) update recurrents
                dwr[b] += -lr*lsig.reshape(1,nb_hidden)*eij # [1,nb_hidden],[nb_hidden,nb_hidden]-->[nb_hidden,nb_hidden]
            
            # firing rate regularization
#            if t>0:
#                sum_z += z
#                f_ave = sum_z/(t)
#                #print("f_ave=", np.mean(f_ave))
#                dwr[b] += lr* c_reg *(f_target - f_ave)*eij /t              
            
            # (2) update inputs-->recurrent
                dw1[b] += -lr*lsig.reshape(1,nb_hidden)*eij_in # [1,nb_hidden],[nb_inputs,nb_hidden]-->[nb_inputs,nb_hidden]
              
            # (3) update of recurrent-->output neurons (below eq.20 in supplementary3)
                dw2[b] += -lr*eps_jkv.reshape(-1,1)*del_y.reshape(1,-1)
            
            # L2 norm regularization
            
            # (4) bias update
                dbias[b] += -lr*del_y 
                
            z_counts[z_counts>=1] = z_counts[z_counts>=1]-1 # spike count decay
            v_rec[b,t] = v # save v
            z_rec[b,t] = z # save z
            out_rec[b,t] = y # save y
            a_rec[b,t] = a # save a
    return loss, out_rec, dw1, dwr, dw2, dbias, v_rec, z_rec, a_rec






@nb.jit(nopython=True, parallel=True)
def lif_eprop7(w1,wr,w2,bias,B,input_data,target_y,params):
	# parallel version of "lif_eprop7"
    # regression version (see supp info P.17)
    # added firing rate regularization
    
    batch_size,nb_steps,nb_inputs = input_data.shape
    nb_hidden,nb_outputs = w2.shape
    lr,thr,alpha,beta,kappa,rho,t_ref,c_reg,f_target = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7], params[8] # get params
    out_rec = np.zeros((batch_size, nb_steps, nb_outputs)) # output record
    v_rec = np.zeros((batch_size, nb_steps, nb_hidden)) # hidden v record
    z_rec = np.zeros((batch_size, nb_steps, nb_hidden)) # hidden z record
    a_rec = np.zeros((batch_size, nb_steps, nb_hidden)) # hidden a record
    dw1 = np.zeros((batch_size, nb_inputs, nb_hidden)) # input->hidden weight change
    dwr = np.zeros((batch_size, nb_hidden, nb_hidden)) # hidden weight change
    dw2 = np.zeros((batch_size, nb_hidden, nb_outputs)) # hidden->output weight change
    dbias = np.zeros((batch_size, nb_outputs)) # bias change
    loss = np.zeros((batch_size)) # loss value
    sum_z = 0
    f_ave = 0

    for b in nb.prange(int(batch_size)): # prarallel processing, change "nb.prange" to "range" when not using parallel
        syn_from_input = np.dot(input_data[b], w1) # synaptic current from input
        z = np.zeros((nb_hidden,)) # spike or not (1 or 0)
        z_bool = np.zeros((nb_hidden,),dtype=nb.boolean) # spike or not (True or False)
        z_counts = np.zeros((nb_hidden,),dtype=nb.int16) # spike count for refractory period implementation
        v = np.zeros((nb_hidden,)) # voltage
        y = np.zeros((nb_outputs,)) # output
        a = np.zeros((nb_hidden,)) # adaptation
        eps_ijv = np.zeros((nb_hidden,1)) # eligibility vector for v(has only an "i" index) /// = z_i
        eps_ija = np.zeros((nb_hidden, nb_hidden)) # eligibility vector for a
        epsin_ijv = np.zeros((nb_inputs,1)) # eligibility vector for input v
        eps_jkv = np.zeros((nb_hidden,)) # eligibility vector for output v
        phi_j = np.zeros((nb_hidden,)) # pseudo derivative

        for t in range(nb_steps): # t=0
            # adaptation update (t-1)
            a = rho*a + z # adaptation update
            A = thr + beta*a # threshold update
            
            # spike update (t-1)
            z_bool = v > A # find spikes
            z_counts[z_bool] = z_counts[z_bool] + t_ref #!!! refractory period
            nrn_ready = np.where(z_counts==0)[0] # neurons that ready to fire (no refractory period)
            
            # pseudo derivate update (only for non-refractory period neurons) (t)
            phi_j[:] = 0.
            phi_j[nrn_ready] = (0.3/thr)*np.maximum(0, 1-np.abs((v[nrn_ready]-A[nrn_ready])/thr)) #  /// [nb_hidden]
            
            # voltage update (t)
            z = z_bool.astype(nb.float64) # change data type
            syn = syn_from_input[t] + np.dot(z,wr) # total synaptic currents /// [nb_hidden]
            v[z_bool] = A[z_bool] #!!! voltage is not larger than threshold
            v_new = alpha*v - A*z + syn # voltage update
            v = v_new 
            
            # output update (t)
            y_new = kappa*y + np.dot(z,w2) + bias  #!!! + bias
            y = y_new 
            
            # eligibility trace for eij (t)
            eps_ijv = alpha*eps_ijv + z.reshape(-1,1) # notice that, "eps_ijv = eps_iv"
            eps_ija = eps_ijv*phi_j.reshape(1,-1) + rho*eps_ija #!!! faster than using "outer" function
            eij = eps_ijv*phi_j.reshape(1,-1) - beta*eps_ija*phi_j.reshape(1,-1) #!!! faster than using "outer" function
            
            # eligibility trace for eij for input->output (t)
            epsin_ijv = alpha*epsin_ijv + input_data[b,t].reshape(-1,1)
            eij_in = epsin_ijv*phi_j.reshape(1,-1)
            
            # eiligibility trace for eij for output (t)
            eps_jkv = kappa*eps_jkv + z
     
            # update weight every 100ms
            if np.mod((t+1),100) == 0:

                del_y = y - target_y[b,t]   #!!! delta_y /// [nb_outputs]    
                lsig = np.dot(del_y, B) # learning signal /// [nb_outputs],[nb_outputs,nb_hidden]-->[nb_hidden]
                
                # (1) update recurrents
                # eq 27 of Methods
                dwr[b] += -lr*lsig.reshape(1,nb_hidden)*eij # [1,nb_hidden],[nb_hidden,nb_hidden]-->[nb_hidden,nb_hidden]
                
                # firing rate regularization
#               if t>0:
#                    sum_z += z
#                    f_ave = sum_z/(t)
#                    #print("f_ave=", np.mean(f_ave))
#                    dwr[b] += lr* c_reg *(f_target - f_ave)*eij /t 

                # (2) update inputs-->recurrent
                dw1[b] += -lr*lsig.reshape(1,nb_hidden)*eij_in # [1,nb_hidden],[nb_inputs,nb_hidden]-->[nb_inputs,nb_hidden]
                
                # (3) update of recurrent-->output neurons (below eq.20 in supplementary3)
                dw2[b] += -lr*eps_jkv.reshape(-1,1)*del_y.reshape(1,-1)
                
                # L2 norm regularization
                
                # (4) bias update
                dbias[b] += -lr*del_y 
                
                loss[b] += 0.5*np.sum((del_y)**2)   # SE
#                loss[b] += -np.sum(target_1hot[b,t]*np.log(pi+1e-10)) # cross entropy
                
            z_counts[z_counts>=1] = z_counts[z_counts>=1]-1 # spike count decay
            v_rec[b,t] = v # save v
            z_rec[b,t] = z # save z
            out_rec[b,t] = y # save y
            a_rec[b,t] = a # save a
    return loss, out_rec, dw1, dwr, dw2, dbias, v_rec, z_rec, a_rec