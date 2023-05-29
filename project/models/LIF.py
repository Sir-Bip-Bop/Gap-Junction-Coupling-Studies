
import numpy as np 
from scipy.sparse import dok_matrix
def IF_RK(y,order,gl,El,C,I,tau,k,v_neurons):
    '''
    Algorithm that integrates the LIF model, returning the float dydt, the change in the signal
    '''
    Vrest = -80
    dvdt = (-gl * (y[0] - El) + I - k * np.sum(y[0] - v_neurons) - y[1]* (y[0] - Vrest)) / C  

    y = np.append(y,0)

    for i in range(1,1+order):
        y[i] =  -y[i] / tau + y[i+1]
    dydt = [dvdt]
    dydt = np.array(dydt,dtype=object)

    for i in range(1,1+order):
        dydt = np.append(dydt,float(y[i]))

    return dydt

def rk_if(dt,t_final,order,y0,Vth,Vr,w,gl,El,C,I,Isyn,strength,tau,spikelet):
    ''' 
    Runge-Kutta integration of the 4th order of the LIF model
    '''
    Nsteps = int(t_final/dt)
    if type(y0) is int:
        y0 = [y0]
        I = [I]
    num_neurons =len(y0)

    if order >5:
        order = 5
    
    Y = np.zeros( (Nsteps, num_neurons * (1 + order)))
    data = np.zeros((Nsteps, num_neurons))
    data[0,:] = y0 
    end = num_neurons * (1 + order) -1

    for i in range(0,num_neurons):
        Y[0, i * (1+order)] = y0[i]
    
    for i in range(0,Nsteps-1):
        for k in range(0,num_neurons):
            k1 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] ,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order])
            k2 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] +0.5 * dt * k1,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order])
            k3 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] +0.5 * dt * k2,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order])
            k4 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] + dt * k3,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order])

            Y[i+1,k*(1+order):(k+1)*(1+order)] = Y[i,k*(1+order):(k+1)*(1+order)] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        for k in range(0,num_neurons):
            if Y[i+1,k * (1+order)] >= Vth:
                data[i+1,k] = w 
                Y[i+1, k * (1+order)] = Vr 
                for l in range(0,num_neurons):
                    if l != k:
                        Y[i+1,l * (1+order) +order] = Y[i+1,l*(1+order) + order] + Isyn[k,l]
                        Y[i+1,l * (1+order)] = Y[i+1,l *(1+order)] + spikelet
            else:
                data[i+1,k] = Y[i+1,k *(1+order)]
    return data, Y

def rk_if_Rossum(dt,t_final,order,y0,Vth,Vr,w,gl,El,C,I,Isyn,strength,tau,spikelet):
    ''' 
    Runge-Kutta integration of the 4th order of the LIF model
    '''
    Nsteps = int(t_final/dt)
    if type(y0) is int:
        y0 = [y0]
        I = [I]
    num_neurons =len(y0)

    if order >5:
        order = 5
    matrix = dok_matrix((num_neurons,int(t_final/dt)))
    Y = np.zeros( (Nsteps, num_neurons * (1 + order)))
    data = np.zeros((Nsteps, num_neurons))
    data[0,:] = y0 
    end = num_neurons * (1 + order) -1

    for i in range(0,num_neurons):
        Y[0, i * (1+order)] = y0[i]
    
    for i in range(0,Nsteps-1):
        for k in range(0,num_neurons):
            k1 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] ,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order])
            k2 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] +0.5 * dt * k1,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order])
            k3 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] +0.5 * dt * k2,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order])
            k4 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] + dt * k3,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order])

            Y[i+1,k*(1+order):(k+1)*(1+order)] = Y[i,k*(1+order):(k+1)*(1+order)] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        for k in range(0,num_neurons):
            if Y[i+1,k * (1+order)] >= Vth:
                data[i+1,k] = w 
                Y[i+1, k * (1+order)] = Vr 
                matrix[k,i] = 1
                for l in range(0,num_neurons):
                    if l != k:
                        Y[i+1,l * (1+order) +order] = Y[i+1,l*(1+order) + order] + Isyn[k,l]
                        Y[i+1,l * (1+order)] = Y[i+1,l *(1+order)] + spikelet
            else:
                data[i+1,k] = Y[i+1,k *(1+order)]
    return data, Y, matrix


def rk_if_Rossum_parallel(dt,t_final,order,y0,Vth,Vr,w,gl,El,C,I,Isyn,strength,tau,spikelet,return_dict):
    ''' 
    Runge-Kutta integration of the 4th order of the LIF model
    '''
    Nsteps = int(t_final/dt)
    if type(y0) is int:
        y0 = [y0]
        I = [I]
    num_neurons =len(y0)

    if order >5:
        order = 5
    matrix = dok_matrix((num_neurons,int(t_final/dt)))
    Y = np.zeros( (Nsteps, num_neurons * (1 + order)))
    data = np.zeros((Nsteps, num_neurons))
    data[0,:] = y0 
    end = num_neurons * (1 + order) -1

    for i in range(0,num_neurons):
        Y[0, i * (1+order)] = y0[i]
    
    for i in range(0,Nsteps-1):
        for k in range(0,num_neurons):
            k1 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] ,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order])
            k2 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] +0.5 * dt * k1,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order])
            k3 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] +0.5 * dt * k2,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order])
            k4 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] + dt * k3,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order])

            Y[i+1,k*(1+order):(k+1)*(1+order)] = Y[i,k*(1+order):(k+1)*(1+order)] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        for k in range(0,num_neurons):
            if Y[i+1,k * (1+order)] >= Vth:
                data[i+1,k] = w 
                Y[i+1, k * (1+order)] = Vr 
                matrix[k,i] = 1
                for l in range(0,num_neurons):
                    if l != k:
                        Y[i+1,l * (1+order) +order] = Y[i+1,l*(1+order) + order] + Isyn[k,l]
                        Y[i+1,l * (1+order)] = Y[i+1,l *(1+order)] + spikelet
            else:
                data[i+1,k] = Y[i+1,k *(1+order)]
    return_dict['data_IF'] = data 
    return_dict['Y_IF'] = Y 
    return_dict['Matrix_IF'] = np.array(matrix.todense())

def IF_RK_2(y,synaptic,order,gl,El,C,I,tau,k,A):
    '''
    Algorithm that integrates the LIF model, returning the float dydt, the change in the signal
    '''
    Vrest = -80
    I_gap = np.ravel((A.multiply( np.subtract.outer(y, y))).sum(axis=0))
    #print(np.shape(np.subtract(y,El)),np.shape(I_gap),np.shape(np.multiply(synaptic[0:len(y)-1],(y- Vrest))))
    dvdt = (-gl * np.subtract(y,El) + I + k * I_gap - np.multiply(synaptic[0:len(y)],(y- Vrest)) ) / C  

    for i in range(0,order):
        if i == order -1 :
             synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau
        else:
            synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau + synaptic[(i+1)*len(y):(i+2)*len(y)]
    
    dydt = [dvdt]
    dydt = np.array(dydt,dtype=object)
    return dydt

def rk_if_2(dt,t_final,order,y0,Vth,Vr,w,gl,El,C,I,Isyn,strength,tau,spikelet,E_matrix,C_matrix):
    ''' 
    Runge-Kutta integration of the 4th order of the LIF model
    '''

    #obtain the number of steps of the simulation
    Nsteps = int(t_final/dt)
    
    #we are assuming we are working with arrays, so transform everything into one
    if type(y0) is int:
        y0 = [y0]
        I = np.array( [ [I], [I] ] )

    #compute the number of neurons
    num_neurons =len(y0)

    #we are only allowing a synaptic filtering order up to 5
    if order > 5:
        print('We are changing down the filtering order to the maximum: 5')
        order = 5

    Y = np.zeros( (Nsteps, num_neurons))
    data = np.zeros( (Nsteps, num_neurons))
    synaptic = np.zeros((Nsteps,order*num_neurons))

    #assign the initial values
    for i in range(0,num_neurons):
        Y[0,i] = y0[i]
        data[0,i] = y0[i]
    
    #Runge-Kutta 4th order method 
    for i in range(0,Nsteps-1):
        k1 = IF_RK_2( Y[i,:] ,synaptic[i,:],order,gl,El,C,I[i,:],tau,strength,E_matrix)
        k2 = IF_RK_2( np.float64(Y[i,:] +0.5 * dt * k1[0]),synaptic[i,:],order,gl,El,C,I[i,:],tau,strength,E_matrix)
        k3 = IF_RK_2( np.float64(Y[i,:] +0.5 * dt * k2[0]),synaptic[i,:],order,gl,El,C,I[i,:],tau,strength,E_matrix)
        k4 = IF_RK_2( np.float64(Y[i,:] + dt * k3[0]) ,synaptic[i,:],order,gl,El,C,I[i,:],tau,strength,E_matrix)

        Y[i+1,:] = Y[i,:] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        if i > 0:
            spikes = np.where( Y[i+1,:] >= Vth)
            if len(spikes[0]) > 0:
                for spike_ind in spikes[0]:
                    data[i+1,:] = Y[i+1,:]
                    data[i+1,spike_ind] = w 
                    Y[i+1,spike_ind] = Vr 
                    synaptic[i+1,(order-1)*num_neurons:order*num_neurons] = synaptic[i+1,(order-1)*num_neurons:order*num_neurons] + C_matrix[spike_ind,:] *Isyn
                    Y[i+1,:] = Y[i+1,:] +  C_matrix[spike_ind,:] *spikelet
            else:
                data[i+1,:] = Y[i+1,:]
        else:
            data[i+1,:] = Y[i+1,:]
    return data, Y

def rk_if_2_Rossum(dt,t_final,order,y0,Vth,Vr,w,gl,El,C,I,Isyn,strength,tau,spikelet,E_matrix,C_matrix,return_dict):
    ''' 
    Runge-Kutta integration of the 4th order of the LIF model
    '''

    #obtain the number of steps of the simulation
    Nsteps = int(t_final/dt)
    
    #we are assuming we are working with arrays, so transform everything into one
    if type(y0) is int:
        y0 = [y0]
        I = np.array( [ [I], [I] ] )

    #compute the number of neurons
    num_neurons =len(y0)

    #we are only allowing a synaptic filtering order up to 5
    if order > 5:
        print('We are changing down the filtering order to the maximum: 5')
        order = 5

    matrix = dok_matrix((num_neurons,int(t_final/dt)))

    Y = np.zeros( (Nsteps, num_neurons))
    data = np.zeros( (Nsteps, num_neurons))
    synaptic = np.zeros((Nsteps,order*num_neurons))

    #assign the initial values
    for i in range(0,num_neurons):
        Y[0,i] = y0[i]
        data[0,i] = y0[i]

    check = np.zeros(num_neurons)
    
    #Runge-Kutta 4th order method 
    for i in range(0,Nsteps-1):
        k1 = IF_RK_2( Y[i,:] ,synaptic[i,:],order,gl,El,C,I[i,:],tau,strength,E_matrix)
        k2 = IF_RK_2( np.float64(Y[i,:] +0.5 * dt * k1[0]),synaptic[i,:],order,gl,El,C,I[i,:],tau,strength,E_matrix)
        k3 = IF_RK_2( np.float64(Y[i,:] +0.5 * dt * k2[0]),synaptic[i,:],order,gl,El,C,I[i,:],tau,strength,E_matrix)
        k4 = IF_RK_2( np.float64(Y[i,:] + dt * k3[0]) ,synaptic[i,:],order,gl,El,C,I[i,:],tau,strength,E_matrix)

        Y[i+1,:] = Y[i,:] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        if i > 0:
            spikes = np.where( Y[i+1,:] >= Vth)
            if len(spikes[0]) > 0:
                for spike_ind in spikes[0]:
                    if check[spike_ind] == 0:
                        check[spike_ind] = 1
                        matrix[spike_ind,i] = 1
                        data[i+1,:] = Y[i+1,:]
                        data[i+1,spike_ind] = w 
                        Y[i+1,spike_ind] = Vr 
                        synaptic[i+1,(order-1)*num_neurons:order*num_neurons] = synaptic[i+1,(order-1)*num_neurons:order*num_neurons] + C_matrix[spike_ind,:] *Isyn
                        Y[i+1,:] = Y[i+1,:] +  C_matrix[spike_ind,:] *spikelet
            else:
                data[i+1,:] = Y[i+1,:]
        else:
            data[i+1,:] = Y[i+1,:]

        negatives = np.where(Y[i,:]< 0)
        if len(negatives[0]) > 0:
            for index in negatives[0]:
                check[index] = 0 


    return data, Y, matrix

def rk_if_scale_synaptic(dt,t_final,order,y0,Vth,Vr,w,gl,El,C,I,Isyn,strength,tau,spikelet):
    ''' 
    Runge-Kutta integration of the 4th order of the LIF model
    '''
    Nsteps = int(t_final/dt)
    if type(y0) is int:
        y0 = [y0]
        I = [I]
    num_neurons =len(y0)

    if order >5:
        order = 5
    matrix = dok_matrix((num_neurons,int(t_final/dt)))
    Y = np.zeros( (Nsteps, num_neurons * (1 + order)))
    data = np.zeros((Nsteps, num_neurons))
    data[0,:] = y0 
    end = num_neurons * (1 + order) -1

    for i in range(0,num_neurons):
        Y[0, i * (1+order)] = y0[i]

    #Compute the number of connections of the neuron, in the case of two cells, it's just one
    num_connections = np.ones(num_neurons)
    
    for i in range(0,Nsteps-1):
        for k in range(0,num_neurons):
            k1 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] ,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order])
            k2 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] +0.5 * dt * k1,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order])
            k3 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] +0.5 * dt * k2,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order])
            k4 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] + dt * k3,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order])

            Y[i+1,k*(1+order):(k+1)*(1+order)] = Y[i,k*(1+order):(k+1)*(1+order)] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        for k in range(0,num_neurons):
            if Y[i+1,k * (1+order)] >= Vth:
                data[i+1,k] = w 
                Y[i+1, k * (1+order)] = Vr 
                matrix[k,i] = 1
                for l in range(0,num_neurons):
                    if l != k:
                        Y[i+1,l * (1+order) +order] = Y[i+1,l*(1+order) + order] + Isyn[k,l] / num_connections[k]
                        Y[i+1,l * (1+order)] = Y[i+1,l *(1+order)] + spikelet /num_connections[k]
            else:
                data[i+1,k] = Y[i+1,k *(1+order)]
    return data, Y, matrix

def rk_if_2_scale_synaptic(dt,t_final,order,y0,Vth,Vr,w,gl,El,C,I,Isyn,strength,tau,spikelet,E_matrix,C_matrix,return_dict):
    ''' 
    Runge-Kutta integration of the 4th order of the LIF model
    '''

    #obtain the number of steps of the simulation
    Nsteps = int(t_final/dt)
    
    #we are assuming we are working with arrays, so transform everything into one
    if type(y0) is int:
        y0 = [y0]
        I = np.array( [ [I], [I] ] )

    #compute the number of neurons
    num_neurons =len(y0)

    #we are only allowing a synaptic filtering order up to 5
    if order > 5:
        print('We are changing down the filtering order to the maximum: 5')
        order = 5

    matrix = dok_matrix((num_neurons,int(t_final/dt)))

    Y = np.zeros( (Nsteps, num_neurons))
    data = np.zeros( (Nsteps, num_neurons))
    synaptic = np.zeros((Nsteps,order*num_neurons))

    #assign the initial values
    for i in range(0,num_neurons):
        Y[0,i] = y0[i]
        data[0,i] = y0[i]

    check = np.zeros(num_neurons)

    #compute the number of connections of each neuron
    num_connections = np.zeros(num_neurons)
    for i in range(0,num_neurons):
        num_connections[i] = len((np.argwhere(np.array(C_matrix.todense())[i,:]>0) * dt).flatten())
    
    #Runge-Kutta 4th order method 
    for i in range(0,Nsteps-1):
        k1 = IF_RK_2( Y[i,:] ,synaptic[i,:],order,gl,El,C,I[i,:],tau,strength,E_matrix)
        k2 = IF_RK_2( np.float64(Y[i,:] +0.5 * dt * k1[0]),synaptic[i,:],order,gl,El,C,I[i,:],tau,strength,E_matrix)
        k3 = IF_RK_2( np.float64(Y[i,:] +0.5 * dt * k2[0]),synaptic[i,:],order,gl,El,C,I[i,:],tau,strength,E_matrix)
        k4 = IF_RK_2( np.float64(Y[i,:] + dt * k3[0]) ,synaptic[i,:],order,gl,El,C,I[i,:],tau,strength,E_matrix)

        Y[i+1,:] = Y[i,:] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        if i > 0:
            spikes = np.where( Y[i+1,:] >= Vth)
            if len(spikes[0]) > 0:
                for spike_ind in spikes[0]:
                    if check[spike_ind] == 0:
                        check[spike_ind] = 1
                        matrix[spike_ind,i] = 1
                        data[i+1,:] = Y[i+1,:]
                        data[i+1,spike_ind] = w 
                        Y[i+1,spike_ind] = Vr 
                        synaptic[i+1,(order-1)*num_neurons:order*num_neurons] = synaptic[i+1,(order-1)*num_neurons:order*num_neurons] + C_matrix[spike_ind,:] *Isyn / num_connections[spike_ind]
                        Y[i+1,:] = Y[i+1,:] +  E_matrix[spike_ind,:] *spikelet 
            else:
                data[i+1,:] = Y[i+1,:]
        else:
            data[i+1,:] = Y[i+1,:]

        negatives = np.where(Y[i,:]< 0)
        if len(negatives[0]) > 0:
            for index in negatives[0]:
                check[index] = 0 


    return data, Y, matrix, synaptic