import numpy as np 
from scipy.sparse import dok_matrix

def ML_RK(y, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I, tau, k, v_neurons,A ):
    '''
    Algorithm of the evolution of the Morris-Lecar model, returning the numpy array dydt
    '''
    Vrest = -80


    minf = 0.5 * (1 + np.tanh( ((y[0] - V1 )/ V2)))
    Iion = gna * minf * (y[0] - Ena) + gk * y[1] * (y[0] -Ek) + gshunt * (y[0] - Eshunt)

    dvdt = ( - Iion - k * np.sum(A*(y[0] - v_neurons)) + I - y[2] * (y[0]- Vrest)) / C

    #dvdt = (- Iion + I) / C
    #print(gna * minf * (y[0] - Ena),k * y[1] * (y[0] -Ek),gshunt * (y[0] - Eshunt), np.tanh( ((y[0] - V1 )/ V2)))


    winf = 0.5 * (1 + np.tanh( (y[0] - V3) / V4))
    dwdt = psi * (winf - y[1])*np.cosh( (y[0] - V3) / 2 / V4)
    #print(dwdt)

    y = np.append(y,0)
    for i in range(2, 2+order):
        y[i] = -y[i] / tau + y[i+1]

    dydt = [dvdt,dwdt]
    dydt = np.array(dydt,dtype=object)
    for i in range(2, 2+order):
        dydt = np.append(dydt,float(y[i]))
    return dydt




def rk_ml(dt,t_final,order,y0,w0,psi,V1,V2,V3,V4,gna,gk,gshunt,Ena,Ek,Eshunt,C,I,Isyn,strength,tau,E_matrix,C_matrix):
    '''
    Runge-Kutta integration of the 4th order of the Morris-Lecar model
    '''

    Nsteps = int(t_final/dt)
    if type(y0) is int:
        y0 = [y0]
        I = np.array( [ [I], [I] ])
    num_neurons = len(y0)

    if order >= 5:
        order = 5

    Y = np.zeros((Nsteps, num_neurons * (2+order)))
    data = np.zeros((Nsteps, num_neurons))
    end = num_neurons * (2+order) -1
    for i in range (0,num_neurons):
        Y[0,i*(2+order)] = y0[i]
        Y[0,1 + i*(2+order)] = w0[i]

        data[0,i] = y0[i]

    for i in range(0,Nsteps - 1):
        for k in range(0,num_neurons):
            k1 = ML_RK(Y[i, k*(2+order): (k+1) * (2+order)], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, strength, Y[i, 0:end:2+order],E_matrix[k,:] )
            #print('k1',k1) a[0:4+1:4+1]
            k2 = ML_RK(Y[i, k*(2+order): (k+1) * (2+order)] + 0.5*dt*k1, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, strength, Y[i, 0:end:2+order] ,E_matrix[k,:] )
            #print('k2',k2)
            k3 = ML_RK(Y[i, k*(2+order): (k+1) * (2+order)] + 0.5*dt*k2, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, strength, Y[i, 0:end:2+order],E_matrix[k,:] )
            
            k4 = ML_RK(Y[i, k*(2+order): (k+1) * (2+order)] + dt * k3, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, strength, Y[i, 0:end:2+order] ,E_matrix[k,:] )
            

            Y[i + 1, k * (2 + order): (k+1) *(2+order)] = Y[i, k * (2+order): (k+1)*(2+order) ] + 1/6 * dt * (k1 + 2*k2 + 2*k3 + k4)

        for k in range(0,num_neurons):
            if i>0 and ( Y[i, k*(2+order)] >= Y [i-1,k*(2+order)]) and (Y[i,k*(2+order)] >= Y[i+1,k*(2+order)]) and Y[i,k*(2+order)] > 0:
                for l in range(0,num_neurons):
                        Y[i+1,l*(2+order) + 2 + order-1] = Y[i+1,l * (2+order) +2 + order-1] + C_matrix[k,l] *Isyn[k,l]
            data[i+1,k] = Y[i+1,k*(2+order)]  
    return data, Y
def rk_ml_Rossum(dt,t_final,order,y0,w0,psi,V1,V2,V3,V4,gna,gk,gshunt,Ena,Ek,Eshunt,C,I,Isyn,strength,tau):
    '''
    Runge-Kutta integration of the 4th order of the Morris-Lecar model
    '''

    Nsteps = int(t_final/dt)
    if type(y0) is int:
        y0 = [y0]
        I = np.array( [ [I], [I] ])
    num_neurons = len(y0)

    check = np.zeros(num_neurons)
    if order >= 5:
        order = 5
    matrix = dok_matrix((num_neurons,int(t_final/dt)))
    Y = np.zeros((Nsteps, num_neurons * (2+order)))
    data = np.zeros((Nsteps, num_neurons))
    end = num_neurons * (2+order) -1
    for i in range (0,num_neurons):
        Y[0,i*(2+order)] = y0[i]
        Y[0,1 + i*(2+order)] = w0[i]

        data[0,i] = y0[i]

    for i in range(0,Nsteps - 1):
        for k in range(0,num_neurons):
            k1 = ML_RK(Y[i, k*(2+order): (k+1) * (2+order)], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, strength, Y[i, 0:end:2+order] )
            #print('k1',k1) a[0:4+1:4+1]
            k2 = ML_RK(Y[i, k*(2+order): (k+1) * (2+order)] + 0.5*dt*k1, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, strength, Y[i, 0:end:2+order] )
            #print('k2',k2)
            k3 = ML_RK(Y[i, k*(2+order): (k+1) * (2+order)] + 0.5*dt*k2, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, strength, Y[i, 0:end:2+order])
            
            k4 = ML_RK(Y[i, k*(2+order): (k+1) * (2+order)] + dt * k3, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, strength, Y[i, 0:end:2+order] )
            

            Y[i + 1, k * (2 + order): (k+1) *(2+order)] = Y[i, k * (2+order): (k+1)*(2+order) ] + 1/6 * dt * (k1 + 2*k2 + 2*k3 + k4)

        for k in range(0,num_neurons):
            #if i>0 and ( Y[i, k*(2+order)] >= Y [i-1,k*(2+order)]) and (Y[i,k*(2+order)] >= Y[i+1,k*(2+order)]) and Y[i,k*(2+order)] > 10 and check[k] == 0:
            if i > 0 and Y[i,k*(2+order)] > 10 and check[k] == 0:
                matrix[k,i] = 1
                check[k] = 1
                #print('Spike! time:',i * dt, 'neuron:',k)
                for l in range(0,num_neurons):
                    if l != k:
                        Y[i+1,l*(2+order) + 2 + order-1] = Y[i+1,l * (2+order) +2 + order-1] + Isyn[k,l]
            data[i+1,k] = Y[i+1,k*(2+order)]  
            if data[i+1,k] < 0:
                check[k] = 0

    return data, Y, matrix

def rk_ml_Rossum_parallel(dt,t_final,order,y0,w0,psi,V1,V2,V3,V4,gna,gk,gshunt,Ena,Ek,Eshunt,C,I,Isyn,strength,tau,return_dict):
    '''
    Runge-Kutta integration of the 4th order of the Morris-Lecar model
    '''

    Nsteps = int(t_final/dt)
    if type(y0) is int:
        y0 = [y0]
        I = np.array( [ [I], [I] ])
    num_neurons = len(y0)

    check = np.zeros(num_neurons)
    if order >= 5:
        order = 5
    matrix = dok_matrix((num_neurons,int(t_final/dt)))
    Y = np.zeros((Nsteps, num_neurons * (2+order)))
    data = np.zeros((Nsteps, num_neurons))
    end = num_neurons * (2+order) -1
    for i in range (0,num_neurons):
        Y[0,i*(2+order)] = y0[i]
        Y[0,1 + i*(2+order)] = w0[i]

        data[0,i] = y0[i]

    for i in range(0,Nsteps - 1):
        for k in range(0,num_neurons):
            k1 = ML_RK(Y[i, k*(2+order): (k+1) * (2+order)], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, strength, Y[i, 0:end:2+order] )
            #print('k1',k1) a[0:4+1:4+1]
            k2 = ML_RK(Y[i, k*(2+order): (k+1) * (2+order)] + 0.5*dt*k1, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, strength, Y[i, 0:end:2+order] )
            #print('k2',k2)
            k3 = ML_RK(Y[i, k*(2+order): (k+1) * (2+order)] + 0.5*dt*k2, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, strength, Y[i, 0:end:2+order])
            
            k4 = ML_RK(Y[i, k*(2+order): (k+1) * (2+order)] + dt * k3, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, strength, Y[i, 0:end:2+order] )
            

            Y[i + 1, k * (2 + order): (k+1) *(2+order)] = Y[i, k * (2+order): (k+1)*(2+order) ] + 1/6 * dt * (k1 + 2*k2 + 2*k3 + k4)

        for k in range(0,num_neurons):
            #if i>0 and ( Y[i, k*(2+order)] >= Y [i-1,k*(2+order)]) and (Y[i,k*(2+order)] >= Y[i+1,k*(2+order)]) and Y[i,k*(2+order)] > 10 and check[k] == 0:
            if i > 0 and Y[i,k*(2+order)] > 10 and check[k] == 0:
                matrix[k,i] = 1
                check[k] = 1
                #print('Spike! time:',i * dt, 'neuron:',k)
                for l in range(0,num_neurons):
                    if l != k:
                        Y[i+1,l*(2+order) + 2 + order-1] = Y[i+1,l * (2+order) +2 + order-1] + Isyn[k,l]
            data[i+1,k] = Y[i+1,k*(2+order)]  
            if data[i+1,k] < 0:
                check[k] = 0

    return_dict['data_ML'] = data 
    return_dict['Y_ML'] = Y 
    return_dict['Matrix_ML'] = np.array(matrix.todense())
    
def ML_RK_2(y, w, synaptic, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I, tau, k,A ):
    '''
    Algorithm of the evolution of the Morris-Lecar model, returning the numpy array dydt
    '''
    Vrest = -80


    minf = 0.5 * (1 + np.tanh( ((y - V1 )/ V2)))
    Iion = gna * minf * (y - Ena) + gk * w * (y -Ek) + gshunt * (y - Eshunt)

    dvdt = ( - Iion + k * np.ravel((A.multiply( np.subtract.outer(y, y))).sum(axis=0)) + I - np.multiply(synaptic[0:len(y)],(y- Vrest)) )/  C

    #dvdt = (- Iion + I) / C
    #print(gna * minf * (y[0] - Ena),k * y[1] * (y[0] -Ek),gshunt * (y[0] - Eshunt), np.tanh( ((y[0] - V1 )/ V2)))


    winf = 0.5 * (1 + np.tanh( (y - V3) / V4))
    dwdt = psi * np.multiply((winf - w),np.cosh( (y - V3) / 2 / V4))
    #print(dwdt)

    for i in range(0,order):
        if i == order -1 :
             synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau
        else:
            synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau + synaptic[(i+1)*len(y):(i+2)*len(y)]

    dydt = [dvdt,dwdt]
    dydt = np.array(dydt,dtype=object)

    return dydt




def rk_ml_2(dt,t_final,order,y0,w0,psi,V1,V2,V3,V4,gna,gk,gshunt,Ena,Ek,Eshunt,C,I,Isyn,strength,tau,E_matrix,C_matrix):
    '''
    Runge-Kutta integration of the 4th order of the Morris-Lecar model
    '''

    #obtain the number of steps of the simulation
    Nsteps = int(t_final/dt)

    #we are assuming we are working with arrays, so transform everything into one
    if type(y0) is int:
        y0 = [y0]
        w0 = [w0]
        I = np.array( [ [I], [I] ])

    #compute the number of neurons
    num_neurons = len(y0)

    #we are only allowing a synaptic filtering order up to 5
    if order > 5:
        print('We are changing down the filtering order to the maximum: 5')
        order = 5

    #variables that store the signal
    Y = np.zeros((Nsteps, num_neurons))
    W = np.zeros((Nsteps, num_neurons))
    synaptic = np.zeros((Nsteps, order*num_neurons))

    #assign the initial values
    for i in range (0,num_neurons):
        Y[0,i] = y0[i]
        W[0,i] = w0[i]

    #Runge-Kutta 4th order method 
    for i in range(0,Nsteps - 1):
        k1 = ML_RK_2(Y[i,:],W[i,:],synaptic[i,:], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,:], tau, strength,E_matrix )
        #print('k1',k1) a[0:4+1:4+1]
        k2 = ML_RK_2(np.float64(Y[i, :] + 0.5*dt*k1[0]),W[i,:]+0.5*dt*k1[1],synaptic[i,:], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,:], tau, strength,E_matrix)
        #print('k2',k2)
        k3 = ML_RK_2(np.float64(Y[i,:] + 0.5*dt*k2[0]), W[i,:]+0.5*dt*k1[1],synaptic[i,:], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,:], tau, strength,E_matrix )
            
        k4 = ML_RK_2(np.float64(Y[i,:] + dt * k3[0]), W[i,:]+0.5*dt*k3[1],synaptic[i,:], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,:], tau, strength,E_matrix)
            

        Y[i + 1, :] = Y[i,:] + 1/6 * dt * (k1[0] + 2*k2[0] + 2*k3[0]+ k4[0])
        W[i + 1, :] = W[i,:] + 1/6 * dt * (k1[1] + 2*k2[1] + 2*k3[1]+ k4[1])


        if i > 0:
            spikes =  np.where( (Y[i, :] >= Y [i-1,:]) & (Y[i,:] >= Y[i+1,:]) & (Y[i,:] > 0))
            if len(spikes[0]) > 0:
                for spike_ind in spikes[0]:
                        synaptic[i+1,(order-1)*num_neurons:order*num_neurons] = synaptic[i+1,(order-1)*num_neurons:order*num_neurons] + C_matrix[spike_ind,:] *Isyn

    return Y