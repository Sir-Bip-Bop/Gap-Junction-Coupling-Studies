import numpy as np 
from scipy.sparse import dok_matrix

def IS_RK(y,order,C,I,vr,vt,k_2,a,b,k,k_u,tau,v_neurons):
    '''
    Algorithm of the evolution of the Ishikevich model, returning the numpy array dudt
    '''
    Vrest = - 80 #which one to use
    dvdt = (k * (y[0] - vr) * (y[0] - vt) - k_u * y[1] + I - k_2 * np.sum(y[0]-v_neurons) - y[2] *(y[0]-Vrest))   / C
    dudt = a * (b*(y[0] - vr) - y[1])
    y = np.append(y,0)
    for i in range(2,2+order):
        y[i] = -y[i] / tau + y[i+1]
    dydt = [dvdt,dudt]
    dydt = np.array(dydt,dtype=object)
    for i in range(2,2+order):
        dydt = np.append(dydt,float(y[i]))

    return dydt 

def rk_ish(dt,t_final,order,y0,u0,I,Isyn,C,vr,vt,k_2,a,b,c,d,vpeak,k_u,strength,tau):
    '''
    Runge-Kutta integration of the 4th order of the Izhikevich model
    '''

    Nsteps = int(t_final/dt)
    if type(y0) is int:
        y0 = [y0]
        I = np.array( [ [I] , [I] ] )
    num_neurons = len(y0)

    if order >= 5:
        order = 5

    Y = np.zeros( (Nsteps, num_neurons * (2 + order)))
    end = num_neurons * (2 + order) -1
    data = np.zeros( (Nsteps, num_neurons))
    for i in range(0,num_neurons):
        Y[0,i*(2+order)] = y0[i]
        Y[0,1+i*(2+order)] = u0[i]

        #data we are outputing for convenience
        data[0,i] = y0[i]

    for i in range(0,Nsteps - 1):
        for k in range(0,num_neurons):
            k1 = IS_RK( Y[i,k*(2+order):(k+1)*(2+order)] ,order,C,I[i,k],vr,vt,strength,a,b,k_2,k_u,tau,Y[i,0:end:2+order])
            k2 = IS_RK( Y[i,k*(2+order):(k+1)*(2+order)] +0.5 * dt * k1,order,C,I[i,k],vr,vt,strength,a,b,k_2,k_u,tau,Y[i,0:end:2+order])
            k3 = IS_RK( Y[i,k*(2+order):(k+1)*(2+order)] +0.5 * dt * k2,order,C,I[i,k],vr,vt,strength,a,b,k_2,k_u,tau,Y[i,0:end:2+order])
            k4 = IS_RK( Y[i,k*(2+order):(k+1)*(2+order)] + dt * k3,order,C,I[i,k],vr,vt,strength,a,b,k_2,k_u,tau,Y[i,0:end:2+order])

            Y[i+1,k*(2+order):(k+1)*(2+order)] = Y[i,k*(2+order):(k+1)*(2+order)] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        for k in range(0,num_neurons): #never arrive at this conditions
            if Y[i+1, k * (2 + order)] >= vpeak:
                Y[i+1, k * (2+order)] = c 
                Y[i, k * (2 +order)] = vpeak #no influence
                Y[i+1, 1 + k * (2+order)] = Y[i+1, 1+k*(2+order)] + d
                for l in range(0,num_neurons):
                    if l!= k:
                        Y[i+1, l *(2+order) + 2 + order -1] = Y[i+1, l*(2+order) + 2 +order - 1] + Isyn[k,l]

            data[i+1,k] = Y[i+1,k*(2+order)]

    return data, Y

def rk_ish_Rossum(dt,t_final,order,y0,u0,I,Isyn,C,vr,vt,k_2,a,b,c,d,vpeak,k_u,strength,tau):
    '''
    Runge-Kutta integration of the 4th order of the Izhikevich model
    '''

    Nsteps = int(t_final/dt)
    if type(y0) is int:
        y0 = [y0]
        I = np.array( [ [I] , [I] ] )
    num_neurons = len(y0)

    if order >= 5:
        order = 5
    matrix = dok_matrix((num_neurons,int(t_final/dt)))
    Y = np.zeros( (Nsteps, num_neurons * (2 + order)))
    end = num_neurons * (2 + order) -1
    data = np.zeros( (Nsteps, num_neurons))
    for i in range(0,num_neurons):
        Y[0,i*(2+order)] = y0[i]
        Y[0,1+i*(2+order)] = u0[i]

        #data we are outputing for convenience
        data[0,i] = y0[i]

    for i in range(0,Nsteps - 1):
        for k in range(0,num_neurons):
            k1 = IS_RK( Y[i,k*(2+order):(k+1)*(2+order)] ,order,C,I[i,k],vr,vt,strength,a,b,k_2,k_u,tau,Y[i,0:end:2+order])
            k2 = IS_RK( Y[i,k*(2+order):(k+1)*(2+order)] +0.5 * dt * k1,order,C,I[i,k],vr,vt,strength,a,b,k_2,k_u,tau,Y[i,0:end:2+order])
            k3 = IS_RK( Y[i,k*(2+order):(k+1)*(2+order)] +0.5 * dt * k2,order,C,I[i,k],vr,vt,strength,a,b,k_2,k_u,tau,Y[i,0:end:2+order])
            k4 = IS_RK( Y[i,k*(2+order):(k+1)*(2+order)] + dt * k3,order,C,I[i,k],vr,vt,strength,a,b,k_2,k_u,tau,Y[i,0:end:2+order])

            Y[i+1,k*(2+order):(k+1)*(2+order)] = Y[i,k*(2+order):(k+1)*(2+order)] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        for k in range(0,num_neurons): #never arrive at this conditions
            if Y[i+1, k * (2 + order)] >= vpeak:
                Y[i+1, k * (2+order)] = c 
                Y[i, k * (2 +order)] = vpeak #no influence
                Y[i+1, 1 + k * (2+order)] = Y[i+1, 1+k*(2+order)] + d
                matrix[k,i] = 1
                for l in range(0,num_neurons):
                    if l!= k:
                        Y[i+1, l *(2+order) + 2 + order -1] = Y[i+1, l*(2+order) + 2 +order - 1] + Isyn[k,l]

            data[i+1,k] = Y[i+1,k*(2+order)]

    return data, Y, matrix


def rk_ish_Rossum_parallel(dt,t_final,order,y0,u0,I,Isyn,C,vr,vt,k_2,a,b,c,d,vpeak,k_u,strength,tau,return_dict):
    '''
    Runge-Kutta integration of the 4th order of the Izhikevich model
    '''

    Nsteps = int(t_final/dt)
    if type(y0) is int:
        y0 = [y0]
        I = np.array( [ [I] , [I] ] )
    num_neurons = len(y0)

    if order >= 5:
        order = 5
    matrix = dok_matrix((num_neurons,int(t_final/dt)))
    Y = np.zeros( (Nsteps, num_neurons * (2 + order)))
    end = num_neurons * (2 + order) -1
    data = np.zeros( (Nsteps, num_neurons))
    for i in range(0,num_neurons):
        Y[0,i*(2+order)] = y0[i]
        Y[0,1+i*(2+order)] = u0[i]

        #data we are outputing for convenience
        data[0,i] = y0[i]

    for i in range(0,Nsteps - 1):
        for k in range(0,num_neurons):
            k1 = IS_RK( Y[i,k*(2+order):(k+1)*(2+order)] ,order,C,I[i,k],vr,vt,strength,a,b,k_2,k_u,tau,Y[i,0:end:2+order])
            k2 = IS_RK( Y[i,k*(2+order):(k+1)*(2+order)] +0.5 * dt * k1,order,C,I[i,k],vr,vt,strength,a,b,k_2,k_u,tau,Y[i,0:end:2+order])
            k3 = IS_RK( Y[i,k*(2+order):(k+1)*(2+order)] +0.5 * dt * k2,order,C,I[i,k],vr,vt,strength,a,b,k_2,k_u,tau,Y[i,0:end:2+order])
            k4 = IS_RK( Y[i,k*(2+order):(k+1)*(2+order)] + dt * k3,order,C,I[i,k],vr,vt,strength,a,b,k_2,k_u,tau,Y[i,0:end:2+order])

            Y[i+1,k*(2+order):(k+1)*(2+order)] = Y[i,k*(2+order):(k+1)*(2+order)] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        for k in range(0,num_neurons): #never arrive at this conditions
            if Y[i+1, k * (2 + order)] >= vpeak:
                Y[i+1, k * (2+order)] = c 
                Y[i, k * (2 +order)] = vpeak #no influence
                Y[i+1, 1 + k * (2+order)] = Y[i+1, 1+k*(2+order)] + d
                matrix[k,i] = 1
                for l in range(0,num_neurons):
                    if l!= k:
                        Y[i+1, l *(2+order) + 2 + order -1] = Y[i+1, l*(2+order) + 2 +order - 1] + Isyn[k,l]

            data[i+1,k] = Y[i+1,k*(2+order)]

    return_dict['data_IZH'] = data 
    return_dict['Y_IZH'] = Y 
    return_dict['Matrix_IZH'] =np.array(matrix.todense())

def IS_RK_2(y,u,synaptic,order,C,I,vr,vt,k_2,a,b,k,k_u,tau,A):
    '''
    Algorithm of the evolution of the Ishikevich model, returning the numpy array dudt
    '''

    Vrest = - 80 
    dvdt = (k * np.multiply((y - vr),(y - vt)) - k_u * u + I + k_2 *np.ravel((A.multiply( np.subtract.outer(y, y))).sum(axis=0)) -np.multiply(synaptic[0:len(y)],(y- Vrest)) )  / C
    dudt = a * np.subtract(b*(y - vr),u)
    #print(b*(y - vr))
    for i in range(0,order):
        if i == order -1 :
             synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau
        else:
            synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau + synaptic[(i+1)*len(y):(i+2)*len(y)]

    dydt = [dvdt,dudt]
    dydt = np.array(dydt,dtype=object)
    return dydt 

def rk_ish_2(dt,t_final,order,y0,u0,I,Isyn,C,vr,vt,k_2,a,b,c,d,vpeak,k_u,strength,tau,E_matrix,C_matrix):
    '''
    Runge-Kutta integration of the 4th order of the Izhikevich model
    '''

    #obtain the number of steps of the simulation
    Nsteps = int(t_final/dt)

    #we are assuming we are working with arrays, so transform everything into one
    if type(y0) is int:
        y0 = [y0]
        I = np.array( [ [I] , [I] ] )

    #compute the number of neurons    
    num_neurons = len(y0)

    #we are only allowing a synaptic filtering order up to 5
    if order > 5:
        print('We are changing down the filtering order to the maximum: 5')
        order = 5

    #variables that store the signal
    Y = np.zeros( (Nsteps, num_neurons))
    U = np.zeros( (Nsteps,num_neurons))
    data = np.zeros( (Nsteps, num_neurons))
    synaptic = np.zeros((Nsteps,order*num_neurons))

    #assign the initial values
    for i in range(0,num_neurons):
        Y[0,i] = y0[i]
        U[0,i] = u0[i]
        data[0,i] = y0[i]

    #Runge-Kutta 4th order method 
    for i in range(0,Nsteps - 1):
        k1 = IS_RK_2( Y[i,:] ,U[i,:],synaptic[i,:],order,C,I[i,:],vr,vt,strength,a,b,k_2,k_u,tau, E_matrix)
        k2 = IS_RK_2( np.float64(Y[i,:] +0.5 * dt * k1[0]),U[i,:]+ 0.5*dt*k1[1],synaptic[i,:],order,C,I[i,:],vr,vt,strength,a,b,k_2,k_u,tau, E_matrix)
        k3 = IS_RK_2( np.float64(Y[i,:] +0.5 * dt * k2[0]),U[i,:] + 0.5*dt*k2[1],synaptic[i,:],order,C,I[i,:],vr,vt,strength,a,b,k_2,k_u,tau, E_matrix)
        k4 = IS_RK_2( np.float64(Y[i,:] + dt * k3[0]),U[i,:]+ dt*k3[1],synaptic[i,:],order,C,I[i,:],vr,vt,strength,a,b,k_2,k_u,tau, E_matrix)

        Y[i+1,:] = Y[i,:] + (1/6)*dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        U[i+1,:] = U[i,:] + (1/6)*dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])

        if i > 0:
            spikes = np.where( Y[i+1,:] >= vpeak)
            if len(spikes[0]) > 0:
                for spike_ind in spikes[0]:
                    Y[i+1, spike_ind] = c 
                    Y[i, spike_ind] = vpeak
                    U[i+1, spike_ind] = U[i+1, spike_ind] + d
                    synaptic[i+1,(order-1)*num_neurons:order*num_neurons] = synaptic[i+1,(order-1)*num_neurons:order*num_neurons] + C_matrix[spike_ind,:] *Isyn

        data[i+1,:] = Y[i+1,:]

    return data, Y

def rk_ish_2_Rossum(dt,t_final,order,y0,u0,I,Isyn,C,vr,vt,k_2,a,b,c,d,vpeak,k_u,strength,tau,E_matrix,C_matrix,return_dict):
    '''
    Runge-Kutta integration of the 4th order of the Izhikevich model
    '''

    #obtain the number of steps of the simulation
    Nsteps = int(t_final/dt)

    #we are assuming we are working with arrays, so transform everything into one
    if type(y0) is int:
        y0 = [y0]
        I = np.array( [ [I] , [I] ] )

    #compute the number of neurons    
    num_neurons = len(y0)

    #we are only allowing a synaptic filtering order up to 5
    if order > 5:
        print('We are changing down the filtering order to the maximum: 5')
        order = 5

    matrix = dok_matrix((num_neurons,int(t_final/dt)))

    #variables that store the signal
    Y = np.zeros( (Nsteps, num_neurons))
    U = np.zeros( (Nsteps,num_neurons))
    data = np.zeros( (Nsteps, num_neurons))
    synaptic = np.zeros((Nsteps,order*num_neurons))

    #assign the initial values
    for i in range(0,num_neurons):
        Y[0,i] = y0[i]
        U[0,i] = u0[i]
        data[0,i] = y0[i]
    
    check = np.zeros(num_neurons)

    #Runge-Kutta 4th order method 
    for i in range(0,Nsteps - 1):
        k1 = IS_RK_2( Y[i,:] ,U[i,:],synaptic[i,:],order,C,I[i,:],vr,vt,strength,a,b,k_2,k_u,tau, E_matrix)
        k2 = IS_RK_2( np.float64(Y[i,:] +0.5 * dt * k1[0]),U[i,:]+ 0.5*dt*k1[1],synaptic[i,:],order,C,I[i,:],vr,vt,strength,a,b,k_2,k_u,tau, E_matrix)
        k3 = IS_RK_2( np.float64(Y[i,:] +0.5 * dt * k2[0]),U[i,:] + 0.5*dt*k2[1],synaptic[i,:],order,C,I[i,:],vr,vt,strength,a,b,k_2,k_u,tau, E_matrix)
        k4 = IS_RK_2( np.float64(Y[i,:] + dt * k3[0]),U[i,:]+ dt*k3[1],synaptic[i,:],order,C,I[i,:],vr,vt,strength,a,b,k_2,k_u,tau, E_matrix)

        Y[i+1,:] = Y[i,:] + (1/6)*dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
        U[i+1,:] = U[i,:] + (1/6)*dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])

        if i > 0:
            spikes = np.where( Y[i+1,:] >= vpeak)
            if len(spikes[0]) > 0:
                for spike_ind in spikes[0]:
                    if check[spike_ind] == 0 :
                        check[spike_ind] = 1
                        matrix[spike_ind,i] = 1
                        Y[i+1, spike_ind] = c 
                        Y[i, spike_ind] = vpeak
                        U[i+1, spike_ind] = U[i+1, spike_ind] + d
                        synaptic[i+1,(order-1)*num_neurons:order*num_neurons] = synaptic[i+1,(order-1)*num_neurons:order*num_neurons] + C_matrix[spike_ind,:] *Isyn

        data[i+1,:] = Y[i+1,:]

        negatives = np.where(Y[i,:]< 0)
        if len(negatives[0]) > 0:
            for index in negatives[0]:
                check[index] = 0

    return data, Y, matrix
