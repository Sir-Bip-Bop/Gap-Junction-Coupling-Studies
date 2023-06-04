import numpy as np 
from scipy.sparse import dok_matrix
from scipy import sparse
from scipy.sparse import csr_matrix

def an_pairs(v,vt):
    return -0.032 * (v-vt-15) / (np.exp(-(v-vt-15)/5)-1)

def bn_pairs(v,vt):
    return 0.5* np.exp(-(v-vt-10)/40)

def am_pairs(v,vt):

    return -0.32 * (v-vt-13) / (np.exp(-(v-vt-13)/4)-1)

def bm_pairs(v,vt):
    return 0.28 * (v-vt-40) / (np.exp((v-vt-40)/5) -1)

def ah_pairs(v,vt):
    return 0.128*np.exp(-(v-vt-17)/18)

def bh_pairs(v,vt):
    return 4 / (1+np.exp(-(v-vt-40)/5))

def an_network(v,vt):
    v = v.astype(float)
    return np.array(-0.032 * (v-vt-15) / (np.exp(-(v-vt-15)/5)-1))

def bn_network(v,vt):
    v = v.astype(float)
    return np.array(0.5* np.exp(-(v-vt-10)/40))

def am_network(v,vt):
    v = v.astype(float)
    return np.array(-0.32 * (v-vt-13) / (np.exp(-(v-vt-13)/4)-1))

def bm_network(v,vt):
    v = v.astype(float)
    return np.array(0.28 * (v-vt-40) / (np.exp((v-vt-40)/5) -1))

def ah_network(v,vt):
    v = v.astype(float)
    return np.array(0.128*np.exp(-(v-vt-17)/18))

def bh_netowrk(v,vt):
    v = v.astype(float)
    return np.array(4 / (1+np.exp(-(v-vt-40)/5)))

def HH_Equation_Pairs(y,order,gna,gk,gl,Ena,Ek,El,C,I,tau,gap_junction,v_neurons):
    '''
    Algorithm that integrates the equations of the Hodgkin-Huxley model, as well as the synaptic filter.

    Parameters:
        y (tuple[float]):
            The signal of the Neuron - Voltage,  recovery variables : n,m,and h, and Synaptic Current.
        order (int):
            The order of the synaptic filter - Max value of 5
        gna (float):
            Conductance of the Na channel
        gk (float):
            Conductance of the K channel
        gl (float):
            Conductance of the leak channel
        Ena (float):
            Rest voltage of the Na channel
        Ek (float):
            Rest voltage of the K channel
        El (float):
            Rest voltage of the leak channel
        C (float):
            Total conductance of the neuron - Speed of the differential equation
        I (float):
            Injected Current
        tau (float):
            Time constant for the synaptic filter
        gap_junction (float): 
            Gap junction strength
        v_neurons (float):
            Voltage of the neighbouring neurons

    Returns:
        dydt (tuple[float]):
            The result of integrating the input signal and the synaptic current
    '''

    #Definition of the reversal potntial of the neuron, in this case it is inhibitory
    Vreversal = -80
    vt = -58 

    #HH differential equations (voltage and recovery variables)
    Ina = gna * y[2]**3 * y[3] * (y[0] - Ena)
    Ik = gk * y[1]**4 * (y[0]- Ek)
    dvdt = (-Ina -Ik - gl * (y[0] - El) + I - gap_junction * np.sum( (y[0] - v_neurons)) -y[4] * (y[0] - Vreversal)) / C 
    dmdt = am_pairs(y[0],vt) * (1-y[2]) - bm_pairs(y[0],vt) * y[2]
    dhdt = ah_pairs(y[0],vt) * (1-y[3]) - bh_pairs(y[0],vt) * y[3]
    dndt = an_pairs(y[0],vt) * (1-y[1]) - bn_pairs(y[0],vt) * y[1]

    #Computing the synaptic filtering
    y = np.append(y,0)
    for i in range(4,4+order):
        y[i] =  -y[i] / tau + y[i+1] 

    #Returning the data, making sure it is in the correct (numpy array) format
    dydt = [dvdt,dndt,dmdt,dhdt]
    dydt = np.array(dydt,dtype=object)
    for i in range(4,4+order):
        dydt = np.append(dydt,float(y[i]))

    return dydt

def HH_Equation_Network(y,n,m,h,synaptic,order,gna,gk,gl,Ena,Ek,El,C,I,tau,gap_junction,connectivity_matrix):
    '''
    Algorithm that integrates the equations of the Hodgkin-Huxley model, as well as the synaptic filter.

    Parameters:
        y (tuple[float]):
            The signal of the Neurons - Voltage
        n (tuple[float]):
            The signal of the Neurons - n
        m (tuple[float]):
            The signal of the Neurons - m
        h (tuple[float]):
            The signal of the Neurons - h
        synaptic (tuple[float]):
            The synaptic current of the neurons
        order (int):
            The order of the synaptic filter - Max value of 5
        gna (float):
            Conductance of the Na channel
        gk (float):
            Conductance of the K channel
        gl (float):
            Conductance of the leak channel
        Ena (float):
            Rest voltage of the Na channel
        Ek (float):
            Rest voltage of the K channel
        El (float):
            Rest voltage of the leak channel
        C (float):
            Total conductance of the neuron - Speed of the differential equation
        I (float):
            Injected Current
        tau (float):
            Time constant for the synaptic filter
        gap_junction (float): 
            Gap junction strength
        connectivity_matrix (dok_matrix):
            An sparse connectivity matrix, containing ones if the neurons are connected and 0 if the neurons are not connected

    Returns:
        dydt (tuple[float]):
            The result of integrating the input signal and the synaptic current
    '''
    #Definition of the reversal potntial of the neuron, in this case it is inhibitory
    Vreversal = -80
    vt = -58

    #HH differential equations
    Ina = gna * np.multiply(np.multiply(np.power(m,3),h),(y - Ena))
    Ik = gk * np.multiply(np.power(n,4),(y- Ek))
    I_gap = np.ravel((connectivity_matrix.multiply( np.subtract.outer(y, y))).sum(axis=0))
    dvdt = (-Ina -Ik - gl * (y - El) + I + gap_junction * I_gap - np.multiply(synaptic[0:len(y)],(y- Vreversal)) )/ C 

    dmdt = np.subtract(np.multiply(am_network(y,vt), (1-m)) , np.multiply(bm_network(y,vt), m))
    dhdt = np.subtract(np.multiply(ah_network(y,vt), (1-h)) , np.multiply( bh_netowrk(y,vt),h))
    dndt = np.subtract( np.multiply(an_network(y,vt), (1-n)) , np.multiply(bn_network(y,vt), n))

    for i in range(0,order):
        if i == order -1 :
             synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau
        else:
            synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau + synaptic[(i+1)*len(y):(i+2)*len(y)]

    #Returning the data, making sure it is in the correct (numpy array) format
    dydt = [dvdt,dndt,dmdt,dhdt]
    dydt = np.array(dydt,dtype=object)
    return dydt


def HH_Neuron_Pairs(dt, t_final, order, y0, n0, m0, h0, gna, gk, gl, Ena, Ek, El, C, I, Isyn, gap_junction, tau, return_dict=0):
    ''' 
    Runge-Kutta integration of the 4th order of the HH model in the case of pairs of two neurons or single neurons

    Parameters:
        dt (float):
            Time step of the simulation
        t_final (float):
            Final time of the simulation
        order (int):
            Order of the synaptic filtering (maximum value = 5)
        y0 (float):
            Initial voltage of the system
        n0 (float):
            Initial value of the n recovery variable
        m0 (float):
            Initial value of the m recovery variable
        h0 (float):
            Initial value of the h recovery variable
        gna (float):
            Conductance of the Na channel
        gk (float):
            Conductance of the K channel
        gl (float):
            Conductance of the leak channel
        Ena (float):
            Rest voltage of the Na channel
        Ek (float):
            Rest voltage of the K channel
        El (float):
            Rest voltage of the leak channel
        C (float):
            Total conductance of the neuron - Speed of the differential equation
        I (float):
            Injected Current
        Isyn (float):
            Strength of the chemical synaptse
        gap_junction (float): 
            Gap junction strength
        tau (float):
            Time constant for the synaptic filter
        return_dict (dict or 0, optional ):
            This should be  0 in the case you don't need to parallel processing. In the case it is being used, this dictionary should be manager.dict()
            (see synchrony_measurements.ipynb for examples)

    Returns: 
        data (tuple[tuple[float,float]]):
            The voltage of each of the neurons over time
        Y (tuple[tuple[float,float]]):
            The complete signal of each of the neurons - Voltage, recovery variable, and synaptic current
        matrix (dok_matrix):
            A sparse matrix of the spike times of the simulation
    '''
    
    #Copmuting the number of steps of the simulation, and converting int to np.array for the case of single neurons
    Nsteps = int(t_final/dt)
    if type(y0) is int:
        y0 = [y0]
        n0 = [n0]
        m0 = [m0]
        h0 = [h0]
        I = [I]
    num_neurons = len(y0)

    #Setting the limit of the synaptic filtering order
    if order >5:
        print('The maximum order of t he synaptic filter is 5')
        order = 5

    #Initialisating the variables we need for the simulation 
    matrix = dok_matrix((num_neurons,int(t_final/dt)))
    Y = np.zeros((Nsteps,num_neurons*(4+order)))
    data = np.zeros((Nsteps,num_neurons))
    end = num_neurons * (4+order) -1

    #Setting the initial conditions of the system
    for i in range (0,num_neurons): 
        Y[0,i*(4+order)] = y0[i]
        Y[0,1+i*(4+order)] = n0[i]
        Y[0,2+i*(4+order)] = m0[i]
        Y[0,3+i*(4+order)] = h0[i]
        data[0,i] = y0[i]

    check = np.zeros(num_neurons)

    #Runge-Kutta 4th order loop
    for i in range(0,Nsteps-1):
        for k in range(0,num_neurons):
            k1 = HH_Equation_Pairs(Y[i, k*(4+order): (k+1) * (4+order)], order, gna, gk, gl, Ena, Ek, El, C, I[i,k], tau, gap_junction, Y[i, 0:end:4+order] )
            k2 = HH_Equation_Pairs(Y[i, k*(4+order): (k+1) * (4+order)] + 0.5*dt*k1, order, gna, gk, gl, Ena, Ek, El, C, I[i,k], tau, gap_junction, Y[i, 0:end:4+order ] )
            k3 = HH_Equation_Pairs(Y[i, k*(4+order): (k+1) * (4+order)] + 0.5*dt*k2, order, gna, gk, gl, Ena, Ek, El, C, I[i,k], tau, gap_junction, Y[i, 0:end:4+order ] )
            k4 = HH_Equation_Pairs(Y[i, k*(4+order): (k+1) * (4+order)] + dt * k3, order, gna, gk, gl, Ena, Ek, El, C, I[i,k], tau, gap_junction, Y[i, 0:end:4+order ] )
            
            Y[i + 1, k * (4 + order): (k+1) *(4+order)] = Y[i, k * (4+order): (k+1)*(4+order) ] + 1/6 * dt * (k1 + 2*k2 + 2*k3 + k4)

        #Checking for spikes
        for k in range(0,num_neurons):
            if i > 0 and Y[i,k*(4+order)] > 10 and check[k] == 0:
                matrix[k,i] = 1
                check[k] = 1
                for l in range(0,num_neurons):
                    if l != k:
                        Y[i+1,l*(4+order) + 4 + order-1] = Y[i+1,l * (4+order) +4 + order-1] + Isyn[k,l]
            data[i+1,k] = Y[i+1,k*(4+order)]  
            if data[i+1,k] < 0:
                check[k] = 0

    if return_dict == 0:
        return data, Y, matrix
    else:
        return_dict['data_HH'] = data 
        return_dict['Y_HH'] = Y 
        return_dict['Matrix_HH'] = np.array(matrix.todense())

def HH_Neuron_Network(dt, t_final, order, y0, n0, m0, h0, gna, gk, gl, Ena, Ek, El, C, I, Isyn, gap_junction, tau, E_matrix, C_matrix, return_dict=0):
    ''' 
    Runge-Kutta integration of the 4th order of the HH model in the case of pairs of network of neurons

    Parameters:
        dt (float):
            Time step of the simulation
        t_final (float):
            Final time of the simulation
        order (int):
            Order of the synaptic filtering (maximum value = 5)
        y0 (float):
            Initial voltage of the system
        n0 (float):
            Initial value of the n recovery variable
        m0 (float):
            Initial value of the m recovery variable
        h0 (float):
            Initial value of the h recovery variable
        gna (float):
            Conductance of the Na channel
        gk (float):
            Conductance of the K channel
        gl (float):
            Conductance of the leak channel
        Ena (float):
            Rest voltage of the Na channel
        Ek (float):
            Rest voltage of the K channel
        El (float):
            Rest voltage of the leak channel
        C (float):
            Total conductance of the neuron - Speed of the differential equation
        I (float):
            Injected Current
        Isyn (float):
            Strength of the chemical synaptse
        gap_junction (float): 
            Gap junction strength
        tau (float):
            Time constant for the synaptic filter
        E_matrix (sparse_matrix):
            Connectivity matrix for the electrical synapses, 1 for connection, 0 for not connected
        C_matrix (sparse_matrix):
            Connectivity matrix for the chemical synapses, 1 for connection, 0 for not connected
        return_dict (dict or 0, optional ):
            This should be  0 in the case you don't need to parallel processing. In the case it is being used, this dictionary should be manager.dict()
            (see synchrony_measurements.ipynb for examples)

    Returns: 
        data (tuple[tuple[float,float]]):
            The voltage of each of the neurons over time
        Y (tuple[tuple[float,float]]):
            The complete signal of each of the neurons - Voltage, recovery variables
        matrix (dok_matrix):
            A sparse matrix of the spike times of the simulation
        synaptic (tuple[tuple[[float,float]):
            The synaptic current signal
    '''

    #obtain the number of steps of the simulation
    Nsteps = int(t_final/dt)

    #we are assuming we are working with arrays, so transform everything into one
    if type(y0) is int:
        y0 = [y0]
        n0 = [n0]
        m0 = [m0]
        h0 = [h0]
        I = np.array( [ [I] , [I] ] )

    #compute the number of neurons
    num_neurons = len(y0)

    #we are only allowing a synaptic filtering order up to 5
    if order > 5:
        print('We are changing down the filtering order to the maximum: 5')
        order = 5

    matrix = dok_matrix((num_neurons,int(t_final/dt)))
        
    #variables that store the signal
    Y = np.zeros((Nsteps,num_neurons))
    N = np.zeros((Nsteps,num_neurons))
    M = np.zeros((Nsteps,num_neurons))
    H = np.zeros((Nsteps,num_neurons))
    synaptic = np.zeros((Nsteps,order*num_neurons))

    #assign the initial values
    for i in range (0,num_neurons): 
        Y[0,i] = y0[i]
        N[0,i] = n0[i]
        M[0,i] = m0[i]
        H[0,i] = h0[i]
    check = np.zeros(num_neurons)

    #compute the number of connections of each neuron
    num_connections = np.zeros(num_neurons)
    for i in range(0,num_neurons):
        num_connections[i] = len((np.argwhere(np.array(C_matrix.todense())[i,:]>0) * dt).flatten())

    #Runge-Kutta 4th order method 
    for i in range(0,Nsteps-1):
        k1 = HH_Equation_Network(Y[i,:], N[i,:], M[i,:], H[i,:], synaptic[i,:],order, gna, gk, gl, Ena, Ek, El, C, I[i,:], tau, gap_junction, E_matrix)
        k2 = HH_Equation_Network(np.float64(Y[i,:]+ 0.5*dt*k1[0]),N[i,:] + 0.5*dt*k1[1] , M[i,:] + 0.5*dt*k1[2], H[i,:] + 0.5*dt*k1[3],synaptic[i,:],order, gna, gk, gl, Ena, Ek, El, C, I[i,:], tau, gap_junction, E_matrix )
        k3 = HH_Equation_Network(np.float64(Y[i,:]+ 0.5*dt*k2[0]),N[i,:] + 0.5*dt*k2[1] , M[i,:] + 0.5*dt*k2[2], H[i,:] + 0.5*dt*k2[3],synaptic[i,:],order, gna, gk, gl, Ena, Ek, El, C, I[i,:], tau, gap_junction, E_matrix )            
        k4 = HH_Equation_Network(np.float64(Y[i,:]+dt*k3[0]),N[i,:] + dt*k3[1] , M[i,:] + dt*k3[2], H[i,:] + dt*k3[3],synaptic[i,:],order, gna, gk, gl, Ena, Ek, El, C, I[i,:], tau, gap_junction, E_matrix  )
            
        Y[i + 1, :] = Y[i, :] + 1/6 * dt * (k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        N[i + 1, :] = N[i, :] + 1/6 * dt * (k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        M[i + 1, :] = M[i, :] + 1/6 * dt * (k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        H[i + 1, :] = H[i, :] + 1/6 * dt * (k1[3] + 2*k2[3] + 2*k3[3] + k4[3])

        #searching for spikes (positive local maximums)
        if(i>0):
            spikes =  np.where( (Y[i,:] > 0))
            if len(spikes[0]) > 0:
                for spike_ind in spikes[0]:
                    if check[spike_ind] == 0:
                        check[spike_ind] = 1
                        matrix[spike_ind,i] = 1
                        synaptic[i+1,(order-1)*num_neurons:order*num_neurons] = synaptic[i+1,(order-1)*num_neurons:order*num_neurons] + C_matrix[spike_ind,:] *Isyn / num_connections[spike_ind]
        
            negatives = np.where(Y[i,:]< 0)
            if len(negatives[0]) > 0:
                for index in negatives[0]:
                    check[index] = 0

    if return_dict == 0:
        return  Y, [Y,N,M,H], matrix, synaptic
    else:
        return_dict['data_HH'] = Y 
        return_dict['Y_HH'] = [Y,N,M,H] 
        return_dict['Matrix_HH'] = np.array(matrix.todense())
        return_dict['synaptic_HH'] = synaptic
