import numpy as np 
from scipy.sparse import dok_matrix


def LIF_Equation_Pairs(y,order,gl,El,C,I,tau,gap_junction,v_neurons):
    '''
    Algorithm that integrates the equation of the Leaky Integrate and Fire model, as well as the synaptic filter.

    Parameters:
        y (tuple[float]):
            The signal of the Neuron - Voltage and Synaptic Current.
        order (int):
            The order of the synaptic filter - Max value of 5
        gl (float):
            The conductance of the leak channel
        El (float):
            The resting voltage of the leak channel
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

    #LIF differential equation
    dvdt = (-gl * (y[0] - El) + I - gap_junction * np.sum(y[0] - v_neurons) - y[1]* (y[0] - Vreversal)) / C 

    #Computing the synaptic filtering
    y = np.append(y,0)
    for i in range(1,1+order):
        y[i] =  -y[i] / tau + y[i+1]

    
    #Returning the data, making sure it is in the correct (numpy array) format
    dydt = [dvdt]
    dydt = np.array(dydt,dtype=object)
    for i in range(1,1+order):
        dydt = np.append(dydt,float(y[i]))

    return dydt

def LIF_Equation_Network(y,synaptic,order,gl,El,C,I,tau,gap_junction,connectivity_matrix):
    '''
    Algorithm that integrates the equation of the Leaky Integrate and Fire model, as well as the synaptic filter.

    Parameters:
        y (tuple[float]):
            The signal of the Neurons - Voltage
        synaptic (tuple[float]):
            The synaptic current of the neurons
        order (int):
            The order of the synaptic filter - Max value of 5
        gl (float):
            The conductance of the leak channel
        El (float):
            The resting voltage of the leak channel
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
            The result of integrating the input signal
    '''

    #Definition of the reversal potntial of the neuron, in this case it is inhibitory
    Vreversal = -80

    #LIF differential equation
    I_gap = np.ravel((connectivity_matrix.multiply( np.subtract.outer(y, y))).sum(axis=0))
    dvdt = (-gl * np.subtract(y,El) + I + gap_junction * I_gap - np.multiply(synaptic[0:len(y)],(y- Vreversal)) ) / C  

    for i in range(0,order):
        if i == order -1 :
             synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau
        else:
            synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau + synaptic[(i+1)*len(y):(i+2)*len(y)]

    #Returning the data, making sure it is in the correct (numpy array) format
    dydt = [dvdt]
    dydt = np.array(dydt,dtype=object)
    return dydt


def LIF_Neuron_Pairs(dt,t_final,order,y0,Vth,Vr,Vpeak,gl,El,C,I,Isyn,gap_junction,tau,spikelet,return_dict=0):
    ''' 
    Runge-Kutta integration of the 4th order of the LIF model in the case of pairs of two neurons or single neurons

    Parameters:
        dt (float):
            Time step of the simulation
        t_final (float):
            Final time of the simulation
        order (int):
            Order of the synaptic filtering (maximum value = 5)
        y0 (float):
            Initial conditions of the system
        Vth (float):
            Threshold potential of the neurons
        Vr (float):
            Reset potential of the neurons
        Vpeak (float):
            Voltage of the peak of the spike
        gl (float):
            The conductance of the leak channel
        El (float):
            The resting voltage of the leak channel
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
        spikelet (float):
            Value of the spikelet of the Integrate and fire model
        return_dict (dict or 0, optional ):
            This should be  0 in the case you don't need to parallel processing. In the case it is being used, this dictionary should be manager.dict()
            (see synchrony_measurements.ipynb for examples)

    Returns: 
        data (tuple[tuple[float,float]]):
            The voltage of each of the neurons over time
        Y (tuple[tuple[float,float]]):
            The complete signal of each of the neurons - Voltage and synaptic current
        matrix (dok_matrix):
            A sparse matrix of the spike times of the simulation
    '''

    #Copmuting the number of steps of the simulation, and converting int to np.array for the case of single neurons
    Nsteps = int(t_final/dt)
    if type(y0) is int:
        y0 = [y0]
        I = [I]
    num_neurons =len(y0)

    #Setting the limit of the synaptic filtering order
    if order >5:
        print('The maximum order of t he synaptic filter is 5')
        order = 5
    
    #Initialisating the variables we need for the simulation
    Y = np.zeros( (Nsteps, num_neurons * (1 + order)))
    matrix = dok_matrix((num_neurons,int(t_final/dt)))
    data = np.zeros((Nsteps, num_neurons))
    data[0,:] = y0 
    end = num_neurons * (1 + order) -1

    #Setting the initial conditions of the system
    for i in range(0,num_neurons):
        Y[0, i * (1+order)] = y0[i]
    
    #Runge-Kutta 4th order loop
    for i in range(0,Nsteps-1):
        for k in range(0,num_neurons):
            k1 = LIF_Equation_Pairs( Y[i,k*(1+order):(k+1)*(1+order)] ,order,gl,El,C,I[i,k],tau,gap_junction,Y[i,0:end:1+order])
            k2 = LIF_Equation_Pairs( Y[i,k*(1+order):(k+1)*(1+order)] +0.5 * dt * k1,order,gl,El,C,I[i,k],tau,gap_junction,Y[i,0:end:1+order])
            k3 = LIF_Equation_Pairs( Y[i,k*(1+order):(k+1)*(1+order)] +0.5 * dt * k2,order,gl,El,C,I[i,k],tau,gap_junction,Y[i,0:end:1+order])
            k4 = LIF_Equation_Pairs( Y[i,k*(1+order):(k+1)*(1+order)] + dt * k3,order,gl,El,C,I[i,k],tau,gap_junction,Y[i,0:end:1+order])

            Y[i+1,k*(1+order):(k+1)*(1+order)] = Y[i,k*(1+order):(k+1)*(1+order)] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        #Checking for spikes, if it does reset the voltage of the neuron that spiked, and send a spikelet to their neighbour
        for k in range(0,num_neurons):
            if Y[i+1,k * (1+order)] >= Vth:
                data[i+1,k] = Vpeak
                Y[i+1, k * (1+order)] = Vr 
                matrix[k,i] = 1
                for l in range(0,num_neurons):
                    if l != k:
                        Y[i+1,l * (1+order) +order] = Y[i+1,l*(1+order) + order] + Isyn[k,l]
                        if gap_junction !=0:
                            Y[i+1,l * (1+order)] = Y[i+1,l *(1+order)] + spikelet
            else:
                data[i+1,k] = Y[i+1,k *(1+order)]
        #Reset once again the voltage of the neurons that spiked
        # TODO
        for k in range(0,num_neurons):
            if Y[i+1,k * (1+order)] >= Vth:
                Y[i+1, k * (1+order)] = Vr 
            
    if return_dict == 0:
        return data, Y, matrix
    else:
        return_dict['data_IF'] = data 
        return_dict['Y_IF'] = Y 
        return_dict['Matrix_IF'] = np.array(matrix.todense())

def LIF_Neuron_Network(dt,t_final,order,y0,Vth,Vr,Vpeak,gl,El,C,I,Isyn,gap_junction,tau,spikelet,E_matrix,C_matrix,return_dict=0):
    ''' 
    Runge-Kutta integration of the 4th order of the LIF model in the case of neuron networks
    Parameters:
        dt (float):
            Time step of the simulation
        t_final (float):
            Final time of the simulation
        order (int):
            Order of the synaptic filtering (maximum value = 5)
        y0 (float):
            Initial conditions of the system
        Vth (float):
            Threshold potential of the neurons
        Vr (float):
            Reset potential of the neurons
        Vpeak (float):
            Voltage of the peak of the spike
        gl (float):
            The conductance of the leak channel
        El (float):
            The resting voltage of the leak channel
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
        spikelet (float):
            Value of the spikelet of the Integrate and fire model
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
            The complete signal of each of the neurons - Voltage
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
    check_aux = check

    #compute the number of connections of each neuron
    num_connections = np.zeros(num_neurons)
    for i in range(0,num_neurons):
        num_connections[i] = len((np.argwhere(np.array(C_matrix.todense())[i,:]>0) * dt).flatten())
    
    #Runge-Kutta 4th order method 
    for i in range(0,Nsteps-1):
        k1 = LIF_Equation_Network( Y[i,:] ,synaptic[i,:],order,gl,El,C,I[i,:],tau,gap_junction,E_matrix)
        k2 = LIF_Equation_Network( np.float64(Y[i,:] +0.5 * dt * k1[0]),synaptic[i,:],order,gl,El,C,I[i,:],tau,gap_junction,E_matrix)
        k3 = LIF_Equation_Network( np.float64(Y[i,:] +0.5 * dt * k2[0]),synaptic[i,:],order,gl,El,C,I[i,:],tau,gap_junction,E_matrix)
        k4 = LIF_Equation_Network( np.float64(Y[i,:] + dt * k3[0]) ,synaptic[i,:],order,gl,El,C,I[i,:],tau,gap_junction,E_matrix)

        Y[i+1,:] = Y[i,:] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        if i > 0:
            spikes = np.where( Y[i+1,:] >= Vth)
            if len(spikes[0]) > 0:
                for spike_ind in spikes[0]:
                    if check[spike_ind] == 0:
                        check_aux[spike_ind] = 1
                        matrix[spike_ind,i] = 1
                        data[i+1,:] = Y[i+1,:]
                        data[i+1,spike_ind] = Vpeak
                        Y[i+1,spike_ind] = Vr 
                        synaptic[i+1,(order-1)*num_neurons:order*num_neurons] = synaptic[i+1,(order-1)*num_neurons:order*num_neurons] + C_matrix[spike_ind,:] *Isyn / num_connections[spike_ind]
                        Y[i+1,:] = Y[i+1,:] +  E_matrix[spike_ind,:] *spikelet 
                #for spike_ind in spikes[0]:
                #    if check[spike_ind] == 0 :
                #        Y[i+1,spike_ind] = Vr
            else:
                data[i+1,:] = Y[i+1,:]
        else:
            data[i+1,:] = Y[i+1,:]

        negatives = np.where(Y[i,:]< 0)
        if len(negatives[0]) > 0:
            for index in negatives[0]:
                check_aux[index] = 0 
        check = check_aux

    if return_dict == 0:
        return data, Y, matrix, synaptic
    else:
        return_dict['data_IF'] = data 
        return_dict['Y_IF'] = Y 
        return_dict['Matrix_IF'] = np.array(matrix.todense())
        return_dict['synaptic_IF'] = synaptic

def LIF_Equation_Network_tests(y,synaptic,order,gl,El,C,I,tau,gap_junction,connectivity_matrix,gap_current,synaptic_current):
    '''
    Algorithm that integrates the equation of the Leaky Integrate and Fire model, as well as the synaptic filter.

    Parameters:
        y (tuple[float]):
            The signal of the Neurons - Voltage
        synaptic (tuple[float]):
            The synaptic current of the neurons
        order (int):
            The order of the synaptic filter - Max value of 5
        gl (float):
            The conductance of the leak channel
        El (float):
            The resting voltage of the leak channel
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
            The result of integrating the input signal
    '''

    #Definition of the reversal potntial of the neuron, in this case it is inhibitory
    Vreversal = -80

    #LIF differential equation
    I_gap = np.ravel((connectivity_matrix.multiply( np.subtract.outer(y, y))).sum(axis=0))
    gap_current[:] = I_gap
    synaptic_current[:] = np.multiply(synaptic[0:len(y)],(y- Vreversal))
    dvdt = (-gl * np.subtract(y,El) + I + gap_junction * I_gap - np.multiply(synaptic[0:len(y)],(y- Vreversal)) ) / C  

    for i in range(0,order):
        if i == order -1 :
             synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau
        else:
            synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau + synaptic[(i+1)*len(y):(i+2)*len(y)]

    #Returning the data, making sure it is in the correct (numpy array) format
    dydt = [dvdt]
    dydt = np.array(dydt,dtype=object)
    return dydt

def LIF_Neuron_Network_tests(dt,t_final,order,y0,Vth,Vr,Vpeak,gl,El,C,I,Isyn,gap_junction,tau,spikelet,E_matrix,C_matrix,gap_current,synaptic_current,return_dict=0):
    ''' 
    Runge-Kutta integration of the 4th order of the LIF model in the case of neuron networks
    Parameters:
        dt (float):
            Time step of the simulation
        t_final (float):
            Final time of the simulation
        order (int):
            Order of the synaptic filtering (maximum value = 5)
        y0 (float):
            Initial conditions of the system
        Vth (float):
            Threshold potential of the neurons
        Vr (float):
            Reset potential of the neurons
        Vpeak (float):
            Voltage of the peak of the spike
        gl (float):
            The conductance of the leak channel
        El (float):
            The resting voltage of the leak channel
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
        spikelet (float):
            Value of the spikelet of the Integrate and fire model
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
            The complete signal of each of the neurons - Voltage
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
    check_aux = check

    #compute the number of connections of each neuron
    num_connections = np.zeros(num_neurons)
    for i in range(0,num_neurons):
        num_connections[i] = len((np.argwhere(np.array(C_matrix.todense())[i,:]>0) * dt).flatten())
    
    #Runge-Kutta 4th order method 
    for i in range(0,Nsteps-1):
        k1 = LIF_Equation_Network_tests( Y[i,:] ,synaptic[i,:],order,gl,El,C,I[i,:],tau,gap_junction,E_matrix,gap_current[i,:],synaptic_current[i,:])
        k2 = LIF_Equation_Network_tests( np.float64(Y[i,:] +0.5 * dt * k1[0]),synaptic[i,:],order,gl,El,C,I[i,:],tau,gap_junction,E_matrix,gap_current[i,:],synaptic_current[i,:])
        k3 = LIF_Equation_Network_tests( np.float64(Y[i,:] +0.5 * dt * k2[0]),synaptic[i,:],order,gl,El,C,I[i,:],tau,gap_junction,E_matrix, gap_current[i,:],synaptic_current[i,:])
        k4 = LIF_Equation_Network_tests( np.float64(Y[i,:] + dt * k3[0]) ,synaptic[i,:],order,gl,El,C,I[i,:],tau,gap_junction,E_matrix,gap_current[i,:],synaptic_current[i,:])

        Y[i+1,:] = Y[i,:] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        if i > 0:
            spikes = np.where( Y[i+1,:] >= Vth)
            if len(spikes[0]) > 0:
                for spike_ind in spikes[0]:
                    if check[spike_ind] == 0:
                        check_aux[spike_ind] = 1
                        matrix[spike_ind,i] = 1
                        data[i+1,:] = Y[i+1,:]
                        data[i+1,spike_ind] = Vpeak
                        Y[i+1,spike_ind] = Vr 
                        synaptic[i+1,(order-1)*num_neurons:order*num_neurons] = synaptic[i+1,(order-1)*num_neurons:order*num_neurons] + C_matrix[spike_ind,:] *Isyn / num_connections[spike_ind]
                        Y[i+1,:] = Y[i+1,:] +  E_matrix[spike_ind,:] *spikelet 
                #for spike_ind in spikes[0]:
                #    if check[spike_ind] == 0 :
                #        Y[i+1,spike_ind] = Vr
            else:
                data[i+1,:] = Y[i+1,:]
        else:
            data[i+1,:] = Y[i+1,:]

        negatives = np.where(Y[i,:]< 0)
        if len(negatives[0]) > 0:
            for index in negatives[0]:
                check_aux[index] = 0 
        check = check_aux

    if return_dict == 0:
        return data, Y, matrix, synaptic
    else:
        return_dict['data_IF'] = data 
        return_dict['Y_IF'] = Y 
        return_dict['Matrix_IF'] = np.array(matrix.todense())
        return_dict['synaptic_IF'] = synaptic

def LIF_Equation_Pairs_tests(y,order,gl,El,C,I,tau,gap_junction,v_neurons,gap_current,synaptic_current):
    '''
    Algorithm that integrates the equation of the Leaky Integrate and Fire model, as well as the synaptic filter.

    Parameters:
        y (tuple[float]):
            The signal of the Neuron - Voltage and Synaptic Current.
        order (int):
            The order of the synaptic filter - Max value of 5
        gl (float):
            The conductance of the leak channel
        El (float):
            The resting voltage of the leak channel
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

    #LIF differential equation
    dvdt = (-gl * (y[0] - El) + I - gap_junction * np.sum(y[0] - v_neurons) - y[1]* (y[0] - Vreversal)) / C 
    gap_current[:] = gap_junction * np.sum(y[0] - v_neurons)
    synaptic_current[:] =  y[1]* (y[0] - Vreversal)

    #Computing the synaptic filtering
    y = np.append(y,0)
    for i in range(1,1+order):
        y[i] =  -y[i] / tau + y[i+1]

    
    #Returning the data, making sure it is in the correct (numpy array) format
    dydt = [dvdt]
    dydt = np.array(dydt,dtype=object)
    for i in range(1,1+order):
        dydt = np.append(dydt,float(y[i]))

    return dydt

def LIF_Neuron_Pairs_tests(dt,t_final,order,y0,Vth,Vr,Vpeak,gl,El,C,I,Isyn,gap_junction,tau,spikelet,gap_current,synaptic_current,return_dict=0):
    ''' 
    Runge-Kutta integration of the 4th order of the LIF model in the case of pairs of two neurons or single neurons

    Parameters:
        dt (float):
            Time step of the simulation
        t_final (float):
            Final time of the simulation
        order (int):
            Order of the synaptic filtering (maximum value = 5)
        y0 (float):
            Initial conditions of the system
        Vth (float):
            Threshold potential of the neurons
        Vr (float):
            Reset potential of the neurons
        Vpeak (float):
            Voltage of the peak of the spike
        gl (float):
            The conductance of the leak channel
        El (float):
            The resting voltage of the leak channel
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
        spikelet (float):
            Value of the spikelet of the Integrate and fire model
        return_dict (dict or 0, optional ):
            This should be  0 in the case you don't need to parallel processing. In the case it is being used, this dictionary should be manager.dict()
            (see synchrony_measurements.ipynb for examples)

    Returns: 
        data (tuple[tuple[float,float]]):
            The voltage of each of the neurons over time
        Y (tuple[tuple[float,float]]):
            The complete signal of each of the neurons - Voltage and synaptic current
        matrix (dok_matrix):
            A sparse matrix of the spike times of the simulation
    '''

    #Copmuting the number of steps of the simulation, and converting int to np.array for the case of single neurons
    Nsteps = int(t_final/dt)
    if type(y0) is int:
        y0 = [y0]
        I = [I]
    num_neurons =len(y0)

    #Setting the limit of the synaptic filtering order
    if order >5:
        print('The maximum order of t he synaptic filter is 5')
        order = 5
    
    #Initialisating the variables we need for the simulation
    Y = np.zeros( (Nsteps, num_neurons * (1 + order)))
    matrix = dok_matrix((num_neurons,int(t_final/dt)))
    data = np.zeros((Nsteps, num_neurons))
    data[0,:] = y0 
    end = num_neurons * (1 + order) -1

    #Setting the initial conditions of the system
    for i in range(0,num_neurons):
        Y[0, i * (1+order)] = y0[i]
    
    #Runge-Kutta 4th order loop
    for i in range(0,Nsteps-1):
        for k in range(0,num_neurons):
            k1 = LIF_Equation_Pairs_tests( Y[i,k*(1+order):(k+1)*(1+order)] ,order,gl,El,C,I[i,k],tau,gap_junction,Y[i,0:end:1+order],gap_current[i,k],synaptic_current[i,k])
            k2 = LIF_Equation_Pairs_tests( Y[i,k*(1+order):(k+1)*(1+order)] +0.5 * dt * k1,order,gl,El,C,I[i,k],tau,gap_junction,Y[i,0:end:1+order],gap_current[i,k], synaptic_current[i,k])
            k3 = LIF_Equation_Pairs_tests( Y[i,k*(1+order):(k+1)*(1+order)] +0.5 * dt * k2,order,gl,El,C,I[i,k],tau,gap_junction,Y[i,0:end:1+order],gap_current[i,k],synaptic_current[i,k])
            k4 = LIF_Equation_Pairs_tests( Y[i,k*(1+order):(k+1)*(1+order)] + dt * k3,order,gl,El,C,I[i,k],tau,gap_junction,Y[i,0:end:1+order],gap_current[i,k],synaptic_current[i,k])

            Y[i+1,k*(1+order):(k+1)*(1+order)] = Y[i,k*(1+order):(k+1)*(1+order)] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        #Checking for spikes, if it does reset the voltage of the neuron that spiked, and send a spikelet to their neighbour
        for k in range(0,num_neurons):
            if Y[i+1,k * (1+order)] >= Vth:
                data[i+1,k] = Vpeak
                Y[i+1, k * (1+order)] = Vr 
                matrix[k,i] = 1
                for l in range(0,num_neurons):
                    if l != k:
                        Y[i+1,l * (1+order) +order] = Y[i+1,l*(1+order) + order] + Isyn[k,l]
                        if gap_junction !=0:
                            Y[i+1,l * (1+order)] = Y[i+1,l *(1+order)] + spikelet
            else:
                data[i+1,k] = Y[i+1,k *(1+order)]
        #Reset once again the voltage of the neurons that spiked
        for k in range(0,num_neurons):
            if Y[i+1,k * (1+order)] >= Vth:
                Y[i+1, k * (1+order)] = Vr 
            
    if return_dict == 0:
        return data, Y, matrix
    else:
        return_dict['data_IF'] = data 
        return_dict['Y_IF'] = Y 
        return_dict['Matrix_IF'] = np.array(matrix.todense())