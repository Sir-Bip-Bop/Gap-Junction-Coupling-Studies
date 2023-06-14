import numpy as np 
from scipy.sparse import dok_matrix

def ML_Equation_Pairs(y, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I, tau, gap_junction, v_neurons ):
    '''
    Algorithm that integrates the equations of the modified Morris-Lecar model, as well as the synaptic filter.

    Parameters:
        y (tuple[float]):
            The signal of the Neuron - Voltage,  recovery variable, and Synaptic Current.
        order (int):
            The order of the synaptic filter - Max value of 5
        psi (float, optional):
            Scaling constant
        V1 (float, optinoal):
            Potential at wich m_inf = 0.5
        V2 (float, optional):
            Reciprocal of the voltage dependance of m_inf
        V3 (float, optional):
            Potential at wich w_inf = 0.5
        V4 (float, optional):
            Reciprocal of the voltage dependance of w_inf
        gna (float, optional):
            Conductance of the Na channel
        gk (float, optional):
            Conductance of the K channel
        gshunt (float, optional):
            Conductance of the Shunt channel
        Ena (float, optional):
            Resting potential of the Na channel
        Ek (float, optional):
            Resting potential of the K channel
        Eshunt (float, optinoal):
            Resting potential of the shunt channel
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

    #HH differential equations (voltage and recovery variable)
    minf = 0.5 * (1 + np.tanh( ((y[0] - V1 )/ V2)))
    Iion = gna * minf * (y[0] - Ena) + gk * y[1] * (y[0] -Ek) + gshunt * (y[0] - Eshunt)
    dvdt = ( - Iion - gap_junction * np.sum(y[0] - v_neurons) + I - y[2] * (y[0]- Vreversal)) / C
    winf = 0.5 * (1 + np.tanh( (y[0] - V3) / V4))
    dwdt = psi * (winf - y[1])*np.cosh( (y[0] - V3) / 2 / V4)

    #Computing the synaptic filtering
    y = np.append(y,0)
    for i in range(2, 2+order):
        y[i] = -y[i] / tau + y[i+1]

    #Returning the data, making sure it is in the correct (numpy array) format
    dydt = [dvdt,dwdt]
    dydt = np.array(dydt,dtype=object)
    for i in range(2, 2+order):
        dydt = np.append(dydt,float(y[i]))
    return dydt

def ML_Equation_Network(y, w, synaptic, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I, tau, gap_junction,connectivity_matrix ):
    '''
    Algorithm that integrates the equations of the modified Morris-Lecar model, as well as the synaptic filter.

    Parameters:
        y (tuple[float]):
            The signal of the Neurons - Voltage
        w (tuple[float]):
            The signal of the Neurons - w
        synaptic (tuple[float]):
            The synaptic current of the Neurons
        order (int):
            The order of the synaptic filter - Max value of 5
        psi (float, optional):
            Scaling constant
        V1 (float, optinoal):
            Potential at wich m_inf = 0.5
        V2 (float, optional):
            Reciprocal of the voltage dependance of m_inf
        V3 (float, optional):
            Potential at wich w_inf = 0.5
        V4 (float, optional):
            Reciprocal of the voltage dependance of w_inf
        gna (float, optional):
            Conductance of the Na channel
        gk (float, optional):
            Conductance of the K channel
        gshunt (float, optional):
            Conductance of the Shunt channel
        Ena (float, optional):
            Resting potential of the Na channel
        Ek (float, optional):
            Resting potential of the K channel
        Eshunt (float, optinoal):
            Resting potential of the shunt channel
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

    #ML differential equations
    minf = 0.5 * (1 + np.tanh( ((y - V1 )/ V2)))
    Iion = gna * minf * (y - Ena) + gk * w * (y -Ek) + gshunt * (y - Eshunt)
    dvdt = ( - Iion + gap_junction * np.ravel((connectivity_matrix.multiply( np.subtract.outer(y, y))).sum(axis=0)) + I - np.multiply(synaptic[0:len(y)],(y- Vreversal)) )/  C
    winf = 0.5 * (1 + np.tanh( (y - V3) / V4))
    dwdt = psi * np.multiply((winf - w),np.cosh( (y - V3) / 2 / V4))

    for i in range(0,order):
        if i == order -1 :
             synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau
        else:
            synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau + synaptic[(i+1)*len(y):(i+2)*len(y)]

    #Returning the data, making sure it is in the correct (numpy array) format
    dydt = [dvdt,dwdt]
    dydt = np.array(dydt,dtype=object)

    return dydt

def ML_Neuron_Pairs(dt,t_final,order,y0,w0,psi,V1,V2,V3,V4,gna,gk,gshunt,Ena,Ek,Eshunt,C,I,Isyn,gap_junction,tau,return_dict=0):
    ''' 
    Runge-Kutta integration of the 4th order of the ML model in the case of pairs of two neurons or single neurons

    Parameters:
        dt (float):
            Time step of the simulation
        t_final (float):
            Final time of the simulation
        order (int):
            Order of the synaptic filtering (maximum value = 5)
        y0 (float):
            Initial voltage of the system
        w0 (float):
            Initial value of the recovery variable
        psi (float, optional):
            Scaling constant
        V1 (float, optinoal):
            Potential at wich m_inf = 0.5
        V2 (float, optional):
            Reciprocal of the voltage dependance of m_inf
        V3 (float, optional):
            Potential at wich w_inf = 0.5
        V4 (float, optional):
            Reciprocal of the voltage dependance of w_inf
        gna (float, optional):
            Conductance of the Na channel
        gk (float, optional):
            Conductance of the K channel
        gshunt (float, optional):
            Conductance of the Shunt channel
        Ena (float, optional):
            Resting potential of the Na channel
        Ek (float, optional):
            Resting potential of the K channel
        Eshunt (float, optinoal):
            Resting potential of the shunt channel
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
        I = np.array( [ [I], [I] ])
    num_neurons = len(y0)
    check = np.zeros(num_neurons)
    
    #Setting the limit of the synaptic filtering order
    if order >5:
        print('The maximum order of t he synaptic filter is 5')
        order = 5

    #Initialisating the variables we need for the simulation 
    matrix = dok_matrix((num_neurons,int(t_final/dt)))
    Y = np.zeros((Nsteps, num_neurons * (2+order)))
    data = np.zeros((Nsteps, num_neurons))
    end = num_neurons * (2+order) -1

    #Setting the initial conditions of the system
    for i in range (0,num_neurons):
        Y[0,i*(2+order)] = y0[i]
        Y[0,1 + i*(2+order)] = w0[i]
        data[0,i] = y0[i]

    #Runge-Kutta 4th order loop
    for i in range(0,Nsteps - 1):
        for k in range(0,num_neurons):
            k1 = ML_Equation_Pairs(Y[i, k*(2+order): (k+1) * (2+order)], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, gap_junction, Y[i, 0:end:2+order] )
            k2 = ML_Equation_Pairs(Y[i, k*(2+order): (k+1) * (2+order)] + 0.5*dt*k1, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, gap_junction, Y[i, 0:end:2+order] )
            k3 = ML_Equation_Pairs(Y[i, k*(2+order): (k+1) * (2+order)] + 0.5*dt*k2, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, gap_junction, Y[i, 0:end:2+order])
            k4 = ML_Equation_Pairs(Y[i, k*(2+order): (k+1) * (2+order)] + dt * k3, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, gap_junction, Y[i, 0:end:2+order] )

            Y[i + 1, k * (2 + order): (k+1) *(2+order)] = Y[i, k * (2+order): (k+1)*(2+order) ] + 1/6 * dt * (k1 + 2*k2 + 2*k3 + k4)

        #Checking for spikes
        for k in range(0,num_neurons):
            if i > 0 and Y[i,k*(2+order)] > 10 and check[k] == 0:
                matrix[k,i] = 1
                check[k] = 1
                for l in range(0,num_neurons):
                    if l != k:
                        Y[i+1,l*(2+order) + 2 + order-1] = Y[i+1,l * (2+order) +2 + order-1] + Isyn[k,l]
            data[i+1,k] = Y[i+1,k*(2+order)]  
            if data[i+1,k] < 0:
                check[k] = 0

    if return_dict == 0:
        return data, Y, matrix
    else:
        return_dict['data_ML'] = data 
        return_dict['Y_ML'] = Y 
        return_dict['Matrix_ML'] = np.array(matrix.todense())

def ML_Neuron_Network(dt,t_final,order,y0,w0,psi,V1,V2,V3,V4,gna,gk,gshunt,Ena,Ek,Eshunt,C,I,Isyn,gap_junction,tau,E_matrix,C_matrix,return_dict=0):
    ''' 
    Runge-Kutta integration of the 4th order of the ML model in the case of pairs of two neurons or single neurons

    Parameters:
        dt (float):
            Time step of the simulation
        t_final (float):
            Final time of the simulation
        order (int):
            Order of the synaptic filtering (maximum value = 5)
        y0 (float):
            Initial voltage of the system
        w0 (float):
            Initial value of the recovery variable
        psi (float, optional):
            Scaling constant
        V1 (float, optinoal):
            Potential at wich m_inf = 0.5
        V2 (float, optional):
            Reciprocal of the voltage dependance of m_inf
        V3 (float, optional):
            Potential at wich w_inf = 0.5
        V4 (float, optional):
            Reciprocal of the voltage dependance of w_inf
        gna (float, optional):
            Conductance of the Na channel
        gk (float, optional):
            Conductance of the K channel
        gshunt (float, optional):
            Conductance of the Shunt channel
        Ena (float, optional):
            Resting potential of the Na channel
        Ek (float, optional):
            Resting potential of the K channel
        Eshunt (float, optinoal):
            Resting potential of the shunt channel
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
            The complete signal of each of the neurons - Voltage, recovery variable, and synaptic current
        matrix (dok_matrix):
            A sparse matrix of the spike times of the simulation
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

    matrix = dok_matrix((num_neurons,int(t_final/dt)))

    #variables that store the signal
    Y = np.zeros((Nsteps, num_neurons))
    W = np.zeros((Nsteps, num_neurons))
    synaptic = np.zeros((Nsteps, order*num_neurons))

    #assign the initial values
    for i in range (0,num_neurons):
        Y[0,i] = y0[i]
        W[0,i] = w0[i]

    check = np.zeros(num_neurons)

    #compute the number of connections of each neuron
    num_connections = np.zeros(num_neurons)
    for i in range(0,num_neurons):
        num_connections[i] = len((np.argwhere(np.array(C_matrix.todense())[i,:]>0) * dt).flatten())

    #Runge-Kutta 4th order method 
    for i in range(0,Nsteps - 1):
        k1 = ML_Equation_Network(Y[i,:],W[i,:],synaptic[i,:], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,:], tau, gap_junction,E_matrix )
        k2 = ML_Equation_Network(np.float64(Y[i, :] + 0.5*dt*k1[0]),W[i,:]+0.5*dt*k1[1],synaptic[i,:], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,:], tau, gap_junction,E_matrix)
        k3 = ML_Equation_Network(np.float64(Y[i,:] + 0.5*dt*k2[0]), W[i,:]+0.5*dt*k1[1],synaptic[i,:], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,:], tau, gap_junction,E_matrix )
        k4 = ML_Equation_Network(np.float64(Y[i,:] + dt * k3[0]), W[i,:]+0.5*dt*k3[1],synaptic[i,:], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,:], tau, gap_junction,E_matrix)
            
        Y[i + 1, :] = Y[i,:] + 1/6 * dt * (k1[0] + 2*k2[0] + 2*k3[0]+ k4[0])
        W[i + 1, :] = W[i,:] + 1/6 * dt * (k1[1] + 2*k2[1] + 2*k3[1]+ k4[1])


        if i > 0:
            spikes =  np.where( (Y[i, :] >= Y [i-1,:]) & (Y[i,:] >= Y[i+1,:]) & (Y[i,:] > 0))
            if len(spikes[0]) > 0:
                for spike_ind in spikes[0]:
                        if check[spike_ind] == 0:
                            check[spike_ind] = 1
                            matrix[spike_ind, i] = 1
                            synaptic[i+1,(order-1)*num_neurons:order*num_neurons] = synaptic[i+1,(order-1)*num_neurons:order*num_neurons] + C_matrix[spike_ind,:] *Isyn / num_connections[spike_ind]

            negatives = np.where(Y[i,:]< 0)
            if len(negatives[0]) > 0:
                for index in negatives[0]:
                    check[index] = 0


    if return_dict == 0:
        return  Y, [Y,W], matrix, synaptic
    else:
        return_dict['data_ML'] = Y 
        return_dict['Y_ML'] = [Y,W] 
        return_dict['Matrix_ML'] = np.array(matrix.todense())
        return_dict['synaptic_ML'] = synaptic

def ML_Equation_Network_tests(y, w, synaptic, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I, tau, gap_junction,connectivity_matrix,gap_current,synaptic_current ):
    '''
    Algorithm that integrates the equations of the modified Morris-Lecar model, as well as the synaptic filter.

    Parameters:
        y (tuple[float]):
            The signal of the Neurons - Voltage
        w (tuple[float]):
            The signal of the Neurons - w
        synaptic (tuple[float]):
            The synaptic current of the Neurons
        order (int):
            The order of the synaptic filter - Max value of 5
        psi (float, optional):
            Scaling constant
        V1 (float, optinoal):
            Potential at wich m_inf = 0.5
        V2 (float, optional):
            Reciprocal of the voltage dependance of m_inf
        V3 (float, optional):
            Potential at wich w_inf = 0.5
        V4 (float, optional):
            Reciprocal of the voltage dependance of w_inf
        gna (float, optional):
            Conductance of the Na channel
        gk (float, optional):
            Conductance of the K channel
        gshunt (float, optional):
            Conductance of the Shunt channel
        Ena (float, optional):
            Resting potential of the Na channel
        Ek (float, optional):
            Resting potential of the K channel
        Eshunt (float, optinoal):
            Resting potential of the shunt channel
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

    #ML differential equations
    gap_current[:] = gap_junction * np.ravel((connectivity_matrix.multiply( np.subtract.outer(y, y))).sum(axis=0))
    synaptic_current[:] = np.multiply(synaptic[0:len(y)],(y- Vreversal))
    minf = 0.5 * (1 + np.tanh( ((y - V1 )/ V2)))
    Iion = gna * minf * (y - Ena) + gk * w * (y -Ek) + gshunt * (y - Eshunt)
    dvdt = ( - Iion + gap_junction * np.ravel((connectivity_matrix.multiply( np.subtract.outer(y, y))).sum(axis=0)) + I - np.multiply(synaptic[0:len(y)],(y- Vreversal)) )/  C
    winf = 0.5 * (1 + np.tanh( (y - V3) / V4))
    dwdt = psi * np.multiply((winf - w),np.cosh( (y - V3) / 2 / V4))

    for i in range(0,order):
        if i == order -1 :
             synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau
        else:
            synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau + synaptic[(i+1)*len(y):(i+2)*len(y)]

    #Returning the data, making sure it is in the correct (numpy array) format
    dydt = [dvdt,dwdt]
    dydt = np.array(dydt,dtype=object)

    return dydt

def ML_Neuron_Network_tests(dt,t_final,order,y0,w0,psi,V1,V2,V3,V4,gna,gk,gshunt,Ena,Ek,Eshunt,C,I,Isyn,gap_junction,tau,E_matrix,C_matrix,gap_current,synaptic_current,return_dict=0):
    ''' 
    Runge-Kutta integration of the 4th order of the ML model in the case of pairs of two neurons or single neurons

    Parameters:
        dt (float):
            Time step of the simulation
        t_final (float):
            Final time of the simulation
        order (int):
            Order of the synaptic filtering (maximum value = 5)
        y0 (float):
            Initial voltage of the system
        w0 (float):
            Initial value of the recovery variable
        psi (float, optional):
            Scaling constant
        V1 (float, optinoal):
            Potential at wich m_inf = 0.5
        V2 (float, optional):
            Reciprocal of the voltage dependance of m_inf
        V3 (float, optional):
            Potential at wich w_inf = 0.5
        V4 (float, optional):
            Reciprocal of the voltage dependance of w_inf
        gna (float, optional):
            Conductance of the Na channel
        gk (float, optional):
            Conductance of the K channel
        gshunt (float, optional):
            Conductance of the Shunt channel
        Ena (float, optional):
            Resting potential of the Na channel
        Ek (float, optional):
            Resting potential of the K channel
        Eshunt (float, optinoal):
            Resting potential of the shunt channel
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
            The complete signal of each of the neurons - Voltage, recovery variable, and synaptic current
        matrix (dok_matrix):
            A sparse matrix of the spike times of the simulation
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

    matrix = dok_matrix((num_neurons,int(t_final/dt)))

    #variables that store the signal
    Y = np.zeros((Nsteps, num_neurons))
    W = np.zeros((Nsteps, num_neurons))
    synaptic = np.zeros((Nsteps, order*num_neurons))

    #assign the initial values
    for i in range (0,num_neurons):
        Y[0,i] = y0[i]
        W[0,i] = w0[i]

    check = np.zeros(num_neurons)

    #compute the number of connections of each neuron
    num_connections = np.zeros(num_neurons)
    for i in range(0,num_neurons):
        num_connections[i] = len((np.argwhere(np.array(C_matrix.todense())[i,:]>0) * dt).flatten())

    #Runge-Kutta 4th order method 
    for i in range(0,Nsteps - 1):
        k1 = ML_Equation_Network_tests(Y[i,:],W[i,:],synaptic[i,:], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,:], tau, gap_junction,E_matrix , gap_current[i,:], synaptic_current[i,:])
        k2 = ML_Equation_Network_tests(np.float64(Y[i, :] + 0.5*dt*k1[0]),W[i,:]+0.5*dt*k1[1],synaptic[i,:], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,:], tau, gap_junction,E_matrix, gap_current[i,:], synaptic_current[i,:])
        k3 = ML_Equation_Network_tests(np.float64(Y[i,:] + 0.5*dt*k2[0]), W[i,:]+0.5*dt*k1[1],synaptic[i,:], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,:], tau, gap_junction,E_matrix, gap_current[i,:], synaptic_current[i,:])
        k4 = ML_Equation_Network_tests(np.float64(Y[i,:] + dt * k3[0]), W[i,:]+0.5*dt*k3[1],synaptic[i,:], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,:], tau, gap_junction,E_matrix, gap_current[i,:], synaptic_current[i,:])
            
        Y[i + 1, :] = Y[i,:] + 1/6 * dt * (k1[0] + 2*k2[0] + 2*k3[0]+ k4[0])
        W[i + 1, :] = W[i,:] + 1/6 * dt * (k1[1] + 2*k2[1] + 2*k3[1]+ k4[1])


        if i > 0:
            spikes =  np.where( (Y[i, :] >= Y [i-1,:]) & (Y[i,:] >= Y[i+1,:]) & (Y[i,:] > 0))
            if len(spikes[0]) > 0:
                for spike_ind in spikes[0]:
                        if check[spike_ind] == 0:
                            check[spike_ind] = 1
                            matrix[spike_ind, i] = 1
                            synaptic[i+1,(order-1)*num_neurons:order*num_neurons] = synaptic[i+1,(order-1)*num_neurons:order*num_neurons] + C_matrix[spike_ind,:] *Isyn / num_connections[spike_ind]

            negatives = np.where(Y[i,:]< 0)
            if len(negatives[0]) > 0:
                for index in negatives[0]:
                    check[index] = 0


    if return_dict == 0:
        return  Y, [Y,W], matrix, synaptic
    else:
        return_dict['data_ML'] = Y 
        return_dict['Y_ML'] = [Y,W] 
        return_dict['Matrix_ML'] = np.array(matrix.todense())
        return_dict['synaptic_ML'] = synaptic

def ML_Equation_Pairs_tests(y, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I, tau, gap_junction, v_neurons,gap_current,synaptic_current ):
    '''
    Algorithm that integrates the equations of the modified Morris-Lecar model, as well as the synaptic filter.

    Parameters:
        y (tuple[float]):
            The signal of the Neuron - Voltage,  recovery variable, and Synaptic Current.
        order (int):
            The order of the synaptic filter - Max value of 5
        psi (float, optional):
            Scaling constant
        V1 (float, optinoal):
            Potential at wich m_inf = 0.5
        V2 (float, optional):
            Reciprocal of the voltage dependance of m_inf
        V3 (float, optional):
            Potential at wich w_inf = 0.5
        V4 (float, optional):
            Reciprocal of the voltage dependance of w_inf
        gna (float, optional):
            Conductance of the Na channel
        gk (float, optional):
            Conductance of the K channel
        gshunt (float, optional):
            Conductance of the Shunt channel
        Ena (float, optional):
            Resting potential of the Na channel
        Ek (float, optional):
            Resting potential of the K channel
        Eshunt (float, optinoal):
            Resting potential of the shunt channel
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

    #HH differential equations (voltage and recovery variable)
    minf = 0.5 * (1 + np.tanh( ((y[0] - V1 )/ V2)))
    Iion = gna * minf * (y[0] - Ena) + gk * y[1] * (y[0] -Ek) + gshunt * (y[0] - Eshunt)
    dvdt = ( - Iion - gap_junction * np.sum(y[0] - v_neurons) + I - y[2] * (y[0]- Vreversal)) / C
    winf = 0.5 * (1 + np.tanh( (y[0] - V3) / V4))
    dwdt = psi * (winf - y[1])*np.cosh( (y[0] - V3) / 2 / V4)

    gap_current[:] = gap_junction * np.sum(y[0] - v_neurons)
    synaptic_current[:] = y[2] * (y[0]- Vreversal)
    #Computing the synaptic filtering
    y = np.append(y,0)
    for i in range(2, 2+order):
        y[i] = -y[i] / tau + y[i+1]

    #Returning the data, making sure it is in the correct (numpy array) format
    dydt = [dvdt,dwdt]
    dydt = np.array(dydt,dtype=object)
    for i in range(2, 2+order):
        dydt = np.append(dydt,float(y[i]))
    return dydt

def ML_Neuron_Pairs_tests(dt,t_final,order,y0,w0,psi,V1,V2,V3,V4,gna,gk,gshunt,Ena,Ek,Eshunt,C,I,Isyn,gap_junction,tau,gap_current,synaptic_current,return_dict=0):
    ''' 
    Runge-Kutta integration of the 4th order of the ML model in the case of pairs of two neurons or single neurons

    Parameters:
        dt (float):
            Time step of the simulation
        t_final (float):
            Final time of the simulation
        order (int):
            Order of the synaptic filtering (maximum value = 5)
        y0 (float):
            Initial voltage of the system
        w0 (float):
            Initial value of the recovery variable
        psi (float, optional):
            Scaling constant
        V1 (float, optinoal):
            Potential at wich m_inf = 0.5
        V2 (float, optional):
            Reciprocal of the voltage dependance of m_inf
        V3 (float, optional):
            Potential at wich w_inf = 0.5
        V4 (float, optional):
            Reciprocal of the voltage dependance of w_inf
        gna (float, optional):
            Conductance of the Na channel
        gk (float, optional):
            Conductance of the K channel
        gshunt (float, optional):
            Conductance of the Shunt channel
        Ena (float, optional):
            Resting potential of the Na channel
        Ek (float, optional):
            Resting potential of the K channel
        Eshunt (float, optinoal):
            Resting potential of the shunt channel
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
        I = np.array( [ [I], [I] ])
    num_neurons = len(y0)
    check = np.zeros(num_neurons)
    
    #Setting the limit of the synaptic filtering order
    if order >5:
        print('The maximum order of t he synaptic filter is 5')
        order = 5

    #Initialisating the variables we need for the simulation 
    matrix = dok_matrix((num_neurons,int(t_final/dt)))
    Y = np.zeros((Nsteps, num_neurons * (2+order)))
    data = np.zeros((Nsteps, num_neurons))
    end = num_neurons * (2+order) -1

    #Setting the initial conditions of the system
    for i in range (0,num_neurons):
        Y[0,i*(2+order)] = y0[i]
        Y[0,1 + i*(2+order)] = w0[i]
        data[0,i] = y0[i]

    #Runge-Kutta 4th order loop
    for i in range(0,Nsteps - 1):
        for k in range(0,num_neurons):
            k1 = ML_Equation_Pairs_tests(Y[i, k*(2+order): (k+1) * (2+order)], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, gap_junction, Y[i, 0:end:2+order],gap_current[i,k],synaptic_current[i,k] )
            k2 = ML_Equation_Pairs_tests(Y[i, k*(2+order): (k+1) * (2+order)] + 0.5*dt*k1, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, gap_junction, Y[i, 0:end:2+order],gap_current[i,k],synaptic_current[i,k] )
            k3 = ML_Equation_Pairs_tests(Y[i, k*(2+order): (k+1) * (2+order)] + 0.5*dt*k2, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, gap_junction, Y[i, 0:end:2+order], gap_current[i,k],synaptic_current[i,k])
            k4 = ML_Equation_Pairs_tests(Y[i, k*(2+order): (k+1) * (2+order)] + dt * k3, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, gap_junction, Y[i, 0:end:2+order],gap_current[i,k],synaptic_current[i,k] )

            Y[i + 1, k * (2 + order): (k+1) *(2+order)] = Y[i, k * (2+order): (k+1)*(2+order) ] + 1/6 * dt * (k1 + 2*k2 + 2*k3 + k4)

        #Checking for spikes
        for k in range(0,num_neurons):
            if i > 0 and Y[i,k*(2+order)] > 10 and check[k] == 0:
                matrix[k,i] = 1
                check[k] = 1
                for l in range(0,num_neurons):
                    if l != k:
                        Y[i+1,l*(2+order) + 2 + order-1] = Y[i+1,l * (2+order) +2 + order-1] + Isyn[k,l]
            data[i+1,k] = Y[i+1,k*(2+order)]  
            if data[i+1,k] < 0:
                check[k] = 0

    if return_dict == 0:
        return data, Y, matrix
    else:
        return_dict['data_ML'] = data 
        return_dict['Y_ML'] = Y 
        return_dict['Matrix_ML'] = np.array(matrix.todense())