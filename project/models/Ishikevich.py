import numpy as np 
from scipy.sparse import dok_matrix

def IZH_Equation_Pairs(y,order,C,I,vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau,v_neurons):
    '''
    Algorithm that integrates the equations of the Izhikevich model, as well as the synaptic filter.

    Parameters:
        y (tuple[float]):
            The signal of the Neuron - Voltage, recovery variable, and Synaptic Current.
        order (int):
            The order of the synaptic filter - Max value of 5
        C (float):
            Total conductance of the neuron - Speed of the differential equation
        I (float):
            Injected Current
        vr (float):
            Rest Voltage
        vt (float):
            Check
        gap_junction (float): 
            Gap junction strength
        a (float):
            Recovery variable time constant
        b (float):
            Neuron's input resistance
        rheobase_strength (float):
            Neuron's rheobase strength
        scale_u (float):
            Scale of the recovery variable
        tau (float):
            Time constant for the synaptic filter
        v_neurons (float):
            Voltage of the neighbouring neurons

    Returns:
        dydt (tuple[float]):
            The result of integrating the input signal and the synaptic current
    '''
    #Definition of the reversal potntial of the neuron, in this case it is inhibitory
    Vreversal = -80

   #IZH differential equations (voltage and recovery variable)
    dvdt = (rheobase_strength * (y[0] - vr) * (y[0] - vt) - scale_u * y[1] + I - gap_junction * np.sum(y[0]-v_neurons) - y[2] *(y[0]-Vreversal))   / C
    dudt = a * (b*(y[0] - vr) - y[1])

    #Computing the synaptic filtering
    y = np.append(y,0)
    for i in range(2,2+order):
        y[i] = -y[i] / tau + y[i+1]

    #Returning the data, making sure it is in the correct (numpy array) format
    dydt = [dvdt,dudt]
    dydt = np.array(dydt,dtype=object)
    for i in range(2,2+order):
        dydt = np.append(dydt,float(y[i]))

    return dydt 


def IZH_Equation_Network(y,u,synaptic,order,C,I,vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau,connectivity_matrix):
    '''
    Algorithm that integrates the equations of the Izhikevich model, as well as the synaptic filter.

    Parameters:
        y (tuple[float]):
            The signal of the Neuron - Voltage
        u (tuple[float]):
            The signal of the Neuron - Recovery variable
        synaptic (tuple[float]):
            The synaptic current
        order (int):
            The order of the synaptic filter - Max value of 5
        C (float):
            Total conductance of the neuron - Speed of the differential equation
        I (float):
            Injected Current
        vr (float):
            Rest Voltage
        vt (float):
            Check
        gap_junction (float): 
            Gap junction strength
        a (float):
            Recovery variable time constant
        b (float):
            Neuron's input resistance
        rheobase_strength (float):
            Neuron's rheobase strength
        scale_u (float):
            Scale of the recovery variable
        tau (float):
            Time constant for the synaptic filter
        connectivity_matrix (dok_matrix):
            An sparse connectivity matrix, containing ones if the neurons are connected and 0 if the neurons are not connected

    Returns:
        dydt (tuple[float]):
            The result of integrating the input signal and the synaptic current
    '''


    #Definition of the reversal potntial of the neuron, in this case it is inhibitory
    Vreversal = -80

    #IZH differential equation
    dvdt = (rheobase_strength * np.multiply((y - vr),(y - vt)) - scale_u * u + I + gap_junction *np.ravel((connectivity_matrix.multiply( np.subtract.outer(y, y))).sum(axis=0)) -np.multiply(synaptic[0:len(y)],(y- Vreversal)) )  / C
    dudt = a * np.subtract(b*(y - vr),u)

    for i in range(0,order):
        if i == order -1 :
             synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau
        else:
            synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau + synaptic[(i+1)*len(y):(i+2)*len(y)]

    #Returning the data, making sure it is in the correct (numpy array) format
    dydt = [dvdt,dudt]
    dydt = np.array(dydt,dtype=object)
    return dydt 

def IZH_Neuron_Pairs(dt,t_final,order,y0,u0,I,Isyn,C,vr,vt,rheobase_strength,a,b,c,d,vpeak,scale_u,gap_junction,tau,return_dict = 0):
    ''' 
    Runge-Kutta integration of the 4th order of the IZH model in the case of pairs of two neurons or single neurons

    Parameters:
        dt (float):
            Time step of the simulation
        t_final (float):
            Final time of the simulation
        order (int):
            Order of the synaptic filtering (maximum value = 5)
        y0 (float):
            Initial voltage of the system
        u0 (float):
            Initial value of the recovery variable of the system
        I (float):
            Injected Current
        Isyn (float):
            Strength of the chemical synaptse
        C (float):
            Total conductance of the neuron - Speed of the differential equation
        vr (float):
            Rest Voltage
        vt (float):
            Check
        rheobase_strength (float):
            Neuron's rheobase strength
        a (float):
            Recovery variable time constant
        b (float):
            Neuron's input resistance
        c (float):
            Voltage reset value
        d (float):  
            Injected current after a spike
        vpeak (float):
            Voltage of the peak of the spike
        scale_u (float):
            Scale of the recovery variable
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
        I = np.array( [ [I] , [I] ] )
    num_neurons = len(y0)

    #Setting the limit of the synaptic filtering order
    if order >5:
        print('The maximum order of t he synaptic filter is 5')
        order = 5

    #Initialisating the variables we need for the simulation
    Y = np.zeros( (Nsteps, num_neurons * (2 + order)))
    matrix = dok_matrix((num_neurons,int(t_final/dt)))
    end = num_neurons * (2 + order) -1
    data = np.zeros( (Nsteps, num_neurons))

    #Setting the initial conditions of the system
    for i in range(0,num_neurons):
        Y[0,i*(2+order)] = y0[i]
        Y[0,1+i*(2+order)] = u0[i]
        data[0,i] = y0[i]

    #Runge-Kutta 4th order loop
    for i in range(0,Nsteps - 1):
        for k in range(0,num_neurons):
            k1 = IZH_Equation_Pairs( Y[i,k*(2+order):(k+1)*(2+order)] ,order,C,I[i,k],vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau,Y[i,0:end:2+order])
            k2 = IZH_Equation_Pairs( Y[i,k*(2+order):(k+1)*(2+order)] +0.5 * dt * k1,order,C,I[i,k],vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau,Y[i,0:end:2+order])
            k3 = IZH_Equation_Pairs( Y[i,k*(2+order):(k+1)*(2+order)] +0.5 * dt * k2,order,C,I[i,k],vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau,Y[i,0:end:2+order])
            k4 = IZH_Equation_Pairs( Y[i,k*(2+order):(k+1)*(2+order)] + dt * k3,order,C,I[i,k],vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau,Y[i,0:end:2+order])

            Y[i+1,k*(2+order):(k+1)*(2+order)] = Y[i,k*(2+order):(k+1)*(2+order)] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        #Checking for spikes
        for k in range(0,num_neurons):
            if Y[i+1, k * (2 + order)] >= vpeak:
                Y[i+1, k * (2+order)] = c 
                Y[i, k * (2 +order)] = vpeak #no influence
                Y[i+1, 1 + k * (2+order)] = Y[i+1, 1+k*(2+order)] + d
                matrix[k,i] = 1
                for l in range(0,num_neurons):
                    if l!= k:
                        Y[i+1, l *(2+order) + 2 + order -1] = Y[i+1, l*(2+order) + 2 +order - 1] + Isyn[k,l]

            data[i+1,k] = Y[i+1,k*(2+order)]

    if return_dict == 0:
        return data, Y, matrix
    else:
        return_dict['data_IZH'] = data 
        return_dict['Y_IZH'] = Y 
        return_dict['Matrix_IZH'] = np.array(matrix.todense())


def IZH_Neuron_Network(dt,t_final,order,y0,u0,I,Isyn,C,vr,vt,rheobase_strength,a,b,c,d,vpeak,scale_u,gap_junction,tau,E_matrix,C_matrix,return_dict=0):
    ''' 
    Runge-Kutta integration of the 4th order of the IZH model in the case of pairs of two neurons or single neurons

    Parameters:
        dt (float):
            Time step of the simulation
        t_final (float):
            Final time of the simulation
        order (int):
            Order of the synaptic filtering (maximum value = 5)
        y0 (float):
            Initial voltage of the system
        u0 (float):
            Initial value of the recovery variable of the system
        I (float):
            Injected Current
        Isyn (float):
            Strength of the chemical synaptse
        C (float):
            Total conductance of the neuron - Speed of the differential equation
        vr (float):
            Rest Voltage
        vt (float):
            Check
        rheobase_strength (float):
            Neuron's rheobase strength
        a (float):
            Recovery variable time constant
        b (float):
            Neuron's input resistance
        c (float):
            Voltage reset value
        d (float):  
            Injected current after a spike
        vpeak (float):
            Voltage of the peak of the spike
        scale_u (float):
            Scale of the recovery variable
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
            The complete signal of each of the neurons - Voltage and recovery variable
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

    #compute the number of connections of each neuron
    num_connections = np.zeros(num_neurons)
    for i in range(0,num_neurons):
        num_connections[i] = len((np.argwhere(np.array(C_matrix.todense())[i,:]>0) * dt).flatten())

    #Runge-Kutta 4th order method 
    for i in range(0,Nsteps - 1):
        k1 = IZH_Equation_Network( Y[i,:] ,U[i,:],synaptic[i,:],order,C,I[i,:],vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau, E_matrix)
        k2 = IZH_Equation_Network( np.float64(Y[i,:] +0.5 * dt * k1[0]),U[i,:]+ 0.5*dt*k1[1],synaptic[i,:],order,C,I[i,:],vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau, E_matrix)
        k3 = IZH_Equation_Network( np.float64(Y[i,:] +0.5 * dt * k2[0]),U[i,:] + 0.5*dt*k2[1],synaptic[i,:],order,C,I[i,:],vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau, E_matrix)
        k4 = IZH_Equation_Network( np.float64(Y[i,:] + dt * k3[0]),U[i,:]+ dt*k3[1],synaptic[i,:],order,C,I[i,:],vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau, E_matrix)

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
                        synaptic[i+1,(order-1)*num_neurons:order*num_neurons] = synaptic[i+1,(order-1)*num_neurons:order*num_neurons] + C_matrix[spike_ind,:] *Isyn / num_connections[spike_ind]

        data[i+1,:] = Y[i+1,:]

        negatives = np.where(Y[i,:]< 0)
        if len(negatives[0]) > 0:
            for index in negatives[0]:
                check[index] = 0

    if return_dict == 0:
        return data, Y, matrix, synaptic
    else:
        return_dict['data_IZH'] = data 
        return_dict['Y_IZH'] = Y 
        return_dict['Matrix_IZH'] = np.array(matrix.todense())
        return_dict['synaptic_IZH'] = synaptic

def IZH_Equation_Network_tests(y,u,synaptic,order,C,I,vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau,connectivity_matrix,gap_current,synaptic_current):
    '''
    Algorithm that integrates the equations of the Izhikevich model, as well as the synaptic filter.

    Parameters:
        y (tuple[float]):
            The signal of the Neuron - Voltage
        u (tuple[float]):
            The signal of the Neuron - Recovery variable
        synaptic (tuple[float]):
            The synaptic current
        order (int):
            The order of the synaptic filter - Max value of 5
        C (float):
            Total conductance of the neuron - Speed of the differential equation
        I (float):
            Injected Current
        vr (float):
            Rest Voltage
        vt (float):
            Check
        gap_junction (float): 
            Gap junction strength
        a (float):
            Recovery variable time constant
        b (float):
            Neuron's input resistance
        rheobase_strength (float):
            Neuron's rheobase strength
        scale_u (float):
            Scale of the recovery variable
        tau (float):
            Time constant for the synaptic filter
        connectivity_matrix (dok_matrix):
            An sparse connectivity matrix, containing ones if the neurons are connected and 0 if the neurons are not connected

    Returns:
        dydt (tuple[float]):
            The result of integrating the input signal and the synaptic current
    '''


    #Definition of the reversal potntial of the neuron, in this case it is inhibitory
    Vreversal = -80

    #IZH differential equation
    gap_current[:] = gap_junction *np.ravel((connectivity_matrix.multiply( np.subtract.outer(y, y))).sum(axis=0))
    synaptic_current[:] = np.multiply(synaptic[0:len(y)],(y- Vreversal))
    dvdt = (rheobase_strength * np.multiply((y - vr),(y - vt)) - scale_u * u + I + gap_junction *np.ravel((connectivity_matrix.multiply( np.subtract.outer(y, y))).sum(axis=0)) -np.multiply(synaptic[0:len(y)],(y- Vreversal)) )  / C
    dudt = a * np.subtract(b*(y - vr),u)

    for i in range(0,order):
        if i == order -1 :
             synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau
        else:
            synaptic[i*len(y):(i+1)*len(y)] = -synaptic[i*len(y):(i+1)*len(y)] / tau + synaptic[(i+1)*len(y):(i+2)*len(y)]

    #Returning the data, making sure it is in the correct (numpy array) format
    dydt = [dvdt,dudt]
    dydt = np.array(dydt,dtype=object)
    return dydt 

def IZH_Neuron_Network_tests(dt,t_final,order,y0,u0,I,Isyn,C,vr,vt,rheobase_strength,a,b,c,d,vpeak,scale_u,gap_junction,tau,E_matrix,C_matrix,gap_current,synaptic_current,return_dict=0):
    ''' 
    Runge-Kutta integration of the 4th order of the IZH model in the case of pairs of two neurons or single neurons

    Parameters:
        dt (float):
            Time step of the simulation
        t_final (float):
            Final time of the simulation
        order (int):
            Order of the synaptic filtering (maximum value = 5)
        y0 (float):
            Initial voltage of the system
        u0 (float):
            Initial value of the recovery variable of the system
        I (float):
            Injected Current
        Isyn (float):
            Strength of the chemical synaptse
        C (float):
            Total conductance of the neuron - Speed of the differential equation
        vr (float):
            Rest Voltage
        vt (float):
            Check
        rheobase_strength (float):
            Neuron's rheobase strength
        a (float):
            Recovery variable time constant
        b (float):
            Neuron's input resistance
        c (float):
            Voltage reset value
        d (float):  
            Injected current after a spike
        vpeak (float):
            Voltage of the peak of the spike
        scale_u (float):
            Scale of the recovery variable
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
            The complete signal of each of the neurons - Voltage and recovery variable
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

    #compute the number of connections of each neuron
    num_connections = np.zeros(num_neurons)
    for i in range(0,num_neurons):
        num_connections[i] = len((np.argwhere(np.array(C_matrix.todense())[i,:]>0) * dt).flatten())

    #Runge-Kutta 4th order method 
    for i in range(0,Nsteps - 1):
        k1 = IZH_Equation_Network_tests( Y[i,:] ,U[i,:],synaptic[i,:],order,C,I[i,:],vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau, E_matrix, gap_current[i,:], synaptic_current[i,:])
        k2 = IZH_Equation_Network_tests( np.float64(Y[i,:] +0.5 * dt * k1[0]),U[i,:]+ 0.5*dt*k1[1],synaptic[i,:],order,C,I[i,:],vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau, E_matrix, gap_current[i,:], synaptic_current[i,:])
        k3 = IZH_Equation_Network_tests( np.float64(Y[i,:] +0.5 * dt * k2[0]),U[i,:] + 0.5*dt*k2[1],synaptic[i,:],order,C,I[i,:],vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau, E_matrix, gap_current[i,:], synaptic_current[i,:])
        k4 = IZH_Equation_Network_tests( np.float64(Y[i,:] + dt * k3[0]),U[i,:]+ dt*k3[1],synaptic[i,:],order,C,I[i,:],vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau, E_matrix, gap_current[i,:], synaptic_current[i,:])

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
                        synaptic[i+1,(order-1)*num_neurons:order*num_neurons] = synaptic[i+1,(order-1)*num_neurons:order*num_neurons] + C_matrix[spike_ind,:] *Isyn / num_connections[spike_ind]

        data[i+1,:] = Y[i+1,:]

        negatives = np.where(Y[i,:]< 0)
        if len(negatives[0]) > 0:
            for index in negatives[0]:
                check[index] = 0

    if return_dict == 0:
        return data, Y, matrix, synaptic
    else:
        return_dict['data_IZH'] = data 
        return_dict['Y_IZH'] = Y 
        return_dict['Matrix_IZH'] = np.array(matrix.todense())
        return_dict['synaptic_IZH'] = synaptic

def IZH_Equation_Pairs_tests(y,order,C,I,vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau,v_neurons,gap_current,synaptic_current):
    '''
    Algorithm that integrates the equations of the Izhikevich model, as well as the synaptic filter.

    Parameters:
        y (tuple[float]):
            The signal of the Neuron - Voltage, recovery variable, and Synaptic Current.
        order (int):
            The order of the synaptic filter - Max value of 5
        C (float):
            Total conductance of the neuron - Speed of the differential equation
        I (float):
            Injected Current
        vr (float):
            Rest Voltage
        vt (float):
            Check
        gap_junction (float): 
            Gap junction strength
        a (float):
            Recovery variable time constant
        b (float):
            Neuron's input resistance
        rheobase_strength (float):
            Neuron's rheobase strength
        scale_u (float):
            Scale of the recovery variable
        tau (float):
            Time constant for the synaptic filter
        v_neurons (float):
            Voltage of the neighbouring neurons

    Returns:
        dydt (tuple[float]):
            The result of integrating the input signal and the synaptic current
    '''
    #Definition of the reversal potntial of the neuron, in this case it is inhibitory
    Vreversal = -80

   #IZH differential equations (voltage and recovery variable)
    dvdt = (rheobase_strength * (y[0] - vr) * (y[0] - vt) - scale_u * y[1] + I - gap_junction * np.sum(y[0]-v_neurons) - y[2] *(y[0]-Vreversal))   / C
    dudt = a * (b*(y[0] - vr) - y[1])
    gap_current[:] = gap_junction * np.sum(y[0]-v_neurons)
    synaptic_current[:] = y[2] *(y[0]-Vreversal)

    #Computing the synaptic filtering
    y = np.append(y,0)
    for i in range(2,2+order):
        y[i] = -y[i] / tau + y[i+1]

    #Returning the data, making sure it is in the correct (numpy array) format
    dydt = [dvdt,dudt]
    dydt = np.array(dydt,dtype=object)
    for i in range(2,2+order):
        dydt = np.append(dydt,float(y[i]))

    return dydt 

def IZH_Neuron_Pairs_tests(dt,t_final,order,y0,u0,I,Isyn,C,vr,vt,rheobase_strength,a,b,c,d,vpeak,scale_u,gap_junction,tau,gap_current,synaptic_current,return_dict = 0):
    ''' 
    Runge-Kutta integration of the 4th order of the IZH model in the case of pairs of two neurons or single neurons

    Parameters:
        dt (float):
            Time step of the simulation
        t_final (float):
            Final time of the simulation
        order (int):
            Order of the synaptic filtering (maximum value = 5)
        y0 (float):
            Initial voltage of the system
        u0 (float):
            Initial value of the recovery variable of the system
        I (float):
            Injected Current
        Isyn (float):
            Strength of the chemical synaptse
        C (float):
            Total conductance of the neuron - Speed of the differential equation
        vr (float):
            Rest Voltage
        vt (float):
            Check
        rheobase_strength (float):
            Neuron's rheobase strength
        a (float):
            Recovery variable time constant
        b (float):
            Neuron's input resistance
        c (float):
            Voltage reset value
        d (float):  
            Injected current after a spike
        vpeak (float):
            Voltage of the peak of the spike
        scale_u (float):
            Scale of the recovery variable
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
        I = np.array( [ [I] , [I] ] )
    num_neurons = len(y0)

    #Setting the limit of the synaptic filtering order
    if order >5:
        print('The maximum order of t he synaptic filter is 5')
        order = 5

    #Initialisating the variables we need for the simulation
    Y = np.zeros( (Nsteps, num_neurons * (2 + order)))
    matrix = dok_matrix((num_neurons,int(t_final/dt)))
    end = num_neurons * (2 + order) -1
    data = np.zeros( (Nsteps, num_neurons))

    #Setting the initial conditions of the system
    for i in range(0,num_neurons):
        Y[0,i*(2+order)] = y0[i]
        Y[0,1+i*(2+order)] = u0[i]
        data[0,i] = y0[i]

    #Runge-Kutta 4th order loop
    for i in range(0,Nsteps - 1):
        for k in range(0,num_neurons):
            k1 = IZH_Equation_Pairs_tests( Y[i,k*(2+order):(k+1)*(2+order)] ,order,C,I[i,k],vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau,Y[i,0:end:2+order],gap_current[i,k],synaptic_current[i,k])
            k2 = IZH_Equation_Pairs_tests( Y[i,k*(2+order):(k+1)*(2+order)] +0.5 * dt * k1,order,C,I[i,k],vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau,Y[i,0:end:2+order],gap_current[i,k],synaptic_current[i,k])
            k3 = IZH_Equation_Pairs_tests( Y[i,k*(2+order):(k+1)*(2+order)] +0.5 * dt * k2,order,C,I[i,k],vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau,Y[i,0:end:2+order],gap_current[i,k],synaptic_current[i,k])
            k4 = IZH_Equation_Pairs_tests( Y[i,k*(2+order):(k+1)*(2+order)] + dt * k3,order,C,I[i,k],vr,vt,gap_junction,a,b,rheobase_strength,scale_u,tau,Y[i,0:end:2+order],gap_current[i,k],synaptic_current[i,k])

            Y[i+1,k*(2+order):(k+1)*(2+order)] = Y[i,k*(2+order):(k+1)*(2+order)] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        #Checking for spikes
        for k in range(0,num_neurons):
            if Y[i+1, k * (2 + order)] >= vpeak:
                Y[i+1, k * (2+order)] = c 
                Y[i, k * (2 +order)] = vpeak #no influence
                Y[i+1, 1 + k * (2+order)] = Y[i+1, 1+k*(2+order)] + d
                matrix[k,i] = 1
                for l in range(0,num_neurons):
                    if l!= k:
                        Y[i+1, l *(2+order) + 2 + order -1] = Y[i+1, l*(2+order) + 2 +order - 1] + Isyn[k,l]

            data[i+1,k] = Y[i+1,k*(2+order)]

    if return_dict == 0:
        return data, Y, matrix
    else:
        return_dict['data_IZH'] = data 
        return_dict['Y_IZH'] = Y 
        return_dict['Matrix_IZH'] = np.array(matrix.todense())