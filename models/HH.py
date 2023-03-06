import numpy as np 

def an(v,vt):
    return -0.032 * (v-vt-15) / (np.exp(-(v-vt-15)/5)-1)

def bn(v,vt):
    return 0.5* np.exp(-(v-vt-10)/40)

def am(v,vt):

    return -0.32 * (v-vt-13) / (np.exp(-(v-vt-13)/4)-1)

def bm(v,vt):
    return 0.28 * (v-vt-40) / (np.exp((v-vt-40)/5) -1)

def ah(v,vt):
    return 0.128*np.exp(-(v-vt-17)/18)

def bh(v,vt):
    return 4 / (1+np.exp(-(v-vt-40)/5))

def HH_RK(y,order,gna,gk,gl,Ena,Ek,El,C,I,tau,k,v_neurons,A):
    '''
    Algorithm that calculates the changes in v, m, h, and n as per the HH model equations. It returns an np.array containing these changes
    '''
    Vrest = - 80 #because it's an inhibitory neuron
    vt = -58 
    Ina = gna * y[2]**3 * y[3] * (y[0] - Ena)
    Ik = gk * y[1]**4 * (y[0]- Ek)
    print(v_neurons)
    dvdt = (-Ina -Ik - gl * (y[0] - El) + I - k * np.sum(A * (y[0] - v_neurons)) -y[4] * (y[0] - Vrest)) / C 

    dmdt = am(y[0],vt) * (1-y[2]) - bm(y[0],vt) * y[2]
    dhdt = ah(y[0],vt) * (1-y[3]) - bh(y[0],vt) * y[3]
    dndt = an(y[0],vt) * (1-y[1]) - bn(y[0],vt) * y[1]
    y = np.append(y,0)
    for i in range(4,4+order):
        y[i] =  -y[i] / tau + y[i+1] 

    dydt = [dvdt,dndt,dmdt,dhdt]
    dydt = np.array(dydt,dtype=object)
    for i in range(4,4+order):
        dydt = np.append(dydt,float(y[i]))

    return dydt

def rk_simplemodel(dt, t_final, order, y0, n0, m0, h0, gna, gk, gl, Ena, Ek, El, C, I, Isyn, strength, tau, E_matrix, C_matrix):
    ''' 
    Runge Kutta integration of the 4th order of the HH model, for various orders and numbers of neurons
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

    #variables that store the signal
    Y = np.zeros((Nsteps,num_neurons*(4+order)))
    data = np.zeros((Nsteps,num_neurons))

    #computing where is the end of our array, a tool that will help us later (to be concise)
    end = num_neurons * (4+order) -1

    #assign the initial values
    for i in range (0,num_neurons): 
        Y[0,i*(4+order)] = y0[i]
        Y[0,1+i*(4+order)] = n0[i]
        Y[0,2+i*(4+order)] = m0[i]
        Y[0,3+i*(4+order)] = h0[i]
    
        #data we are outputing for convenience
        data[0,i] = y0[i]

    #Runge-Kutta 4th order method 
    for i in range(0,Nsteps-1):
        for k in range(0,num_neurons):
            k1 = HH_RK(Y[i, k*(4+order): (k+1) * (4+order)], order, gna, gk, gl, Ena, Ek, El, C, I[i,k], tau, strength, Y[i, 0:end:4+order], E_matrix[k,:] )
            print(Y[i,0],Y[i,4+order]) 
            k2 = HH_RK(Y[i, k*(4+order): (k+1) * (4+order)] + 0.5*dt*k1, order, gna, gk, gl, Ena, Ek, El, C, I[i,k], tau, strength, Y[i, 0:end:4+order ], E_matrix[k,:] )
            #print('k2',k2)
            k3 = HH_RK(Y[i, k*(4+order): (k+1) * (4+order)] + 0.5*dt*k2, order, gna, gk, gl, Ena, Ek, El, C, I[i,k], tau, strength, Y[i, 0:end:4+order ], E_matrix[k,:] )
            
            k4 = HH_RK(Y[i, k*(4+order): (k+1) * (4+order)] + dt * k3, order, gna, gk, gl, Ena, Ek, El, C, I[i,k], tau, strength, Y[i, 0:end:4+order ], E_matrix[k,:] )
            

            Y[i + 1, k * (4 + order): (k+1) *(4+order)] = Y[i, k * (4+order): (k+1)*(4+order) ] + 1/6 * dt * (k1 + 2*k2 + 2*k3 + k4)

        for k in range(0,num_neurons):
            if i>0 and ( Y[i, k*(4+order)] >= Y [i-1,k*(4+order)]) and (Y[i,k*(4+order)] >= Y[i+1,k*(4+order)]) and Y[i,k*(4+order)] > 0:
                for l in range(0,num_neurons):
                        Y[i+1,l*(4+order) + 4 + order-1] = Y[i+1,l * (4+order) +4 + order-1] + C_matrix[k,l] *Isyn[k,l]
            data[i+1,k] = Y[i+1,k*(4+order)]  
    return data, Y

def rk_Icst(dt, t_final, order, y0, n0, m0, h0, gna, gk, gl, Ena, Ek, El, C, I, Isyn, strength, tau):
    ''' 
    Runge Kutta integration of the HH model for the case of a single intensity, the following function has the more complete approeach
    '''
    Nsteps = int(t_final/dt)

    if type(y0) is int:
        y0 = [y0]
        n0 = [n0]
        m0 = [m0]
        h0 = [h0]
        I = [I]
    num_neurons = len(y0)
    if order > 5: #why?
        order = 5

    Y = np.zeros((Nsteps,num_neurons*(4+order)))
    data = np.zeros((Nsteps,num_neurons))
    end = num_neurons * (4+order) -1
    for i in range (0,num_neurons): #assign the initial values
        Y[0,i*(4+order)] = y0[i]
        Y[0,1+i*(4+order)] = n0[i]
        Y[0,2+i*(4+order)] = m0[i]
        Y[0,3+i*(4+order)] = h0[i]

        #data we are outputing out of convenience
        data[0,i] = y0[i]
    
    for i in range(0,Nsteps-1):
        for k in range(0,num_neurons):
            k1 = HH_RK(Y[i, k*(4+order): (k+1) * (4+order)], order, gna, gk, gl, Ena, Ek, El, C, I[k], tau, strength, Y[i, 0:end:4+order] )
            #print('k1',k1) a[0:4+1:4+1]
            k2 = HH_RK(Y[i, k*(4+order): (k+1) * (4+order)] + 0.5*dt*k1, order, gna, gk, gl, Ena, Ek, El, C, I[k], tau, strength, Y[i, 0:end:4+order ] )
            #print('k2',k2)
            k3 = HH_RK(Y[i, k*(4+order): (k+1) * (4+order)] + 0.5*dt*k2, order, gna, gk, gl, Ena, Ek, El, C, I[k], tau, strength, Y[i, 0:end:4+order ] )
            
            k4 = HH_RK(Y[i, k*(4+order): (k+1) * (4+order)] + dt * k3, order, gna, gk, gl, Ena, Ek, El, C, I[k], tau, strength, Y[i, 0:end:4+order ] )
            

            Y[i + 1, k * (4 + order): (k+1) *(4+order)] = Y[i, k * (4+order): (k+1)*(4+order)] + 1/6 * dt * (k1 + 2*k2 + 2*k3 + k4)

        for k in range(0,num_neurons):
            if i>0 and ( Y[i, k*(4+order)] >= Y [i-1,k*(4+order)]) and (Y[i,k*(4+order)] >= Y[i+1,k*(4+order)]) and Y[i,k*(4+order)] > 0:
                for l in range(0,num_neurons):
                    if l != k:
                        Y[i+1,l*(4+order) + 4 + order-1] = Y[i+1,l * (4+order) +4 + order-1] + Isyn[k,l]
            data[i+1,k] = Y[i+1,k*(4+order)]  
    return data, Y