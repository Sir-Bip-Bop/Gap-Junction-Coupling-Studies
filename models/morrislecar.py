import numpy as np 
from scipy.sparse import dok_matrix

def ML_RK(y, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I, tau, k, v_neurons ):
    '''
    Algorithm of the evolution of the Morris-Lecar model, returning the numpy array dydt
    '''
    Vrest = -80


    minf = 0.5 * (1 + np.tanh( ((y[0] - V1 )/ V2)))
    Iion = gna * minf * (y[0] - Ena) + gk * y[1] * (y[0] -Ek) + gshunt * (y[0] - Eshunt)

    dvdt = ( - Iion - k * np.sum(y[0] - v_neurons) + I - y[2] * (y[0]- Vrest)) / C

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




def rk_ml(dt,t_final,order,y0,w0,psi,V1,V2,V3,V4,gna,gk,gshunt,Ena,Ek,Eshunt,C,I,Isyn,strength,tau):
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
            k1 = ML_RK(Y[i, k*(2+order): (k+1) * (2+order)], order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, strength, Y[i, 0:end:2+order] )
            #print('k1',k1) a[0:4+1:4+1]
            k2 = ML_RK(Y[i, k*(2+order): (k+1) * (2+order)] + 0.5*dt*k1, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, strength, Y[i, 0:end:2+order] )
            #print('k2',k2)
            k3 = ML_RK(Y[i, k*(2+order): (k+1) * (2+order)] + 0.5*dt*k2, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, strength, Y[i, 0:end:2+order])
            
            k4 = ML_RK(Y[i, k*(2+order): (k+1) * (2+order)] + dt * k3, order, psi,V1,V2,V3,V4,gna, gk, gshunt, Ena, Ek, Eshunt, C, I[i,k], tau, strength, Y[i, 0:end:2+order] )
            

            Y[i + 1, k * (2 + order): (k+1) *(2+order)] = Y[i, k * (2+order): (k+1)*(2+order) ] + 1/6 * dt * (k1 + 2*k2 + 2*k3 + k4)

        for k in range(0,num_neurons):
            if i>0 and ( Y[i, k*(2+order)] >= Y [i-1,k*(2+order)]) and (Y[i,k*(2+order)] >= Y[i+1,k*(2+order)]) and Y[i,k*(2+order)] > 0:
                for l in range(0,num_neurons):
                    if l != k:
                        Y[i+1,l*(2+order) + 2 + order-1] = Y[i+1,l * (2+order) +2 + order-1] + Isyn[k,l]
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
            if i>0 and ( Y[i, k*(2+order)] >= Y [i-1,k*(2+order)]) and (Y[i,k*(2+order)] >= Y[i+1,k*(2+order)]) and Y[i,k*(2+order)] > 0 and check[k] == 0:
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