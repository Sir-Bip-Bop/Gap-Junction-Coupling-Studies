
import numpy as np 
def IF_RK(y,order,gl,El,C,I,tau,k,v_neurons,A):
    '''
    Algorithm that integrates the LIF model, returning the float dydt, the change in the signal
    '''
    Vrest = -80
    dvdt = (-gl * (y[0] - El) + I - k * np.sum(A * (y[0] - v_neurons)) - y[1]* (y[0] - Vrest)) / C  

    y = np.append(y,0)

    for i in range(1,1+order):
        y[i] =  -y[i] / tau + y[i+1]
    dydt = [dvdt]
    dydt = np.array(dydt,dtype=object)

    for i in range(1,1+order):
        dydt = np.append(dydt,float(y[i]))

    return dydt

def rk_if(dt,t_final,order,y0,Vth,Vr,w,gl,El,C,I,Isyn,strength,tau,spikelet,E_matrix,C_matrix):
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
            k1 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] ,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order],E_matrix[k,:])
            k2 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] +0.5 * dt * k1,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order], E_matrix[k,:])
            k3 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] +0.5 * dt * k2,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order], E_matrix[k,:])
            k4 = IF_RK( Y[i,k*(1+order):(k+1)*(1+order)] + dt * k3,order,gl,El,C,I[i,k],tau,strength,Y[i,0:end:1+order], E_matrix[k,:])

            Y[i+1,k*(1+order):(k+1)*(1+order)] = Y[i,k*(1+order):(k+1)*(1+order)] + (1/6)*dt*(k1+2*k2+2*k3+k4)

        for k in range(0,num_neurons):
            if Y[i+1,k * (1+order)] >= Vth:
                data[i+1,k] = w 
                Y[i+1, k * (1+order)] = Vr 
                for l in range(0,num_neurons):
                        Y[i+1,l * (1+order) +order] = Y[i+1,l*(1+order) + order] + C_matrix[k,l] *Isyn[k,l]
                        Y[i+1,l * (1+order)] = Y[i+1,l *(1+order)] + spikelet
            else:
                data[i+1,k] = Y[i+1,k *(1+order)]
    return data, Y
