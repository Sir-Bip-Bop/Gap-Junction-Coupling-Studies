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
    return_dict['Matrix_IZH'] = matrix