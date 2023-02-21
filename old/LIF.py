import numpy as np 
def LIF_RK4(dt,t_final,v0,I,C,El,Ena,Ek,gl,gna,gk,vt,vth,vr,vspike):
    '''Function that takes all the required parameters and does an integration via Runge-Kutta of 4th order
    the order of the parameters depends on the input file, the last three parameters are not important but necessary for the code to work.

    It uses Runge-Kutta of 4th order to integrate LIF model ignoring the synaptic current
    '''
    Nsteps = int(t_final/dt)
    Y = np.zeros(Nsteps)
    Y[0] = v0
    Y1 = Y[0]
    
    for i in range(0,Nsteps-1):
        k1 = LIF_RHS(Y1,gl,El,C,I)
        k2 = LIF_RHS(Y1 + 0.5 *dt * k1,gl,El,C,I)
        k3 = LIF_RHS(Y1 + 0.5 *dt * k2,gl,El,C,I)
        k4 = LIF_RHS(Y1 + dt * k3,gl,El,C,I)

        Y2 = Y1 + (1/6) * dt * (k1 + 2 * k2 + 2* k3 + k4)

        Y[i+1] = Y2 

        if Y2 >= vth:
            Y2 = vr 
            Y[i + 1] = vspike 
        Y1 = Y2

    return Y 

def LIF_RHS(v,gl,El,C,I):
    return (-gl * (v - El) + I) / C 
