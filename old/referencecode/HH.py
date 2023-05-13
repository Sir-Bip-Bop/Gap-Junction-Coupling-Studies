import numpy as np 

def HH_RK4(dt,t_final,y0,I,C,El,Ena,Ek,gl,gna,gk,vt,vth,vr,vspike):
    '''Function that takes all the required parameters and does an integration via Runge-Kutta of 4th order
    the order of the parameters depends on the input file, the last three parameters are not important but necessary for the code to work.

    It uses Runge-kutta of 4th order to integrate the HH set of equations with the v_th change.
    '''
    Nsteps = int(t_final/dt)
    Y = np.zeros((Nsteps,len(y0)))
    Y[0,:] = y0 

    for i in range(0,Nsteps-1):
        k1 = HH_RHS(Y[i,:],gna,gk,gl,Ena,Ek,El,C,vt,I)
        k2 = HH_RHS(Y[i,:] +0.5*dt*k1,gna,gk,gl,Ena,Ek,El,C,vt,I)
        k3 = HH_RHS(Y[i,:] + 0.5*dt*k2,gna,gk,gl,Ena,Ek,El,C,vt,I)
        k4 = HH_RHS(Y[i,:] + dt*k3,gna,gk,gl,Ena,Ek,El,C,vt,I)

        Y[i+1,:] = Y[i,:] + (1/6) * dt * (k1+2*k2+2*k3 +k4)

    return Y 

def HH_RHS(y,gna,gk,gl,Ena,Ek,El,C,vt,I):
    '''This model uses the complete set of Hodgkin-Huxley equations, wich is:
    'dV/dt = (I - gk * n^4 *(V - Ek) - gna * m^3 ** h * (V - Ek) - gl * (V - El)) / C'

    'dn/dt = an(V) * (1-n) - bn(V) * n'
    'dm/dt = am(V) * (1-m) - bm(V) * m'
    'dh/dt = ah(V) * (1-h) - bh(V) * h'

    '''
    v = y[0]
    n = y[1]
    m = y[2]
    h = y[3]

    Ina = gna * m**3 * h * (v-Ena)
    Ik = gk * n**4 * (v-Ek)

    dvdt = (-Ina - Ik - gl * (v-El) + I) / C 

    dndt = an(v,vt) * (1-n) - bn(v,vt)* n 
    dmdt = am(v,vt) * (1-m) - bm(v,vt)* m
    dhdt = ah(v,vt) * (1-h) - bh(v,vt)* h 

    dydt = [dvdt,dndt,dmdt,dhdt]

    return np.array(dydt)

#Where do these values come from?
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

