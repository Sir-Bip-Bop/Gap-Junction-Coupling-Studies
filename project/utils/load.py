def load_HH(gna=30,gk=5,gl=0.1,Ena=30,Ek=-90,El=-70,C=1):
    '''
    Function that initialites the parameters for HH.
    If the function is called without any parameters, it will return the base values of:
    'gna = 30, gk = 5, gl = 0.1, Ena = 30, Ek = -90, El = -70, C =1 '

    Parameters:
        gna (float, optional):
            constant related to the Na ion channel
        gk (float, optional):
            constant related to the K ion channel
        gl (float, optional):
            constant related to the leak channel
        Ena (float, optional):
            Resting potential for the Na channel
        Ek (float, optional):
            Resting potential for the K channel
        El (float, optional):
            Resting potential for the leakage channel
        C (float, optional):
            Time scale of the differential equation
    '''
    return [gna,gk,gl,Ena,Ek,El,C]

def load_LIF(Vth = -49.2, Vr = -66.9, w = 25, gl = 0.1, El = -70, C = 1):
    '''
    Function that initialites the parameters for LIF.
    If the function is called without any parameters, it will return the base values of:
    'Vth = -49.2, Vr = -66.9, w = 25, gl = 0.1, El = 70-,C =1 '

    Paremeters:
        Vth (float, optional):
            Spike threshold potential
        Vr (float, optional):
            Spike reset potential
        w (float, optional):
            Spike value in the data set
        gl (float, optional):
            Constant related to the leakage channels
        El (float, optional):
            Resting potential of the leakage channel
        C (float, optional):
            Time scale of the differential equation
    '''
    return [Vth,Vr,w,gl,El,C]

def load_ISH(C=1,vr = -70, vt = -48.5, k_ish = 0.019, a = 0.5, b = -1.3,c=-60, d = 100,vpeak=25, k_u = 0.06):
    '''
    Function that initialites the parameters for Izh.
    If the function is called without any parameters, it will return the base values of:
    'C=1,vr = -70, vt = -48.5, k_ish = 0.019, a = 0.5, b = -1.3,c = -60, d = 100 , vpeak=25, k_u = 0.06'

    Parameters:
        C (float, optional):
            Time scale of the differential equation 
        vr (float, optional):
            Neuron resting potential
        vt (float, optional):
            CHECK
        k_ish (float, optional):
            Neuron's rheobase strength
        a (float, optional):
            Speed of the recovery variable differential equation
        b (float, optional):
            Scale of the recovery variable
        c (float, optional):
            Spike reset potential
        d (float, optional).
            Total current injected into the neuron due to spiking
        v_peak (float, optional):
            Spike voltage in the data set
        k_u (float, optional):
            Scale value of the recovery variable
    '''
    return [C,vr,vt,k_ish,a,b,c,d,vpeak,k_u]

def load_ML(psi = 0.93, V1 = -3, V2 = 20, V3= -3, V4= 19, gna = 2, gk = 2, gshunt = 0.085, Ena = 90, Ek = -110, Eshunt = -75, C = 1):
    '''
    Function that initialites the parameters for ML.
    If the function is called without any parameters, it will return the base values of:
    'psi = 0.93, V1 = -3, V2 = 20, V3= -3, V4= 19, gna = 2, gk = 2, gshunt = 0.085, Ena = 90, Ek = -110, Eshunt = -75, C = 1'

    Parameters:
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
        C (float, optional):
            Time scale of the differential equation
    '''
    return [psi,V1,V2,V3,V4,gna,gk,gshunt,Ena,Ek,Eshunt,C]