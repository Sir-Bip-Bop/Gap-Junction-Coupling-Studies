def load_HH(gna=30,gk=5,gl=0.1,Ena=30,Ek=-90,El=-70,C=1):
    '''
    Function that initialites the parameters for HH.
    If the function is called without any parameters, it will return the base values of:
    'gna = 30, gk = 5, gl = 0.1, Ena = 30, Ek = -90, El = -70, C =1 '

    Parameters:
        gna (float, optional):
            constant re
    '''
    return [gna,gk,gl,Ena,Ek,El,C]

def load_LIF(Vth = -49.2, Vr = -66.9, w = 25, gl = 0.1, El = -70, C = 1):
    '''
    Function that initialites the parameters for LIF.
    If the function is called without any parameters, it will return the base values of:
    'gna = 30, gk = 5, gl = 0.1, Ena = 30, Ek = -90, El = -70, C =1 '

    If the function is called with a parameter and its value specified, it will change that specific value.
    The order matters
    '''
    return [Vth,Vr,w,gl,El,C]

def load_ISH(C=1,vr = -70, vt = -48.5, k_ish = 0.019, a = 0.5, b = -1.3,c=-60, d = 100,vpeak=25, k_u = 0.06):
    '''
    Function that initialites the parameters for Izh.
    If the function is called without any parameters, it will return the base values of:
    'C=0.6,vr = -70, vt = -40.2, k_ish = 0.008, a = 0.00505, b = -0.2, d = 0.51 , vpeak=25, k_u = 0.06'

    If the function is called with a parameter and its value specified, it will change that specific value.
    The order matters
    '''
    return [C,vr,vt,k_ish,a,b,c,d,vpeak,k_u]

def load_ML(psi = 0.93, V1 = -3, V2 = 20, V3= -3, V4= 19, gna = 2, gk = 2, gshunt = 0.085, Ena = 90, Ek = -110, Eshunt = -75, C = 1):
    '''
    Function that initialites the parameters for ML.
    If the function is called without any parameters, it will return the base values of:
    'psi = 0.3, V1 = -1.1, V2 = 24, V3= -0.7, V4= 14, gna = 0.5, gk = 0.5, gshunt = 0.064, Ena = 100, Ek = -140, Eshunt = -75, C = 0.75'

    If the function is called with a parameter and its value specified, it will change that specific value.
    The order matters
    '''
    return [psi,V1,V2,V3,V4,gna,gk,gshunt,Ena,Ek,Eshunt,C]