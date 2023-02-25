import numpy as np 

def compute_chi(data):
    '''
    Computes the synchrony measure chi for a given set of voltage traces,
    data - voltage traces, each row being a different neuron
    '''

    #calculate the average voltage as a function of time
    mean_voltage = np.mean(data,1) #revisar que eje

    #calculate the variance of each trave and the average voltage