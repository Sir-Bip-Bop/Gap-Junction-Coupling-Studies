import numpy as np 

def compute_chi(data):
    '''
    Computes the synchrony measure chi for a given set of voltage traces,
    data - voltage traces, each row being a different neuron
    '''

    #calculate the average voltage as a function of time
    mean_voltage = np.mean(data,axis=1)

    #calculate the variance of each trave and the average voltage
    ind_variance = np.mean(data**2,axis=0) - np.mean(data,axis=0)**2
    total_variance = np.mean(mean_voltage**2,axis=0) - np.mean(mean_voltage,axis=0)**2

    #calculate chi
    chi = np.sqrt(total_variance**2 / np.mean(ind_variance**2,axis=0))

    return chi 