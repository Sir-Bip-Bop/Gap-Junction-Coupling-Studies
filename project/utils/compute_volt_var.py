import numpy as np 

def compute_volt_var(data):
    '''
    Computes the synchrony measure of the average voltage and variance for a given set of voltage traces,
    data - voltage traces, each row being a different neuron
    '''

    #calculate the average voltage as a function of time and the total variance
    mean_voltage = np.mean(data,axis=1)
    total_variance = np.mean(np.square(mean_voltage)) - np.mean(mean_voltage)**2

    return mean_voltage, total_variance