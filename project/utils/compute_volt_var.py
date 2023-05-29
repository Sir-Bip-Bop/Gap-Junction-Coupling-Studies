import numpy as np 

def compute_volt_var(data):
    '''
    Computes the mean of the voltage in function of time and neurons, the variance of the mean, and the variance of each individual neuron.

    Parameters:
        data (list[float]):
            Voltage traces with each row corersponding one neuron.

    Returns:
        mean_voltage (float):
            The Mean voltage in function of time of the different neurons
        total_mean (float):
            The mean of the voltage of each neuron.
        ind_variance (float):
            The variance of each individual neuron.
        total_variance (float):
            The variance of mean_voltage
    '''
    data  = data.T

    #calculate the average voltage as a function of time and the total variance
    mean_voltage = np.mean(data,axis=0)
    total_mean = np.mean(data,axis=1)

    #calculate the variance of each trace and the average voltage
    ind_variance = np.mean(np.square(data),axis = 1) - np.mean(data, axis = 1) ** 2
    total_variance = np.mean(np.square(mean_voltage)) - np.mean(mean_voltage)**2

    return mean_voltage, total_mean, ind_variance, total_variance