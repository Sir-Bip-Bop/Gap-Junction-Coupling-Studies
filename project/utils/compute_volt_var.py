def compute_volt_var(data):
    '''
    Computes the synchrony measure of the average voltage and variance for a given set of voltage traces,
    data - voltage traces, each row being a different neuron
    '''
    data  = data.T

    #calculate the average voltage as a function of time and the total variance
    mean_voltage = np.mean(data,axis=0)
    total_mean = np.mean(data,axis=1)

    #calculate the variance of each trace and the average voltage
    ind_variance = np.mean(np.square(data),axis = 1) - np.mean(data, axis = 1) ** 2
    total_variance = np.mean(np.square(mean_voltage)) - np.mean(mean_voltage)**2

    return mean_voltage, total_mean, ind_variance, total_variance