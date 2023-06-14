import numpy as np 
from scipy.sparse import dok_matrix
import scipy as sp

def compute_firing_rate(data,t_final,num_neurons):
    '''
    Computes the firing rate, that is the number of spikes divided by the total time and converted to herzts

    Parameters:
        data (list[float]):
            Spike matrix with 1's in the time steps where the neuron spiked.
        t_final (float):
            Total simulation time

        Returns:
            (float):
                The resulting firing rate
    '''

    if type(data) is not np.ndarray:
        data = np.array(data.todense())
    spike_number = np.zeros(num_neurons)
    firing_rate = np.zeros(num_neurons)
    for i in range(0,num_neurons):
        spike_number[i]=  len((np.argwhere(data[i,:]>0) ).flatten())
        firing_rate[i] = spike_number[i] * 1000 / t_final 
    return np.mean(firing_rate)