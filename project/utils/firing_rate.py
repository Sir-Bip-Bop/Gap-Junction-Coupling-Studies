import numpy as np 
from scipy.sparse import dok_matrix
import scipy as sp

def compute_firing_rate(data,t_final):
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
        
    spike_number=  (np.argwhere(data[0,:]>0) ).flatten()
    return len(spike_number) * 1000 / t_final 