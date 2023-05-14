import numpy as np 
from scipy.sparse import dok_matrix 
import scipy as sp 

def compute_Reliability(spike_matrix,t,t_R,dt_og):
    '''
    Computes the Reliability statistic measurement of a set of spike trains
    spike_matrix - matrix containing spike trains, each row corresponds to a diffent neuron
    t - time vector (time points for columns of spike_matrix)
    t_R - time constant of the exponential kernel
    '''

    dt = (t[len(t)-1] - t[0] ) / (len(t) - 1)
    t_R = t_R / dt_og 
    if type(spike_matrix) is not np.ndarray:
        spike_matrix = np.array(spike_matrix.todense())

    joined_matrix = np.sum(spike_matrix,axis=0)
    kernel = t_R *np.exp(-t/t_R)
    
    #Convolve the spike train with kernel
    Convolved_matrix = sp.signal.convolve(joined_matrix,kernel)

    reliability = 1/t[len(t)-1] * np.trapz(np.square(joined_matrix),dx = dt) - (1/t[len(t)-1]*np.trapz(joined_matrix,dx =dt)*np.trapz(joined_matrix,dx =dt))

    return reliability