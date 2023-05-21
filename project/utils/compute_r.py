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

    dt = (t[len(t)-1] - t[0] ) / len(t)
    t_R = t_R / dt_og 
    if type(spike_matrix) is not np.ndarray:
        spike_matrix = np.array(spike_matrix.todense())
    num_spikes =  len((np.argwhere(np.array(spike_matrix)[0,:]>0)).flatten())
    num_spikes = num_spikes + len((np.argwhere(np.array(spike_matrix)[1,:]>0)).flatten())


    joined_matrix = np.sum(spike_matrix,axis=0)
    kernel = 1/ t_R *np.exp(-t / t_R)
    
    #Convolve the spike train with kernel
    Convolved_matrix = sp.signal.convolve(joined_matrix,kernel)[0:len(spike_matrix[0,:])]
    #Convolved_matrix = joined_matrix

    reliability = 1 / t[len(t) - 1] * np.trapz(np.square(Convolved_matrix), dx = dt) - np.square( 1 / t[len(t) - 1] * np.trapz(Convolved_matrix,dx =dt))


    reliability_max= ( 4 * num_spikes / t_R / ( 2 * t[len(t)-1] ) - 4 * num_spikes * num_spikes / (t[len(t)-1] * t[len(t)-1]) )
    #reliability_max = 1
    return reliability/reliability_max, Convolved_matrix