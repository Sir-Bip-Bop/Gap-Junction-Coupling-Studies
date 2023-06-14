import numpy as np 
from scipy.sparse import dok_matrix 
import scipy as sp 

def compute_Reliability(spike_matrix,t,t_R,num_neurons):
    '''
    Computes the value of Reliability, a synchrony measurement that computes the variance of a convoluted spike train that is the sum of the spike trains of each neuron.

    Parameters:
        spike_matrix (tuple[tuple[int,int]] | sparse_matrix):
            matrix containing spike trains, each row contains a diffent neuron.
        t (list[float]):
            time array, time points of the simulation
        t_R (float):
            Time constant

    Returns:
        reliability / reliability_max (float):
            The computed value of reliability, normalised so (theorically) is between 0 and 1.
        Convolved_matrix (tuple[float]):
            The convolved spike train.
    '''

    #compute the time step of the simulation
    dt = (t[len(t)-1] - t[0] ) / len(t)

    #we need to work with a np.ndarray for the convolution, if it is a sparse_matrix, change it to that type
    if type(spike_matrix) is not np.ndarray:
        spike_matrix = np.array(spike_matrix.todense())

    #Compute the number of spikes
    num_spikes = 0 
    for i in range(0,num_neurons):
        num_spikes =  num_spikes + len((np.argwhere(np.array(spike_matrix)[i,:]>0)).flatten())
    num_spikes = num_spikes / num_neurons

    #Compute the kernel, and convolve the sum spike train with it
    joined_matrix = np.sum(spike_matrix,axis=0)
    kernel = 1/ t_R *np.exp(-t / t_R)
    Convolved_matrix = sp.signal.convolve(joined_matrix,kernel)[0:len(spike_matrix[0,:])]
    #Convolved_matrix = joined_matrix

    #compute the measurements of reliability
    reliability = 1 / t[len(t) - 1] * np.trapz(np.square(Convolved_matrix), dx = dt) - np.square( 1 / t[len(t) - 1] * np.trapz(Convolved_matrix,dx =dt))
    reliability_max= ( num_neurons * num_neurons * num_spikes / t_R / ( 2 * t[len(t)-1] ) - num_neurons * num_neurons * num_spikes * num_spikes / (t[len(t)-1] * t[len(t)-1]) )
    #reliability_max = 1
    return reliability/reliability_max, Convolved_matrix