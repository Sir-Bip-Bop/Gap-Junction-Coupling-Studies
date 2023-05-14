import numpy as np 
from scipy.sparse import dok_matrix
import scipy as sp

def compute_van_Rossum_distance(spike_matrix,t,t_R,dt_og):
    '''
    Computes the van Rossum distance between a set of spike trains
    spike matrix - matrix containing spike train, each row corresponds to a different neuron/spike train
    t - time vector (time points for columns of spike_matrix)
    t_R - time constant of exponential kernel
    '''
    dt = (t[len(t)-1] - t[0] ) / (len(t)-1)
    t_R = t_R / dt_og
    if type(spike_matrix) is not np.ndarray:
        spike_matrix = np.array(spike_matrix.todense())
    N = len(spike_matrix[:,0])  
    van_Rossum = np.zeros((N,N))
    #construct kernel
    kernel = np.exp(-t/t_R)
    test = sp.signal.convolve(spike_matrix[0,:],kernel)

    waveforms = np.zeros((N,len(test)))

    #Convolve spike trains with kernel
    # (2D convolution iwth 1 as column convolution, i.e. no convolution)
    for j in range(0,N):
        waveforms[j,:] = sp.signal.convolve(spike_matrix[j,:], kernel) 
        #waveforms[j,:] = np.convolve(spike_matrix[j,:], kernel) 

    #compute van Rossum distance between each pair of spike trains
    for j in range(0,N):
        waveform_difference = waveforms - waveforms[j,:]
        van_Rossum[j,:] = np.sqrt(np.trapz(np.square(waveform_difference)/t_R,dx=dt))
    return van_Rossum, waveforms[0,:], waveforms[1,:]

import numpy as np 
from scipy.sparse import dok_matrix
import scipy as sp

def compute_van_Rossum_distance_2(spike_matrix,t,t_R,dt_og):
    '''
    Computes the van Rossum distance between a set of spike trains
    spike matrix - matrix containing spike train, each row corresponds to
    different neuron/spike train
    t - time vector (time points for columns of spike_matrix)
    t_R - time constant of exponential kernel
    '''
    dt = (t[len(t)-1] - t[0] ) / (len(t)-1)
    t_R = t_R / dt_og
    if type(spike_matrix) is not np.ndarray:
        spike_matrix = np.array(spike_matrix.todense())
    N = len(spike_matrix[:,0])  
    van_Rossum = np.zeros((N,N))
    #construct kernel
    kernel = np.exp(-t/t_R)
    test = sp.signal.convolve(spike_matrix[0,:],kernel)

    waveforms = np.zeros((N,len(test)))

    #Convolve spike trains with kernel
    # (2D convolution iwth 1 as column convolution, i.e. no convolution)
    for j in range(0,N):
        waveforms[j,:] = sp.signal.convolve(spike_matrix[j,:], kernel) 
        #waveforms[j,:] = np.convolve(spike_matrix[j,:], kernel) 

    #compute van Rossum distance between each pair of spike trains
    for j in range(0,N):
        waveform_difference = waveforms - waveforms[j,:]
        van_Rossum[j,:] = np.sqrt(np.trapz(np.square(waveform_difference)/t_R,dx=dt))
    return van_Rossum
