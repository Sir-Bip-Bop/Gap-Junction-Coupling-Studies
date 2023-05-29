import numpy as np 
from scipy.sparse import dok_matrix
import scipy as sp

def compute_van_Rossum_distance(spike_matrix,t,t_R,traces = False):
    '''
    Computes the Van Rossum Distance, a synchrony measurement that computes the area underneath the curve that is the difference between the convolved spike trains of two neurons.

    Parameters:
        spike_matrix (tuple[tuple[int,int]] | sparse_matrix):
            matrix containing spike trains, each row contains a diffent neuron.
        t (list[float]):
            time array, time points of the simulation
        t_R (float):
            Time constant
        traces (bool, optional):
            False, default: the function does not return the convolved spike trains
            True: the function returns the convolved spike trains

    Returns:
        van_Rossum (float):
            The computed value of Van Rossum distance
        waveforms[0,:] (tuple[float]):
            The convolved spike train of the first neuron.
        waveforms[1,:] (tuple[float]):
            The convoled spike train of the second neuron.
    '''

    #compute the time step of the simulation
    dt = (t[len(t)-1] - t[0] ) / len(t) 

    #we need to work with a np.ndarray for the convolution, if it is a sparse_matrix, change it to that type
    if type(spike_matrix) is not np.ndarray:
        spike_matrix = np.array(spike_matrix.todense())

    #construct kernel and prepare the spike trains to be convolved
    N = len(spike_matrix[:,0])  
    van_Rossum = np.zeros((N,N))
    kernel = np.exp(-t/t_R)
    test = sp.signal.convolve(spike_matrix[0,:],kernel)[0:len(spike_matrix[0,:])]
    waveforms = np.zeros((N,len(test)))

    #Convolve spike trains with kernel (2D convolution iwth 1 as column convolution, i.e. no convolution)
    for j in range(0,N):
        waveforms[j,:] = sp.signal.convolve(spike_matrix[j,:], kernel)[0:len(spike_matrix[j,:])]

    #compute van Rossum distance between each pair of spike trains
    for j in range(0,N):
        waveform_difference = waveforms - waveforms[j,:]
        van_Rossum[j,:] = np.sqrt(np.trapz(np.square(waveform_difference)/t_R,dx=dt))

    if traces:
        return van_Rossum, waveforms[0,:], waveforms[1,:]
    else:
        return van_Rossum

def compute_van_Rossum_distance_2(spike_matrix,t,t_R):
    '''
    Computes the Van Rossum Distance, a synchrony measurement that computes the area underneath the curve that is the difference between the convolved spike trains of two neurons.

    Parameters:
        spike_matrix (tuple[tuple[int,int]] | sparse_matrix):
            matrix containing spike trains, each row contains a diffent neuron.
        t (list[float]):
            time array, time points of the simulation
        t_R (float):
            Time constant

    Returns:
        van_Rossum (float):
            The computed value of Van Rossum distance
    '''

    #compute the time step of the simulation
    dt = (t[len(t)-1] - t[0] ) / len(t) 

    #we need to work with a np.ndarray for the convolution, if it is a sparse_matrix, change it to that type
    if type(spike_matrix) is not np.ndarray:
        spike_matrix = np.array(spike_matrix.todense())

    #construct kernel and prepare the spike trains to be convolved
    N = len(spike_matrix[:,0])  
    van_Rossum = np.zeros((N,N))
    kernel = np.exp(-t/t_R)
    test = sp.signal.convolve(spike_matrix[0,:],kernel)[0:len(spike_matrix[0,:])]
    waveforms = np.zeros((N,len(test)))

    #Convolve spike trains with kernel (2D convolution iwth 1 as column convolution, i.e. no convolution)
    for j in range(0,N):
        waveforms[j,:] = sp.signal.convolve(spike_matrix[j,:], kernel)[0:len(spike_matrix[j,:])]

    #compute van Rossum distance between each pair of spike trains
    for j in range(0,N):
        waveform_difference = waveforms - waveforms[j,:]
        van_Rossum[j,:] = np.sqrt(np.trapz(np.square(waveform_difference)/t_R,dx=dt))
    return van_Rossum
