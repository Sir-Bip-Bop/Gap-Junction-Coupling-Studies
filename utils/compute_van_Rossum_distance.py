import numpy as np

def compute_van_Rossum_distance(spike_matrix,t,t_R):
    '''
    Computes the van Rossum distance between a set of spike trains
    spike matrix - matrix containing spike train, each row corresponds to
    different neuron/spike train
    t - time vector (time points for columns of spike_matrix)
    t_R - time constant of exponential kernel
    '''
    N = len(spike_matrix[0,:])
    dt = (t[len(t)-1] - t[0] ) / (len(t)-1)
    van_Rossum = np.zeros(N)
    waveforms = np.zeros_like(spike_matrix)

    #construct kernel
    kernel = np.exp(-t/t_R)

    #Convolve spike trains with kernel
    # (2D convolution iwth 1 as column convolution, i.e. no convolution)

    for j in range(0,N):
        waveforms[j,:] = np.convolve(spike_matrix[j,:], kernel, 'valid') #here the original converts from sparse to full
    
    #compute van Rossum distance between each pair of spike trains
    for j in range(0,N):
        waveform_difference = waveforms - waveforms[j,:]
        van_Rossum[j,:] = np.sqrt(np.trapz(dt,waveform_difference**2/t_R))
    return van_Rossum    
