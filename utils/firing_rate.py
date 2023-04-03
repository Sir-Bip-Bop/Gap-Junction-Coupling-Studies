import numpy as np 
from scipy.sparse import dok_matrix
import scipy as sp

def compute_firing_rate(data,dt,t_final):
    spike_number=  (np.argwhere(np.array(data.todense())[0,:]>0) * dt).flatten()
    return len(spike_number) / 2 * 1000 / t_final 