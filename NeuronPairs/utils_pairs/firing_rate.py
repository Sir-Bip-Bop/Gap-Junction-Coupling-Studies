import numpy as np 
from scipy.sparse import dok_matrix
import scipy as sp

def compute_firing_rate(data,dt,t_final):
    if type(data) is not np.ndarray:
        data = np.array(data.todense())
    spike_number=  (np.argwhere(data[0,:]>0) * dt).flatten()
    return len(spike_number) * 1000 / t_final 