import numpy as np
from scipy.sparse import csr_matrix
import scipy.stats as stats

def create_matrix(connection_type,num_neurons = 0, connection_fraction_e = 0.5, connection_fraction_c = 0.5,seed_ini = 1234 ):
    '''
    Function that creates a matrix as one wants, there are different types. It always returns a sparse matrix,
    unless it's the pair type, in which case, it returns an array
    '''

    if connection_type == 'pair':

        matrix = np.array([ [0 , 1] ,[1, 0] ] )
    elif connection_type == 'electrical':
        np.random.seed(seed_ini)
        #Matrix of random numbers between 0 and 1. The heavyside function makes the variables either 0 or 1
        matrix = np.heaviside(connection_fraction_e - stats.uniform.rvs(size=(num_neurons,num_neurons)),0)
        #make the matrix symmetrical
        i_lower = np.tril_indices(len(matrix),-1)
        matrix[i_lower] = matrix.T[i_lower]
        #empty the diagonal
        np.fill_diagonal(matrix,0)
        matrix = csr_matrix(matrix)

    elif connection_type == 'chemical' : 
        np.random.seed(seed_ini + 37)
        #Matrix of random numbers between 0 and 1. The heavyside function makes the variables either 0 or 1
        matrix = np.heaviside(connection_fraction_c - stats.uniform.rvs(size=(num_neurons,num_neurons)),0)
        #remove the entries from the diagonal
        np.fill_diagonal(matrix,0)
        matrix = csr_matrix(matrix)




    return matrix, seed_ini