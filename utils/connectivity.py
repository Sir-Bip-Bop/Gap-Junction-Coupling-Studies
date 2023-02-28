import numpy as np
from scipy.sparse import dok_matrix 

def create_matrix(connection_type,num_neurons = 0, connection_fraction = 1, seed = 1234 ):
    '''
    Function that creates a matrix as one wants, there are different types. It always returns a sparse matrix,
    unless it's the pair type, in which case, it returns an array
    '''
    if connection_type == 'pair':
        matrix = np.array([ [0 , 1] ,[1, 0] ] )
    elif connection_type == 'electrical':
        matrix = dok_matrix((num_neurons,num_neurons))
        #fill the matrix
    elif connection_type == 'chemical' : 
        matrix = dok_matrix((num_neurons,num_neurons))
        #fill the matrix




    return matrix