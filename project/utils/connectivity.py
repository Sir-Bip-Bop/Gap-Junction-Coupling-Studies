import numpy as np
from scipy.sparse import csr_matrix
import scipy.stats as stats

class Neuron:
    def __init__(self,index,x_cord,y_cord,z_cord):
        self.index = index 
        self.x_cord = x_cord
        self.y_cord = y_cord
        self.z_cord = z_cord


def create_matrix(connection_type,num_neurons = 0, synapse_type = 'chemical',connection_fraction = 0.5, barabasi_initial_neurons = 3, barabasi_links = 3,spatial_initial_neurons = 3, spatial_links = 2, spatial_initial_config = 'triangle',spatial_xmin = 0, spatial_xmax = 10, spatial_ymin = 0, spatial_ymax = 10, spatial_zmin = 0, spatial_zmax = 10, seed_ini = 1234 ):
    '''
    Creates a connectivity matrix, which characteristics can be defined while calling the function.

    Parameters:
        connection_type (str):
            'pair': there are only two neurons, connected between each other
            'random': electrical synapse matrix - symmetrical
        num_neurons (int, optional):
            the number of neurons of the network, default value is 0
        synapse_type (str, optional):
            'electrical' for a symmetrical matrix
            'chemical', default value, not symmetrical matrix
        connection_fraction (float, optional):
            the proportion of synapses in the network, default value: 0.5
        seed_ini (int, optional):
            seed of the random number generator, default value 1234

    Returns:
        matrix (tuple[tuple[int,int]] | sparse_matrix):
            for 'pair' it returns an np.array. for the other cases, it resturns a sparse matrix that contains 1's representing connections between neurons.
        seed_ini (int):
            returns the seed used
    '''

    if connection_type == 'pair':
        matrix = np.array([ [0 , 1] ,[1, 0] ] )

    elif connection_type == 'random':
        np.random.seed(seed_ini)

        #Matrix of random numbers between 0 and 1. The heavyside function makes the variables either 0 or 1
        matrix = np.heaviside(connection_fraction - stats.uniform.rvs(size=(num_neurons,num_neurons)),0)

    elif connection_type == 'barabasi' :
        matrix = np.ones((barabasi_initial_neurons,barabasi_initial_neurons))
        #empty the diagonal
        np.fill_diagonal(matrix,0)
        for i in range(barabasi_initial_neurons, num_neurons):

            #Compute the connectivity to obtain the probablities
            connectivity = np.sum(matrix,axis=0)

            #compute the probabilty
            max_connectivity = np.sum(connectivity)
            probability = connectivity / max_connectivity

            #We add a new node
            matrix = np.lib.pad(matrix, ((0,1),(0,1)), 'constant', constant_values=(0))
            #generate a number and try to fill that node
            links = 0
            while links < barabasi_links:
                index = int(np.random.uniform(0,1) * i)
                prob_index = np.random.uniform(0,1) * max_connectivity
                if prob_index <= probability[index] and matrix[index][i] == 0:
                    matrix[index][i] = matrix[i][index] = 1 
                    links = links + 1

    elif connection_type == 'spatial':
        matrix = np.ones((spatial_initial_neurons,spatial_initial_neurons))
        neurons = {}
        if spatial_initial_config == 'triangle':
            for i in range(0,spatial_initial_neurons):
               neurons[i] =  Neuron(i,np.random.uniform(spatial_xmin,spatial_xmax),np.random.uniform(spatial_ymin,spatial_ymax),np.random.uniform(spatial_zmin,spatial_zmax))
        #empty the diagonal
        np.fill_diagonal(matrix,0)
        for i in range(spatial_initial_neurons, num_neurons):

            #We add a new node
            matrix = np.lib.pad(matrix, ((0,1),(0,1)), 'constant', constant_values=(0))

            #add a new neuron
            neurons[i] = Neuron(i,np.random.uniform(spatial_xmin,spatial_xmax),np.random.uniform(spatial_ymin,spatial_ymax),np.random.uniform(spatial_zmin,spatial_zmax))
            distances = np.zeros(i)
            #Compute the distances to obtain the probablities
            for j in range(0,i-1):
                distances[j] = np.sqrt((neurons[j].x_cord - neurons[i].x_cord) * (neurons[j].x_cord - neurons[i].x_cord) + (neurons[j].y_cord - neurons[i].y_cord)*(neurons[j].y_cord - neurons[i].y_cord) + (neurons[j].z_cord - neurons[i].z_cord)*(neurons[j].z_cord - neurons[i].z_cord))

            #compute the probabilty
            max_distance = np.sum(distances)
            probability = distances / max_distance
            #generate a number and try to fill that node
            links = 0
            while links < spatial_links:
                index = int(np.random.uniform(0,1) * i)
                prob_index = np.random.uniform(0,1) * max_distance
                if prob_index <= probability[index] and matrix[index][i] == 0:
                    matrix[index][i] = matrix[i][index] = 1 
                    links = links + 1

    if synapse_type == 'electrical':
        #make the matrix symmetrical
        i_lower = np.tril_indices(len(matrix),-1)
        matrix[i_lower] = matrix.T[i_lower]

    #empty the diagonal
    np.fill_diagonal(matrix,0)

    ratio = len((np.argwhere(matrix>0)).flatten()) /( num_neurons*num_neurons- num_neurons) / 2

    matrix = csr_matrix(matrix)

 


    return matrix, seed_ini, ratio