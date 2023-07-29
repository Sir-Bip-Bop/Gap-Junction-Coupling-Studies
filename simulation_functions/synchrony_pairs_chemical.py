import numpy as np 
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.animation as animation
from scipy.sparse import dok_matrix
import multiprocessing as mp
import scipy as sp
import matplotlib.ticker as ticker
import project

def spikelet_fun(strength):
    return strength*  6.26066980e+01 - 1.44182439e-02

#loading up the parameters

params_HH = project.utils.load_HH()
params_LIF = project.utils.load_LIF()
params_IZH = project.utils.load_ISH()
params_ML = project.utils.load_ML( )

#time parameters
t_final = 1500
dt = 0.01

#Other constant parameters
Is = np.array([ [0, 0.3], [0.3, 0]])
V0 = -65
tau = 0.65
y0 = [-65 , -65]
w0 = [0.0, 0.0]
u0 = [0,0]
n0 = 0
m0 = 0
h0 = 0
order = 2

k = [0,0.00105,0.0021,0.00535,0.012,0.025,0.043,0.068,0.104]
k_izh = [0,0.0033,0.007,0.0175,0.04,0.08,0.135,0.21,0.315]
k_ml = [0,0.00066,0.0014,0.00335,0.008, 0.016,0.027,0.04,0.0614]

#initial random seed, change
np.random.seed(1234)
n_measurements = 15

chi_HH = np.zeros(n_measurements)
chi_IF = np.zeros(n_measurements)
chi_IZH = np.zeros(n_measurements)
chi_ML = np.zeros(n_measurements)

rel_HH = np.zeros(n_measurements)
rel_IF = np.zeros(n_measurements)
rel_IZH = np.zeros(n_measurements)
rel_ML = np.zeros(n_measurements)

rossum_HH = np.zeros((n_measurements,2))
rossum_IF = np.zeros((n_measurements,2))
rossum_IZH = np.zeros((n_measurements,2))
rossum_ML = np.zeros((n_measurements,2))

I_HH = np.zeros((int(t_final/dt),2))
I_IF = np.zeros((int(t_final/dt),2))
I_IZH = np.zeros((int(t_final/dt),2))
I_ML = np.zeros((int(t_final/dt),2))

chi_HH_k = np.zeros((len(k),2))
chi_IF_k = np.zeros((len(k),2))
chi_IZH_k = np.zeros((len(k_izh),2))
chi_ML_k = np.zeros((len(k_ml),2))

chi_rel_HH_k = np.zeros((len(k),2))
chi_rel_IF_k = np.zeros((len(k),2))
chi_rel_IZH_k = np.zeros((len(k_izh),2))
chi_rel_ML_k = np.zeros((len(k_ml),2))

rel_HH_k = np.zeros((len(k),2))
rel_IF_k = np.zeros((len(k),2))
rel_IZH_k = np.zeros((len(k_izh),2))
rel_ML_k = np.zeros((len(k_ml),2))


rossum_HH_k = np.zeros((len(k),2))
rossum_IF_k = np.zeros((len(k),2))
rossum_IZH_k = np.zeros((len(k_izh),2))
rossum_ML_k = np.zeros((len(k_ml),2))

firing_rate_HH = np.zeros(n_measurements)
firing_rate_IF = np.zeros(n_measurements)
firing_rate_IZH = np.zeros(n_measurements)
firing_rate_ML = np.zeros(n_measurements)

firing_rate_HH_k = np.zeros(len(k))
firing_rate_IF_k = np.zeros(len(k))
firing_rate_IZH_k = np.zeros(len(k_izh))
firing_rate_ML_k = np.zeros(len(k_ml))

save_HH_k = {}
matrix_HH_k = {}
save_IF_k = {}
matrix_IF_k = {}
save_IZH_k = {}
matrix_IZH_k = {}
save_ML_k = {}
matrix_ML_k = {}

manager = mp.Manager()
return_dict = manager.dict()
jobs = []

t = np.linspace(0,t_final,int(t_final/dt))
t_R = 0.5
t_Re = 0.5

for j in range(0,len(k)):

    for i in range(0,n_measurements):

        #generation of random intensities
        x1 = np.random.normal(0,5,int(t_final/dt))
        x2 = np.random.normal(0,5,int(t_final/dt))
        x3 = np.random.normal(0,1.5,int(t_final/dt))
        x4 = np.random.normal(0,1.5,int(t_final/dt))

        I_HH[:,0] = 2.5*(1+ 1*x1)
        I_HH[:,1] = 2.5*(1+ 1*x2)
        I_IF[:,0] = 2.5*(1+ 1*x1)
        I_IF[:,1] = 2.5*(1+ 1*x2)
        I_IZH[:,0] = 2.5*(1+ 1*x1)
        I_IZH[:,1] = 2.5*(1+ 1*x2)
        I_ML[:,0] = 2.5*(1+ 1*x1)
        I_ML[:,1] = 2.5*(1+ 1*x2)

        #simulating the models
        proc_HH = mp.Process(target = project.models.HH_Neuron_Pairs, args= (dt,t_final,order,[V0,V0],[n0,n0],[m0,m0],[h0,h0],*params_HH,I_HH,Is,k[j],tau,return_dict) )
        jobs.append(proc_HH)    
        proc_HH.start()

        proc_IF = mp.Process(target = project.models.LIF_Neuron_Pairs, args = (dt, t_final,order,y0,*params_LIF,I_IF,Is,k[j],tau,spikelet_fun(k[j]),return_dict))
        jobs.append(proc_IF)
        proc_IF.start()
    
        proc_IZH = mp.Process(target = project.models.IZH_Neuron_Pairs, args =(dt,t_final,order,y0,u0,I_IZH,Is,*params_IZH,k_izh[j],tau,return_dict))
        jobs.append(proc_IZH)
        proc_IZH.start()

        proc_ML = mp.Process(target= project.models.ML_Neuron_Pairs, args= (dt,t_final,2,y0,w0,*params_ML,I_ML,Is,k_ml[j],tau,return_dict))
        jobs.append(proc_ML)
        proc_ML.start()


        for proc in jobs:
            proc.join()
    
        dataHH_k = return_dict['data_HH']
        completeHH_k = return_dict['Y_HH']
        matrixHH_k = return_dict['Matrix_HH']
        dataIF_k = return_dict['data_IF']
        completeIF_k = return_dict['Y_IF']
        matrixIF_k = return_dict['Matrix_IF']
        dataIZH_k = return_dict['data_IZH']
        completeIZH_k = return_dict['Y_IZH']
        matrixIZH_k = return_dict['Matrix_IZH']
        dataML_k = return_dict['data_ML']
        completeML_k = return_dict['Y_ML']
        matrixML_k = return_dict['Matrix_ML']



        #computing chi 
        chi_HH[i] = project.utils.compute_chi(dataHH_k.T) 
        chi_IF[i] = project.utils.compute_chi(dataIF_k.T) 
        chi_IZH[i] = project.utils.compute_chi(dataIZH_k.T) 
        chi_ML[i] = project.utils.compute_chi(dataML_k.T) 

            #computing reliability
        rel_HH[i] = project.utils.compute_Reliability(matrixHH_k,t,t_Re,2)[0] 
        rel_IF[i] = project.utils.compute_Reliability(matrixIF_k,t,t_Re,2)[0] 
        rel_IZH[i] = project.utils.compute_Reliability(matrixIZH_k,t,t_Re,2)[0]
        rel_ML[i] = project.utils.compute_Reliability(matrixML_k,t,t_Re,2)[0] 


        #computing firing rate
        firing_rate_HH[i] = project.utils.compute_firing_rate(matrixHH_k,t_final,2)
        firing_rate_IF[i] = project.utils.compute_firing_rate(matrixIF_k,t_final,2)
        firing_rate_IZH[i] = project.utils.compute_firing_rate(matrixIZH_k,t_final,2)
        firing_rate_ML[i] = project.utils.compute_firing_rate(matrixML_k,t_final,2)

        #computing van_Rossum distance
        rossum_HH[i] = project.utils.compute_van_Rossum_distance(matrixHH_k,t,t_R).flatten()[1:3] / firing_rate_HH[i]
        rossum_IF[i] = project.utils.compute_van_Rossum_distance(matrixIF_k,t,t_R).flatten()[1:3] / firing_rate_IF[i]
        rossum_IZH[i] = project.utils.compute_van_Rossum_distance(matrixIZH_k,t,t_R).flatten()[1:3] / firing_rate_IZH[i] 
        rossum_ML[i] = project.utils.compute_van_Rossum_distance(matrixML_k,t,t_R).flatten()[1:3] / firing_rate_ML[i]

    save_HH_k[j] = dataHH_k
    matrix_HH_k[j] = matrixHH_k
    save_IF_k[j] = dataIF_k
    matrix_IF_k[j] =  matrixIF_k
    save_IZH_k[j] = dataIZH_k
    matrix_IZH_k[j] = matrixIZH_k
    save_ML_k[j] = dataML_k
    matrix_ML_k[j] = matrixML_k

    chi_HH_k[j] = np.mean(chi_HH),np.std(chi_HH)
    chi_IF_k[j] = np.mean(chi_IF),np.std(chi_IF)
    chi_IZH_k[j]= np.mean(chi_IZH),np.std(chi_IZH)
    chi_ML_k[j] = np.mean(chi_ML),np.std(chi_ML)

    rel_HH_k[j] = np.mean(rel_HH),np.std(rel_HH)
    rel_IF_k[j] = np.mean(rel_IF),np.std(rel_IF)
    rel_IZH_k[j] = np.mean(rel_IZH),np.std(rel_IZH)
    rel_ML_k[j] = np.mean(rel_ML),np.std(rel_ML)

    firing_rate_HH_k[j] = np.mean(firing_rate_HH)
    firing_rate_IF_k[j] = np.mean(firing_rate_IF)
    firing_rate_IZH_k[j] = np.mean(firing_rate_IZH)
    firing_rate_ML_k[j] = np.mean(firing_rate_ML)

    rossum_HH_k[j] = np.mean(rossum_HH[:,0]), np.std(rossum_HH[:,0])

    rossum_IF_k[j] = np.mean(rossum_IF[:,0]), np.std(rossum_IF[:,0])

    rossum_IZH_k[j] = np.mean(rossum_IZH[:,0]), np.std(rossum_IZH[:,0])

    rossum_ML_k[j] = np.mean(rossum_ML[:,0]), np.std(rossum_ML[:,0])

#Saving the data

np.savetxt('simulation_functions/saved_data/sync_k_chemical/chi_HH_k.txt',chi_HH_k)
np.savetxt('simulation_functions/saved_data/sync_k_chemical/chi_IF_k.txt',chi_IF_k)
np.savetxt('simulation_functions/saved_data/sync_k_chemical/chi_IZH_k.txt',chi_IZH_k)
np.savetxt('simulation_functions/saved_data/sync_k_chemical/chi_ML_k.txt',chi_ML_k)

np.savetxt('simulation_functions/saved_data/sync_k_chemical/rel_HH_k.txt',rel_HH_k)
np.savetxt('simulation_functions/saved_data/sync_k_chemical/rel_IF_k.txt',rel_IF_k)
np.savetxt('simulation_functions/saved_data/sync_k_chemical/rel_IZH_k.txt',rel_IZH_k)
np.savetxt('simulation_functions/saved_data/sync_k_chemical/rel_ML_k.txt',rel_ML_k)

np.savetxt('simulation_functions/saved_data/sync_k_chemical/firing_rate_HH_k.txt',firing_rate_HH_k)
np.savetxt('simulation_functions/saved_data/sync_k_chemical/firing_rate_IF_k.txt',firing_rate_IF_k)
np.savetxt('simulation_functions/saved_data/sync_k_chemical/firing_rate_IZH_k.txt',firing_rate_IZH_k)
np.savetxt('simulation_functions/saved_data/sync_k_chemical/firing_rate_ML_k.txt',firing_rate_ML_k)

np.savetxt('simulation_functions/saved_data/sync_k_chemical/rossum_HH_k.txt',rossum_HH_k)
np.savetxt('simulation_functions/saved_data/sync_k_chemical/rossum_IF_k.txt',rossum_IF_k)
np.savetxt('simulation_functions/saved_data/sync_k_chemical/rossum_IZH_k.txt',rossum_IZH_k)
np.savetxt('simulation_functions/saved_data/sync_k_chemical/rossum_ML_k.txt',rossum_ML_k)


