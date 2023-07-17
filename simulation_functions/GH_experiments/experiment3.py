import numpy as np 
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.animation as animation
import project

params_HH = project.utils.load_HH()
params_LIF = project.utils.load_LIF()
params_IZH = project.utils.load_ISH()
params_ML = project.utils.load_ML()
k_ML = 0.008
k_izh = 0.04
k = 0.012 #strength of the gap junction coupling
dt = 0.01 
t_final = 100
Isyn = np.array([ [0, 0.00] , [0, 0.00]]) #delta peak of the chemical synapse
tau = 1 #time constant for the chemical coupling equations

#filtering order
order = 5
V0 = -65 
n0 = 0.2
m0 = 0.1
h0 =0.6
y0 = [-65, -65]
u0 = [0.0,0.0]
y0_ML = [ -71.7061740390072, -71.7061740390072]
w0 = [0.0007223855976593603, 0.0007223855976593603]
I = np.ones((int(t_final/dt),2))
I[:,0] = 4 * I[:,0]
I[:,1] = 0 * I[:,1]

data4B_HH, completeHH_4B, a = project.models.HH_Neuron_Pairs(dt,t_final,order,[V0,V0],[n0,n0],[m0,m0],[h0,h0],*params_HH,I,Isyn,k,tau)

I = np.zeros((int(t_final/dt),2))
I[:,0] = 2.37
data4B_IF, complete_IF_4B, matrix = project.models.LIF_Neuron_Pairs(dt, t_final,order,y0,*params_LIF,I,Isyn,k,tau,1)
I = np.zeros((int(t_final/dt),2))
I[:,0] = 2.0
data4B_Ish, complete_Ish_4B, a = project.models.IZH_Neuron_Pairs(dt,t_final,order,y0,u0,I,Isyn,*params_IZH,k_izh,tau,)
I[:,0] = 1.7
data4B_ML, complete_ML_4B, a = project.models.ML_Neuron_Pairs(dt,t_final,2,y0,w0,*params_ML,I,Isyn,k_ML,tau)

np.savetxt('simulation_functions/saved_data/GH_experiment_3/data_HH.txt',data4B_HH)
np.savetxt('simulation_functions/saved_data/GH_experiment_3/data_IF.txt',data4B_IF)
np.savetxt('simulation_functions/saved_data/GH_experiment_3/data_IZH.txt',data4B_Ish)
np.savetxt('simulation_functions/saved_data/GH_experiment_3/data_ML.txt',data4B_ML)


