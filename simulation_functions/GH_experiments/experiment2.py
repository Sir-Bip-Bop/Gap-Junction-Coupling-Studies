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
Isyn = [ [0, 0.05] , [0, 0.05]] #delta peak of the chemical synapse
tau = 1 #time constant for the chemical coupling equations

#filtering order
order = 2

#Initial conditions of the HH model important quantities
V0 = -70
n0 =0.2
m0 = 0.1
h0 = 0.6 
y0 = [-70, -70]
u0 = [0.0,0.0]
y0_ML = [ -71.7061740390072, -71.7061740390072]
w0 = [0.0007223855976593603, 0.0007223855976593603]

I = np.zeros((int(t_final/dt),2))
t = np.linspace(0,t_final,int(t_final/dt))
I[:,0] = np.sin(2 * np.pi * 0.04 * t) #t in ms, so f in Hz 
np.savetxt('simulation_functions/saved_data/GH_experiment_2/current_HH.txt',I)
data4aHH, completeHH_4a, a = project.models.HH_Neuron_Pairs(dt,t_final,order,[V0,V0],[n0,n0],[m0,m0],[h0,h0],*params_HH,I,Isyn,k,tau)
I = np.zeros(( int(t_final/dt), 2))
t = np.linspace(0,t_final, int(t_final/dt))
I[:,0] = 1 + np.sin(2* np.pi * 0.04 * t)
np.savetxt('simulation_functions/saved_data/GH_experiment_2/current_IF.txt',I)
data4aIF, complete_IF_4a, matrix = project.models.LIF_Neuron_Pairs(dt,t_final,order,y0,*params_LIF,I,Isyn,k,tau,1)
I = np.zeros(( int(t_final/dt), 2))
t = np.linspace(0,t_final, int(t_final/dt))
I[:,0] =  -2.5 +  2*np.sin(2* np.pi * 0.04 * t)
np.savetxt('simulation_functions/saved_data/GH_experiment_2/current_IZH.txt',I)
data4aIsh, complete_Ish_4a, a = project.models.IZH_Neuron_Pairs(dt,t_final,order,y0,u0,I,Isyn,*params_IZH,k_izh,tau)
I = np.zeros(( int(t_final/dt), 2))
t = np.linspace(0,t_final, int(t_final/dt))
I[:,0] =  -1 + np.sin(2* np.pi * 0.04 * t)
np.savetxt('simulation_functions/saved_data/GH_experiment_2/current_ML.txt',I)
data4aML, complete_ML_4a, a = project.models.ML_Neuron_Pairs(dt,t_final,2,y0,w0,*params_ML,I,Isyn,k_ML,tau)

np.savetxt('simulation_functions/saved_data/GH_experiment_2/data_HH.txt',data4aHH)
np.savetxt('simulation_functions/saved_data/GH_experiment_2/data_IF.txt',data4aIF)
np.savetxt('simulation_functions/saved_data/GH_experiment_2/data_IZH.txt',data4aIsh)
np.savetxt('simulation_functions/saved_data/GH_experiment_2/data_ML.txt',data4aML)




