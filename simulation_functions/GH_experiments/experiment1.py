import numpy as np 
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.animation as animation
import project

params_HH = project.utils.load_HH()
k = 0.012 #strength of the gap junction coupling
Isyn = [ [0, 0.05] , [0, 0.05]] #delta peak of the chemical synapse
tau = 1 #time constant for the chemical coupling equations

#filtering order
order = 2

#Initial conditions of the HH model important quantities
V0 = -70
n0 =0.2
m0 = 0.1
h0 = 0.6 

#time constants
dt = 0.01
t_final = 300

#Input intensities for experiment 2A
I1 = np.zeros((int(t_final/dt),2))
I2 = np.zeros((int(t_final/dt),2))

for i in range(4999,15000):
    I1[i,0] =1
    I2[i,0] = -1

#data2aHH, completeHH = rk_simplemodel(dt,t_final,2,[V0,V0],[n0,n0],[m0,m0],[h0,h0],gna,gk,gl,Ena,Ek,El,C,I1,Isyn,k,tau)
data2aHH, completeHH_2a, a = project.models.HH_Neuron_Pairs(dt,t_final,order,[V0,V0],[n0,n0],[m0,m0],[h0,h0],*params_HH,I1,Isyn,k,tau)
data2, completeHH_2a_2, a = project.models.HH_Neuron_Pairs(dt,t_final,order,[V0,V0],[n0,n0],[m0,m0],[h0,h0],*params_HH,I2,Isyn,k,tau)

params_LIF = project.utils.load_LIF()
#Same as before
k = 0.012
tau = 1
Isyn = [[0, 0.05], [0.05, 0]]

#filtering order
order = 2

#Initial conditions for the LIF
y0 = [-70, -70]

#Time conditions
t_final = 300
dt = 0.01 

data2aIF, completeIF_2a, matrix = project.models.LIF_Neuron_Pairs(dt,t_final,order,y0,*params_LIF,I1,Isyn,k,tau,1)
data2b, complete_IF_2a, matrix = project.models.LIF_Neuron_Pairs(dt,t_final,order,y0,*params_LIF,I2,Isyn,k,tau,1)

params_IZH = project.utils.load_ISH()

k_izh = 0.04
dt = 0.01
t_final = 300
Isyn = [[0, 0.05], [0.05, 0]]
Is = np.array(Isyn)
tau = 1

I1 = np.zeros((int(t_final/dt),2))
I2 = np.zeros((int(t_final/dt),2))

for i in range(4999,15000):
    I1[i,0] = 1.5
    I2[i,0] = -1.5
order = 2
y0 = [-70,-70]
u0 = [0.0,0.0]


data2Ish, completeIsh1, a = project.models.IZH_Neuron_Pairs(dt,t_final,order,y0,u0,I1,Is,*params_IZH,k_izh,tau)
data2ish, completeIsh2, a = project.models.IZH_Neuron_Pairs(dt,t_final,order,y0,u0,I2,Is,*params_IZH,k_izh,tau)

params_ML = project.utils.load_ML()

k_ML = 0.008
tau = 1
Isyn = [[0, 0.05], [0.05, 0]]
Isyn = np.array(Isyn)

#filtering order
order = 2

#Initial conditions for the ML
y0 = [ -71.7061740390072, -71.7061740390072]
w0 = [0.0007223855976593603, 0.0007223855976593603]

#Time conditions
t_final = 300
dt = 0.01 

#Input intensities for experiment 2A
I1 = np.zeros((int(t_final/dt),2))
I2 = np.zeros((int(t_final/dt),2))


for i in range(4999,15000):
    I1[i,0] = 0.5 #changes the amplitude
    I2[i,0] = -0.5

data2aML, completeML, a = project.models.ML_Neuron_Pairs(dt,t_final,2,y0,w0,*params_ML,I1,Isyn,k_ML,tau)
data2ML, completeML2, a = project.models.ML_Neuron_Pairs(dt,t_final,2,y0,w0,*params_ML,I2,Isyn,k_ML,tau)

#Saving the data
np.savetxt('simulation_functions/saved_data/GH_experiment_1/data_HH_positive.txt',data2aHH)
np.savetxt('simulation_functions/saved_data/GH_experiment_1/data_HH_negative.txt',data2)
np.savetxt('simulation_functions/saved_data/GH_experiment_1/data_IF_positive.txt',data2aIF)
np.savetxt('simulation_functions/saved_data/GH_experiment_1/data_IF_negative.txt',data2b)
np.savetxt('simulation_functions/saved_data/GH_experiment_1/data_IZH_positive.txt',data2Ish)
np.savetxt('simulation_functions/saved_data/GH_experiment_1/data_IZH_negative.txt',data2ish)
np.savetxt('simulation_functions/saved_data/GH_experiment_1/data_ML_positive.txt',data2aML)
np.savetxt('simulation_functions/saved_data/GH_experiment_1/data_ML_negative.txt',data2ML)