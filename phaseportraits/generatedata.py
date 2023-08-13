from typing import Any
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import project


#Simulation: We are runnign a simple simulation of the HH model, in order to obtain the real-time values of the different variables of the process
params_HH = project.utils.load_HH()

#Initial conditions & the intensity
v0 = -65 
n0 = 0
m0 = 0
h0 = 0
y0 = [v0,n0,m0,h0]
Isyn = np.zeros(([2 , 2 ]))


#variables related to the numerical integration of the problem
dt = 0.001
t_final = 300
I = np.zeros((int(t_final/dt),2))
I[:,0] = 2.5

time = np.array([dt,t_final])

data_HH, complete_HH, a = project.models.HH_Neuron_Pairs(dt,t_final,2,[v0,v0],[n0,n0],[m0,m0],[h0,h0],*params_HH,I,Isyn,0,1)


#After running the simulation, store all the obtained data into txt
V_array = complete_HH[:,0]
m_array = complete_HH[:,1]
n_array = complete_HH[:,2]
h_array = complete_HH[:,3]
t = np.linspace(0,t_final,int(t_final/dt))
plt.plot(t,data_HH[:,0])
plt.xlim(64,66)
plt.show()
np.savetxt('phaseportraits/v_data.txt',V_array)
np.savetxt('phaseportraits/m_data.txt',m_array)
np.savetxt('phaseportraits/n_data.txt',n_array)
np.savetxt('phaseportraits/h_data.txt',h_array)
np.savetxt('phaseportraits/time_data.txt',time)