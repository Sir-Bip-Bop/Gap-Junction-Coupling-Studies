import numpy as np 
import matplotlib.pyplot as plt 
from old.referencecode.HH import HH_RK4
from old.referencecode.LIF import LIF_RK4

def Read_Parameters(file_name):
    '''This function takes a file name, the one containing the parameters for the Excitatory and Inhibitory cases
    and creates a variable for each parameter. 
    
    For the function to properly work the txt file must be set up in the following way:
    variable_name variable_value
    variable_name2 variable_value2

    The function does not print the variable names, one needs to remember which are they, or uncomment the line: 'print(str(p[0]))'
    '''
    params = []
    with open(file_name, 'r') as data:

        for line in data:
            p = line.split()
            #print(str(p[0]))
            #globals()[str(p[0])] = float(p[1])
            params.append(float(p[1]))
    return params

#There are differences between the Excitatory and the inhibitory parameters, let's first set up a condition to define one or the other 
#in a couple of if conditions

case = 'old/Inhibitory.txt' #or 'Inhibitory.txt' for the inhibitory case 
params = Read_Parameters(case)



#Initial conditions & the intensity
v0 = -65 
n0 = 0
m0 = 0
h0 = 0
y0 = [v0,n0,m0,h0]
I = 2.5

#variables related to the numerical integration of the problem
dt = 0.001
t_final = 100



#Simulate HH
HH_sol = HH_RK4(dt,t_final,y0,I,*params)

#Simulate LIF
LIF_sol = LIF_RK4(dt,t_final,v0,I,*params)

#Plot

Nsteps = int(t_final/dt)
t = np.linspace(0,t_final,Nsteps)

plt.plot(t,HH_sol[:,0])
plt.plot(t,LIF_sol)
plt.xlim(10,50)
plt.ylim(-70,30)
plt.show()