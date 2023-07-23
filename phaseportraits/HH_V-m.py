from typing import Any                                                                      
from phaseportrait import *
import matplotlib.pyplot as plt
import matplotlib
import scienceplots
import numpy as np
import scipy as sp


#General plot style used in the project, and size definition
plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,12)
#plt.rcParams.update({"axes.grid" : True})
plt.rcParams.update({"axes.titlesize": 17})
plt.rcParams.update({"axes.labelsize": 15})



#Definition of the function with some extra spice in order to ensure our experiment works
class my_function:
    def __init__(self, h, n, dt):
        self._h = h
        self._n = n
        self._t_index = 0 
        self.dt = dt 
    @property
    def h(self):
        return self._h[self.t]
    @property
    def n(self):
        return self._n[self.t]
    @property
    def t(self):
        return self._t_index 
    @t.setter 
    def t(self, t):
        self._t_index = int(t / self.dt)  
    def __call__(self, V, m, *, t, I = 2, vt = -58):
        self.t = t
        return ( 
            float(-30*m*m*m*self.h*(V-30) - 5*self.n*self.n*self.n*self.n*(V+90) - 0.1*(V+70) + I),
			float( -0.32 * (V-vt-13) / (np.exp(-(V-vt-13)/4)-1) * (1-m) - 0.28 * (V-vt-40) / (np.exp((V-vt-40)/5) -1) * m)
		)


#Loading the needed data from the .txt files 
h_array = np.loadtxt('phaseportraits/h_data.txt')
n_array = np.loadtxt('phaseportraits/n_data.txt')
time = np.loadtxt('phaseportraits/time_data.txt')


#Creation of the phase diagram
ins = my_function(h_array, n_array, time[0])
phase_diagram = PhasePortrait2D(ins, [[-80,80],[0,1]],
	  dF_args = {'I': 2, 'vt': -58},
	  MeshDim = 12,
	  Title = 'HH Phase Portrait (V-m)',
	  xlabel = 'Voltage(mV)',
	  ylabel = 'Recovery Variable m',
	  color= 'cool',
)

phase_diagram.add_slider('t',valinit=0, valinterval=[0,time[1]], valstep=10)
phase_diagram.add_nullclines(xprecision=0.06, yprecision=0.06)
phase_diagram.plot()
plt.show()
