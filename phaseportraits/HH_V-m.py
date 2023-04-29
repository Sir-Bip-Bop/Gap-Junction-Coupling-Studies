from typing import Any                                                                      
from phaseportrait import *
import matplotlib.pyplot as plt
import matplotlib
import scienceplots
import numpy as np
import scipy as sp

plt.style.use('science')
plt.rcParams["figure.figsize"] = (8,8)

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
    
h_array = np.loadtxt('phaseportraits/h.txt')
n_array = np.loadtxt('phaseportraits/n.txt')
time = np.loadtxt('phaseportraits/time.txt')

ins = my_function(h_array, n_array, time[0])

phase_diagram = PhasePortrait2D(ins, [[-80,80],[0,1]],
	  dF_args = {'I': 2, 'vt': -58},
	  MeshDim = 12,
	  Title = 'HH Phase portrait (V-m)',
	  xlabel = r'Voltage$(\mu V)$',
	  ylabel = 'Recovery variable m',
	  color= 'cool',
)

phase_diagram.add_slider('t',valinit=0, valinterval=[0,time[1]], valstep=10)
phase_diagram.add_nullclines(xprecision=0.06, yprecision=0.06)
phase_diagram.plot()


plt.show()
