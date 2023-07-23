from typing import Any
import matplotlib.pyplot as plt
import matplotlib
from phaseportrait import *
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
    def __init__(self, m, n, dt):
           self._m = m
           self._n = n
           self._t_index = 0
           self.dt = dt
    
    @property
    def m(self):
        return self._m[self.t]
    
    @property
    def n(self):
        return self._n[self.t]
    
    @property
    def t(self):
        return self._t_index
    
    @t.setter
    def t(self, t):
        self._t_index = int(t / self.dt)
    
    def __call__(self, V, h, *, t, I=2, vt=-58):
        self.t = t
        return (
            float(-30*self.m*self.m*self.m*h*(V-30) - 5*self.n*self.n*self.n*self.n*(V+90) - 0.1*(V+70) + I),
            float( 0.128 * (np.exp(-(V-vt-17)/18)) * (1-h) - 4 / (1+np.exp(-(V-vt-40)/5)) * h)
        )


#Loading the needed data from the .txt files 
m_array = np.loadtxt('phaseportraits/m_data.txt')
n_array = np.loadtxt('phaseportraits/n_data.txt')
time = np.loadtxt('phaseportraits/time_data.txt')


#Creation of the phase diagram
ins = my_function(m_array, n_array, time[0])
phase_diagram = PhasePortrait2D(ins, [[-80,80],[0,1]],
      dF_args = {'I': 2, 'vt': -58},
      MeshDim = 12,
      Title = 'HH Phase Portrait (V-h)',
      xlabel = 'Voltage(mV)',
      ylabel = 'Recovery Variable h',
      color= 'cool',
)

phase_diagram.add_slider('t',valinit=0, valinterval=[0,time[1]], valstep=10)
phase_diagram.add_nullclines(xprecision=0.01, yprecision=0.01)
phase_diagram.plot()
plt.show()