from typing import Any                                                                                 
from phaseportrait import *
import matplotlib.pyplot as plt
import scienceplots
import matplotlib
import numpy as np
import scipy as sp

plt.style.use('science')
plt.rcParams["figure.figsize"] = (8,8)

class my_function:
   def __init__(self, m , h, dt):
      self._m = m 
      self._h = h 
      self._t_index = 0
      self.dt = dt
   @property
   def m(self):
      return self._m[self.t]
   @property
   def h(self):
      return self._h[self.t]
   @property
   def t(self):
      return self._t_index
   @t.setter
   def t(self, t):
      self._t_index =  int(t / self.dt)
   def __call__(self, V, n, *, t, I = 2, vt = - 58):
      self.t = t 
      return (
         float(-30*self.m*self.m*self.m*self.h*(V-30) - 5*n*n*n*n*(V+90) - 0.1*(V+70) + I),
         float( -0.032 * (V-vt-15) / (np.exp(-(V-vt-15)/5)-1) * (1-n) - 0.5 * (np.exp(-(V-vt-10)/40)) * n)
      )

m_array = np.loadtxt('phaseportraits/m.txt')
h_array = np.loadtxt('phaseportraits/h.txt')
time = np.loadtxt('phaseportraits/time.txt')

ins = my_function(m_array, h_array, time[0])


phase_diagram = PhasePortrait2D(ins, [[-80,80],[0,1]],
	  dF_args = {'I': 2, 'vt': -58},
	  MeshDim = 12,
	  Title = 'HH Phase portrait (V-n)',
	  xlabel = r'Voltage$(\mu V)$',
	  ylabel = 'Recovery variable n',
	  color= 'cool',
)

phase_diagram.add_slider('t',valinit=0, valinterval=[0,time[1]], valstep=10)
phase_diagram.add_nullclines(xprecision=0.01, yprecision=0.01)
phase_diagram.plot()

