from typing import Any                                                                                 
from phaseportrait import *
import matplotlib.pyplot as plt
import scienceplots
import matplotlib
import numpy as np
import scipy as sp


#General plot style used in the project, and size definition
plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,12)
#plt.rcParams.update({"axes.grid" : True})
plt.rcParams.update({"axes.titlesize": 25})
plt.rcParams.update({"axes.labelsize": 23})



#Definition of the function with some extra spice in order to ensure our experiment works
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


#Loading the needed data from the .txt files 
m_array = np.loadtxt('phaseportraits/m_data.txt')
h_array = np.loadtxt('phaseportraits/h_data.txt')
time = np.loadtxt('phaseportraits/time_data.txt')


#Definition of the functions to integrate, that is the HH equations for the Voltage and the n variable
def HHx(z, n, m = m_array[65000], h = h_array[65000], *, I = 2, vt = -58):
  V= z
  return  float(-30*m*m*m*h*(V-30) - 5*n*n*n*n*(V+90) - 0.1*(V+70) + I)

def HHy(z,V,*, m = 1, h = 1, I = 2, vt = -58):
  n = z
  return float( -0.032 * (V-vt-15) / (np.exp(-(V-vt-15)/5)-1) * (1-n) - 0.5 * (np.exp(-(V-vt-10)/40)) * n)

X = []
Y = []
ii = np.linspace(0,1,100)
bb = np.linspace(-80,80,100)

for i in ii:
	solve_x = sp.optimize.root_scalar(HHx,args = (i), x0= -70, x1 = -50)
	X.append(solve_x.root)

for i in bb:
	solve_y = sp.optimize.root_scalar(HHy,args = (i), x0= 0, x1 = 1)
	Y.append(solve_y.root)
        
#print(X)
#Definition of the functions to integrate, that is the HH equations for the Voltage and the h variable

#Creation of the phase diagram
ins = my_function(m_array, h_array, time[0])
phase_diagram = PhasePortrait2D(ins, [[-80,80],[0,1]],
	  dF_args = {'I': 2, 'vt': -58},
	  MeshDim = 40,
	  Title = 'HH Phase Portrait (V-n)',
	  xlabel = 'Voltage(mV)',
	  ylabel = 'Recovery Variable n',
	  color= 'cool',
)

phase_diagram.add_slider('t',valinit=65, valinterval=[0,time[1]], valstep=10)
#phase_diagram.add_nullclines( yprecision=0.001)

fig, ax = phase_diagram.plot()
ax.plot(X,ii, color= 'red', label = 'X - nullcine')
ax.plot(bb,Y, color = 'green', label = 'Y - nullcline')
circle = Trajectory2D(ins, n_points=10000, size=2, Range=[[-80 , 80], [0 , 1]],Fig = fig,Ax=ax,	  Title = 'HH Phase Portrait (V-n)',
	  xlabel = ' ',
	  ylabel = 'Recovery Variable')
circle.initial_position(-60,0)
circle.add_slider('t',valinit=65, valinterval=[0,time[1]], valstep=10)
fig, ax2= circle.plot(color='cool')

plt.show()
