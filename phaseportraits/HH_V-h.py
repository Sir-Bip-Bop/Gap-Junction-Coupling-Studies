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
plt.rcParams.update({"axes.titlesize": 25})
plt.rcParams.update({"axes.labelsize": 23})


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

#Definition of the functions to integrate, that is the HH equations for the Voltage and the n variable
def HHx(z, h, m=m_array[100000], n = n_array[100000], *, I = 2, vt = -58):
  V= z
  return  float(-30*m*m*m*h*(V-30) - 5*n*n*n*n*(V+90) - 0.1*(V+70) + I)

def HHy(z,V,*, m = 1, I = 2, vt = -58):
  h = z
  return float( 0.128 * (np.exp(-(V-vt-17)/18)) * (1-h) - 4 / (1+np.exp(-(V-vt-40)/5)) * h)

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
        

#Creation of the phase diagram
ins = my_function(m_array, n_array, time[0])
phase_diagram = PhasePortrait2D(ins, [[-80,80],[0,1]],
      dF_args = {'I': 2, 'vt': -58},
      MeshDim = 40,
      Title = 'HH Phase Portrait (V-h)',
      #xlabel = 'Voltage(mV)',
      #ylabel = 'Recovery Variable h',
      color= 'cool',
)

phase_diagram.add_slider('t',valinit=100, valinterval=[0,time[1]], valstep=10)
#phase_diagram.add_nullclines(xprecision=0.01, yprecision=0.001)
fig, ax = phase_diagram.plot()
ax.plot(X,ii, color= 'red', label = 'X - nullcine')
ax.plot(bb,Y, color = 'green', label = 'Y - nullcline')
circle = Trajectory2D(ins, n_points=10000, size=2, Range=[[-80 , 80], [0 , 1]],Fig = fig,Ax=ax,	  Title = ' ',
	  xlabel = 'Voltage(mV)',
	  ylabel = ' ')
circle.initial_position(-60,0)
circle.add_slider('t',valinit=100, valinterval=[0,time[1]], valstep=10)
fig, ax2= circle.plot(color='cool')


plt.show()