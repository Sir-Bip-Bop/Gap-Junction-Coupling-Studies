
                                                                                        
from phaseportrait import *
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import scipy as sp

plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,12)

def HH(V,n,*, m = 1, h = 1, I = 2, vt = -58):
  return  float(-30*m*m*m*h*(V-30) - 5*n*n*n*n*(V+90) - 0.1*(V+70) + I),float( -0.032 * (V-vt-15) / (np.exp(-(V-vt-15)/5)-1) * (1-n) - 0.5 * (np.exp(-(V-vt-10)/40)) * n)


phase_diagram = PhasePortrait2D(HH, [[-80,80],[0,1]],
	  dF_args = {'m': 0.2, 'h': 0.3, 'I': 2, 'vt': -58},
	  MeshDim = 20,
	  Title = 'HH Phase portrait (V-n)',
	  xlabel = r'Voltage$(\mu V)$',
	  ylabel = 'recovery variable n',
	  color= 'cool',
)

#phase_diagram.add_slider('C',valinit=1, valinterval=[0,2], valstep=0.2)
phase_diagram.add_slider('m', valinit = 0.5, valinterval=[0,1],valstep=0.05)
phase_diagram.add_slider('h', valinit=0.5, valinterval=[0,1],valstep=0.05)
phase_diagram.add_slider('I', valinit = 2, valinterval=[0,4], valstep= 0.5)
phase_diagram.add_nullclines(xcolor='black', ycolor='green', yprecision=0.005,xprecision=0.05)


def HHx(z, n,*, m = 2, h = 3, I = 2, vt = -58):
  V= z
  return  float(-30*m*m*m*h*(V-30) - 5*n*n*n*n*(V+90) - 0.1*(V+70) + I)

def HHy(z,V,*, m = 1, h = 1, I = 2, vt = -58):
  n = z
  return float( -0.032 * (V-vt-13) / (np.exp(-(V-vt-15)/5)-1) * (1-n) - 0.5 * (np.exp(-(V-vt-10)/40)) * n)

X = []
Y = []
ii = np.linspace(0,1,100)
bb = np.linspace(-80,20,100)

for i in ii:
	solve_x = sp.optimize.fsolve(HHx,-70,args=i)
	X.append(solve_x)

for i in bb:
	solve_y = sp.optimize.fsolve(HHy,0,args=i)
	Y.append(solve_y)

phase_diagram.plot()
#phase_diagram.plot(X,ii, color= 'red')
#phase_diagram.plot(bb,Y, color = 'green')

plt.show()
