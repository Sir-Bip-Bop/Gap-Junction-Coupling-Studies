
                                                                                        
from phaseportrait import *
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import scipy as sp

plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,12)

def HH(V,m,*, n = 0.5, h = 0.5, I = 2, vt = -58):
  return  float(-30*m*m*m*h*(V-30) - 5*n*n*n*n*(V+90) - 0.1*(V+70) + I),float( -0.32 * (V-vt-13) / (np.exp(-(V-vt-13)/4)-1) * (1-m) - 0.28 * (V-vt-40) / (np.exp((V-vt-40)/5) -1) * m)


phase_diagram = PhasePortrait2D(HH, [[-80,20],[0,1]],
	  dF_args = {'n': 0.2, 'h': 0.3, 'I': 2, 'vt': -58},
	  MeshDim = 20,
	  Title = 'HH Phase portrait',
	  xlabel = r'Voltage$(\mu V)$',
	  ylabel = 'recovery variable',
	  color= 'cool',
)

#phase_diagram.add_slider('C',valinit=1, valinterval=[0,2], valstep=0.2)
phase_diagram.add_slider('n', valinit = 0.5, valinterval=[0,1],valstep=0.05)
phase_diagram.add_slider('h', valinit=0.5, valinterval=[0,1],valstep=0.05)
phase_diagram.add_slider('I', valinit = 2, valinterval=[0,4], valstep= 0.5)
#phase_diagram.add_nullclines(xcolor='black', ycolor='green', yprecision=0.05,xprecision=0.1)


def HHx(z, m,*, n = 0.5, h = 0.5, I = 2, vt = -58):
  V= z
  return  float(-30*m*m*m*h*(V-30) - 5*n*n*n*n*(V+90) - 0.1*(V+70) + I)

def HHy(z,V,*, n = 0.5, h = 0.5, I = 2, vt = -58):
  m = z
  return float( -0.32 * (V-vt-13) / (np.exp(-(V-vt-13)/4)-1) * (1-m) - 0.28 * (V-vt-40) / (np.exp((V-vt-40)/5) -1) * m)

X = []
Y = []
ii = np.linspace(0,1,1000)
bb = np.linspace(-80,20,1000)

for i in ii:
	solve_x = sp.optimize.root_scalar(HHx,args=(i), x0 = -70, x1 = -50)
	X.append(solve_x.root)

for i in bb:
	solve_y = sp.optimize.root_scalar(HHy,args=(i), x0 = 0, x1 = 0.1	)
	Y.append(solve_y.root)

phase_diagram.plot()
phase_diagram.ax.plot(X,ii, color= 'red')
phase_diagram.ax.plot(bb,Y, color = 'green')

plt.show()
