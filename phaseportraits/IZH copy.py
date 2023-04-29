
                                                                                        
from phaseportrait import *
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import scipy as sp

plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,12)

def IZH(V,u, *, I =2):
  return  float(1/5*(V+90)* (V+48.5) - 0.06*u + I), float( 0.5* (-1.3 * (V+70)-u) )


phase_diagram = PhasePortrait2D(IZH, [[-65,40],[-50,50]],
	  dF_args = {'I': 2},
	  MeshDim = 20,
	  Title = 'IZH Phase portrait',
	  xlabel = r'Voltage$(\mu V)$',
	  ylabel = 'recovery variable',
	  color= 'cool',
)

#phase_diagram.add_slider('C',valinit=1, valinterval=[0,2], valstep=0.2)
phase_diagram.add_slider('I', valinit = 2, valinterval=[0,4], valstep= 0.5)
#phase_diagram.add_nullclines(xcolor='black', ycolor='green', yprecision=0.05,xprecision=0.1)

def IZHx(z, u,*,I = 2):
  V= z
  return   float(1/5*(V+90)* (V+48.5) - 0.06*u + I)

def IZHy(z,V,*,I = 2):
  u = z
  return float( 0.5* (-1.3 * (V+70)-u) )
X = []
Y = []
ii = np.linspace(-50,50,100)
bb = np.linspace(-65,40,100)

for i in ii:
	solve_x = sp.optimize.root_scalar(IZHx,args = (i), x0= -70, x1 = -50)
	X.append(solve_x.root)

for i in bb:
	solve_y = sp.optimize.root_scalar(IZHy,args = (i), x0= 0, x1 = 1)
	Y.append(solve_y.root)

phase_diagram.plot()
phase_diagram.ax.plot(X,ii, color= 'red', label = 'X - nullcine')
phase_diagram.ax.plot(bb,Y, color = 'green', label = 'Y - nullcline')
phase_diagram.ax.vlines(35,-50,50, color = 'blue', label = 'Peak')
phase_diagram.ax.vlines(-50,-50,50, color = 'black', label = 'Reset')

phase_diagram.ax.legend(loc='right', bbox_to_anchor=(0.9, 1.07),
          ncol=1, fancybox=True, shadow=True)

plt.show()
