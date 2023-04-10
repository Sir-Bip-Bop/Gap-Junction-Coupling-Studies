from phaseportrait import *
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import scipy as sp

plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,12)

def ML(V,w, *, I =2):
   minf = 0.5 * (1 + np.tanh( ((V + 3 )/ 20)))
   Iion = 2 * minf * (V - 90) + 2*w * (V+110) + 0.085 * (V + 75)
   winf = 0.5 * (1 + np.tanh( (V + 3) / 19))
   return  float(-Iion + I), float(0.93 * (winf - 2)*np.cosh( (V + 3) / 2 / 19))



phase_diagram = PhasePortrait2D(ML, [[-80,60],[0,1]],
	  dF_args = {'I': 2},
	  MeshDim = 20,
	  Title = 'ML Phase portrait',
	  xlabel = r'Voltage$(\mu V)$',
	  ylabel = 'recovery variable',
	  color= 'cool',
)

#phase_diagram.add_slider('C',valinit=1, valinterval=[0,2], valstep=0.2)
phase_diagram.add_slider('I', valinit = 2, valinterval=[0,4], valstep= 0.5)
#phase_diagram.add_nullclines(xcolor='black', ycolor='green', yprecision=0.05,xprecision=0.1)


def MLx(z, w,*,I = 2):
  V= z
  return    float(- 2 * 0.5 * (1 + np.tanh( ((V + 3 )/ 20))) * (V - 90) + 2*w * (V+110) + 0.085 * (V + 75)+ I)

def MLy(z,V,*,I = 2):
  w = z
  winf = 0.5 * (1 + np.tanh( (V + 3) / 19))
  return  float(0.93 * (winf - 2)*np.cosh( (V + 3) / 2 / 19))
X = []
Y = []
ii = np.linspace(0,1,100)
bb = np.linspace(-80,20,100)

for i in ii:
	solve_x = sp.optimize.fsolve(MLx,-70,args=i)
	X.append(solve_x)

#for i in bb:
#	solve_y = sp.optimize.fsolve(MLy,0,args=i)
#	Y.append(solve_y)

phase_diagram.plot()
phase_diagram.ax.plot(X,ii, color= 'red', label = 'X - nullcine')
#phase_diagram.ax.plot(bb,Y, color = 'green', label = 'Y - nullcline')
phase_diagram.ax.legend(loc='right', bbox_to_anchor=(0.9, 1.03),
          ncol=1, fancybox=True, shadow=True)
plt.show()
