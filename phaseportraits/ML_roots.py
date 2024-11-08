from phaseportrait import *
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import scipy as sp
import matplotlib

#General plot style used in the project, and size definition
plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,12)
#plt.rcParams.update({"axes.grid" : True})
plt.rcParams.update({"axes.titlesize": 17})
plt.rcParams.update({"axes.labelsize": 15})

#Definition of the functions to integrate, that is the ML equations
def ML(V,w, *, I =2):
   minf = 0.5 * (1 + np.tanh( ((V + 3 )/ 20)))
   Iion = 2 * minf * (V - 90) + 2*w * (V+110) + 0.085 * (V + 75)
   winf = 0.5 * (1 + np.tanh( (V + 3) / 19))
   return  float(-Iion + I), float(0.93 * (winf - w)*np.cosh( (V + 3) / 2 / 19))

def MLx(z, w,*,I = 2):
  V= z
  minf = 0.5 * (1 + np.tanh( ((V + 3 )/ 20)))
  Iion = 2 * minf * (V - 90) + 2*w * (V+110) + 0.085 * (V + 75)
  return     float(-Iion + I)

def MLy(z,V,*,I = 2):
  w = z
  winf = 0.5 * (1 + np.tanh( (V + 3) / 19))
  return  float(0.93 * (winf - w)*np.cosh( (V + 3) / 2 / 19))


#Creation of the phase diagram
phase_diagram = PhasePortrait2D(ML, [[-80,60],[0,1]],
	  dF_args = {'I': 2},
	  MeshDim = 40,
	  Title = 'ML Phase portrait',
	  xlabel = 'Voltage(mV)',
	  ylabel = 'Recovery Variable',
	  color= 'cool',
)


#Obtaining the nullcines in an analytical format - finding the roots
X = []
Y = []
ii = np.linspace(0,1,100)
bb = np.linspace(-80,60,100)

#for i in ii:
  #solve_x = sp.optimize.root_scalar(MLx,args = (i), x0= 0, x1 = 3)
  #X.append(solve_x.root)

for i in bb:
	solve_y = sp.optimize.root_scalar(MLy,args= (i), x0 = 0, x1 =1)
	Y.append(solve_y.root)



#Creation of the plot, the constant lines are representing the threshold and reset values
phase_diagram.add_nullclines(xcolor='red',xprecision=0.2,show='x')
fig, ax = phase_diagram.plot()

circle = Trajectory2D(ML, n_points=10000, size=2, Range=[[-80 , 60], [0 , 1]],Fig = fig,Ax=ax,	  Title = 'ML Phase portrait',
	  xlabel = 'Voltage(mV)',
	  ylabel = 'Recovery Variable')
circle.initial_position(-60,0)
fig, ax2= circle.plot(color='cool')

#phase_diagram.ax.plot(X,ii, color= 'red', label = 'X - nullcine')
ax.plot(bb,Y, color = 'green', label = 'Y - Nullcline')
custom_lines = [matplotlib.lines.Line2D([0], [0], color='red', lw=2),
                matplotlib.lines.Line2D([0], [0], color='green', lw=2),
                matplotlib.lines.Line2D([0], [0], color='blue', lw=2),]

ax.legend(custom_lines, ['X - Nullcline', 'Y - Nullcline', 'Trajectory'],loc='right', bbox_to_anchor=(0.2, 0.83),ncol=1, frameon=True, prop={'size': 12}, fancybox=True, shadow=False)

plt.show()
