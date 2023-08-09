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
def ML(V,w, *, I =90):
   minf = 0.5 * (1 + np.tanh( ((V + 1.2 )/ 18)))
   Iion = 4.4 * minf * (V - 120) + 8*w * (V+84) + 2 * (V + 60)
   winf = 0.5 * (1 + np.tanh( (V - 2) / 30))
   return  float(-Iion + I)/20, float(0.04 * (winf - w)*np.cosh( (V - 2) / 2 / 30))

def MLx(z, w,*,I = 90):
  V= z
  minf = 0.5 * (1 + np.tanh( ((V + 1.2 )/ 18)))
  Iion = 4.4 * minf * (V - 120) + 8*w * (V+84) + 2 * (V + 60)
  return     float(-Iion + I)/20

def MLy(z,V,*,I = 90):
  w = z
  winf = 0.5 * (1 + np.tanh( (V - 2) / 30))
  return  float(0.04 * (winf - w)*np.cosh( (V - 2) / 2 / 30))


#Creation of the phase diagram
phase_diagram = PhasePortrait2D(ML, [[-80,60],[0.0,0.6]],
	  dF_args = {'I': 90},
	  MeshDim = 40,
	  Title = 'ML Phase portrait Hopf',
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
phase_diagram.add_nullclines(xcolor='red',xprecision=0.01,show='x')
fig, ax = phase_diagram.plot()

circle = Trajectory2D(ML, n_points=50000, size=2, Range=[[-80 , 60], [0.0 , 0.6]],Fig = fig,Ax=ax,	  Title = 'ML Phase portrait Hopf',
	  xlabel = 'Voltage(mV)',
	  ylabel = 'Recovery Variable')
circle.initial_position(-60,0)
fig, ax2= circle.plot(color='cool')
#phase_diagram.ax.plot(X,ii, color= 'red', label = 'X - nullcine')
phase_diagram.ax.plot(bb,Y, color = 'green', label = 'Y - ullcline')
custom_lines = [matplotlib.lines.Line2D([0], [0], color='red', lw=2),
                matplotlib.lines.Line2D([0], [0], color='green', lw=2),]

phase_diagram.ax.legend(custom_lines, ['X - Nullcline', 'Y - Nullcline'],loc='right', bbox_to_anchor=(0.2, 0.83),ncol=1, frameon=True, prop={'size': 12}, fancybox=True, shadow=False)

plt.show()
