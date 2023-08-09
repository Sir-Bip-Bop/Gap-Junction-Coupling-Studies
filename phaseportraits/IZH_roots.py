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

#Definition of the functions to integrate, that is the IZH equation
def IZH(V,u, *, I =2.5):
  return  float(0.019*(V+70)* (V+48.5) - 0.06*u + I), float( 0.5* (-1.3 * (V+70)-u) )

def IZHx(z, u,*,I = 2.5):
  V= z
  return   float(0.019*(V+70)* (V+48.5) - 0.06*u + I)

def IZHy(z,V,*,I = 2.5):
  u = z
  return float( 0.5* (-1.3 * (V+70)-u) )


#Creation of the phase diagram
phase_diagram = PhasePortrait2D(IZH, [[-80,60],[-1.5,1.5]],
	  dF_args = {'I': 2.5},
	  MeshDim = 40,
	  Title = 'IZH Phase Portrait',
	  xlabel = 'Voltage(mV)',
	  ylabel = 'Recovery Variable',
	  color= 'cool',
)


#Obtaining the nullcines in an analytical format - finding the roots
X = []
Y = []
ii = np.linspace(-50,50,100)
bb = np.linspace(-65,40,100)

#for i in ii:
	#solve_x = sp.optimize.root_scalar(IZHx,args = (i), x0= -70, x1 = -50)
	#X.append(solve_x.root)

#for i in bb:
	#solve_y = sp.optimize.root_scalar(IZHy,args = (i), x0= 0, x1 = 1)
	#Y.append(solve_y.root)




#Creation of the plot, the constant lines are representing the threshold and reset values
fig, ax = phase_diagram.plot()

circle = Trajectory2D(IZH, n_points=10000, size=2, Range=[ [-65 , 40],[-0.5 , 0.5]],Fig = fig,Ax=ax,	  Title = 'IZH Phase Portrait',
	  ylabel = 'Voltage(mV)',
	  xlabel = 'Recovery Variable')
circle.initial_position(-65,0)
fig,ax2= circle.plot(color='cool')

#fig.axes.append(ax_test)
#fig.add_axes(ax_test)
#phase_diagram.ax.plot(X,ii, color= 'red', label = 'X - nullcine')
#phase_diagram.ax.plot(bb,Y, color = 'green', label = 'Y - nullcline')
ax.vlines(35,-65,50, color = 'blue', label = 'V Peak')
ax.vlines(-50,-65,50, color = 'black', label = 'V Reset')
custom_lines = [matplotlib.lines.Line2D([0], [0], color='blue', lw=2),
                matplotlib.lines.Line2D([0], [0], color='black', lw=2),
                matplotlib.lines.Line2D([0], [0], color='pink', lw=2),]

ax.legend(custom_lines, ['V Peak', 'V Reset', 'Trajectory'],loc='right', bbox_to_anchor=(0.2, 0.83),ncol=1, frameon=True, prop={'size': 12}, fancybox=True, shadow=False)
plt.show()
