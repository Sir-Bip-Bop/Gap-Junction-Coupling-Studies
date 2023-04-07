
                                                                                        
from phaseportrait import *
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import scipy as sp

plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,12)

def IZH(V,u, *, I =2):
  return  float(1/5*(V+90)* (V+48.5) - 0.06*u + I), float( 0.5* (-1.3 * (V+70)-u) )


phase_diagram = PhasePortrait2D(IZH, [[-80,20],[0,1]],
	  dF_args = {'I': 2},
	  MeshDim = 20,
	  Title = 'IZH Phase portrait',
	  xlabel = r'Voltage$(\mu V)$',
	  ylabel = 'recovery variable',
	  color= 'cool',
)

#phase_diagram.add_slider('C',valinit=1, valinterval=[0,2], valstep=0.2)
phase_diagram.add_slider('I', valinit = 2, valinterval=[0,4], valstep= 0.5)
phase_diagram.add_nullclines(xcolor='black', ycolor='green', yprecision=0.05,xprecision=0.1)



phase_diagram.plot()
#phase_diagram.plot(X,ii, color= 'red')
#phase_diagram.plot(bb,Y, color = 'green')

plt.show()
