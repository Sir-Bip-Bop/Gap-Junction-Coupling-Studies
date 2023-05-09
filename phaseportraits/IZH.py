
                                                                                        
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
	  ylabel = 'Recovery variable u',
	  color= 'cool',
)

phase_diagram.add_slider('I', valinit = 2, valinterval=[0,4], valstep= 0.5)
phase_diagram.add_nullclines(xcolor='red', ycolor='green', yprecision=0.05,xprecision=0.1)

phase_diagram.plot()
phase_diagram.ax.vlines(35,-50,50, color = 'blue', label = 'Peak')
phase_diagram.ax.vlines(-50,-50,50, color = 'black', label = 'Reset')

phase_diagram.ax.legend(loc='right', bbox_to_anchor=(0.9, 1.07),
          ncol=1, fancybox=True, shadow=True)

plt.show()
