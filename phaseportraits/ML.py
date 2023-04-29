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
   return  float(-Iion + I), float(0.93 * (winf - w)*np.cosh( (V + 3) / 2 / 19))



phase_diagram = PhasePortrait2D(ML, [[-80,80],[0,1]],
	  dF_args = {'I': 2},
	  MeshDim = 20,
	  Title = 'ML Phase portrait',
	  xlabel = r'Voltage$(\mu V)$',
	  ylabel = 'Recovery variable w',
	  color= 'cool',
)


phase_diagram.add_slider('I', valinit = 2, valinterval=[0,4], valstep= 0.5)
phase_diagram.add_nullclines(xprecision=0.2,yprecision=0.01,xcolor='red',ycolor='green')
phase_diagram.plot()
phase_diagram.ax.legend(loc='right', bbox_to_anchor=(0.9, 1.07),
          ncol=1, fancybox=True, shadow=True)
plt.show()
