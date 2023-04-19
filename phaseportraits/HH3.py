
                                                                                        
from phaseportrait import *
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import scipy as sp

plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,12)

def HH(V,h,*, m = 1, n = 1, I = 2, vt = -58):
  return  float(-30*m*m*m*h*(V-30) - 5*n*n*n*n*(V+90) - 0.1*(V+70) + I),float( 0.128 * (np.exp(-(V-vt-17)/18)) * (1-h) - 4 / (1+np.exp(-(V-vt-40)/5)) * h)


phase_diagram = PhasePortrait2D(HH, [[-80,80],[0,1]],
	  dF_args = {'m': 0.2, 'n': 0.3, 'I': 2, 'vt': -58},
	  MeshDim = 20,
	  Title = 'HH Phase portrait (V-h)',
	  xlabel = r'Voltage$(\mu V)$',
	  ylabel = 'recovery variable h',
	  color= 'cool',
)

#phase_diagram.add_slider('C',valinit=1, valinterval=[0,2], valstep=0.2)
phase_diagram.add_slider('m', valinit = 0.5, valinterval=[0,1],valstep=0.05)
phase_diagram.add_slider('n', valinit=0.5, valinterval=[0,1],valstep=0.05)
phase_diagram.add_slider('I', valinit = 2, valinterval=[0,4], valstep= 0.5)
#phase_diagram.add_nullclines(xcolor='black', ycolor='green', yprecision=0.005,xprecision=0.05)


def HHx(z, h,*, m = 0.5, n = 0.5, I = 2, vt = -58):
  V= z
  return  float(-30*m*m*m*h*(V-30) - 5*n*n*n*n*(V+90) - 0.1*(V+70) + I)

def HHy(z,V,*, m = 1, n = 1, I = 2, vt = -58):
  h = z
  return float( 0.128 * (np.exp(-(V-vt-17)/18)) * (1-h) - 4 / (1+np.exp(-(V-vt-40)/5)) * h)

X = []
Y = []
I = []
ii = np.linspace(0,1,100)
bb = np.linspace(-80,80,100)

for i in ii:
	solve_x = sp.optimize.root_scalar(HHx,args=(i), x0 = -70, x1 = -50)
	X.append(solve_x.root)

for i in bb:
	solve_y = sp.optimize.root_scalar(HHy,args=(i), x0 = 0, x1 = 0.1)
	Y.append(solve_y.root)

X = np.array(X)
Y = np.array(Y)

dist = np.zeros((int(len(ii)),int(len(Y))))
for i in range(len(ii)):
    for j in range(len(Y)):
              diff_y = ii[i] - Y[j]
              diff_x = X[i] - bb[j]
              dist[i,j] = abs(np.sqrt(diff_y*diff_y + diff_x/20*diff_x/20))
              
idx = np.unravel_index(np.argmin(dist, axis=None), dist.shape)  

phase_diagram.plot()
phase_diagram.ax.plot(X,ii, color= 'red', marker= 'o' )
phase_diagram.ax.plot(bb,Y, color = 'green', marker = 'o')
phase_diagram.ax.plot(bb[idx[1]],Y[idx[1]], marker = 'o', markersize = 20,color = 'black')
phase_diagram.ax.plot(X[idx[0]],ii[idx[0]], marker = 'o', markersize = 20,color = 'black')

plt.show()
