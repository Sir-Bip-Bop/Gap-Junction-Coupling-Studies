
                                                                                        
from phaseportrait import *
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import scipy as sp

plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,12)

def HHx(z, n, m, h, *, I = 2, vt = -58):
  V= z
  return  float(-30*m*m*m*h*(V-30) - 5*n*n*n*n*(V+90) - 0.1*(V+70) + I)

def HHy(z,V,*, m = 1, h = 1, I = 2, vt = -58):
  n = z
  return float( -0.032 * (V-vt-15) / (np.exp(-(V-vt-15)/5)-1) * (1-n) - 0.5 * (np.exp(-(V-vt-10)/40)) * n)

m_values = np.linspace(0,1,50)
h_values = np.linspace(0,1,50)
Roots_x = np.zeros((50,50))
Roots_y = np.zeros((50,50))

for p in range(len(m_values)):
    for q in range(len(h_values)):
      X = []
      Y = []
      ii = np.linspace(0,1,1000)
      for i in ii:
        solve_x = sp.optimize.root_scalar(HHx,args=(i,m_values[p],h_values[q]), x0 = -70, x1 = -50)
        X.append(solve_x.root)

      X = np.array(X)

      for i in X:
        solve_y = sp.optimize.root_scalar(HHy,args=(i), x0 = 0, x1 = 0.1)
        Y.append(solve_y.root)

      Y = np.array(Y)


      idx = np.argwhere(np.diff(np.sign(ii- Y))).flatten()
      Roots_x[p,q] = X[idx]
      Roots_y[p,q] = Y[idx]





plt.scatter(Roots_x,Roots_y, marker = 'o',color = 'black', label = 'Nullcline intersection for varying m and h')
plt.grid()
plt.title('HH Phase portrait (V-h) - Intersection of nullclines')
plt.xlabel (r'Voltage$(\mu V)$')
plt.ylabel('recovery variable n')
plt.legend()
plt.show()
