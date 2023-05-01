                                                                                      
from phaseportrait import *
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import scipy as sp

plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,12)

def HHx(z, h, m , n ,*, I = 2, vt = -58):
  V= z
  return  float(-30*m*m*m*h*(V-30) - 5*n*n*n*n*(V+90) - 0.1*(V+70) + I)

def HHy(z,V,*, m = 1, n = 1, I = 2, vt = -58):
  h = z
  return float( 0.128 * (np.exp(-(V-vt-17)/18)) * (1-h) - 4 / (1+np.exp(-(V-vt-40)/5)) * h)



m_values = np.linspace(0,1,50)
n_values = np.linspace(0,1,50)
Roots_x = np.zeros((50,50))
Roots_y = np.zeros((50,50))

for p in range(len(m_values)):
    for q in range(len(n_values)):
      X = []
      Y = []
      ii = np.linspace(0,1,1000)
      for i in ii:
        solve_x = sp.optimize.root_scalar(HHx,args=(i,m_values[p],n_values[q]), x0 = -70, x1 = -50)
        X.append(solve_x.root)

      X = np.array(X)

      for i in X:
        solve_y = sp.optimize.root_scalar(HHy,args=(i), x0 = 0, x1 = 0.1)
        Y.append(solve_y.root)

      Y = np.array(Y)


      idx = np.argwhere(np.diff(np.sign(ii- Y))).flatten()
      Roots_x[p,q] = X[idx]
      Roots_y[p,q] = Y[idx]





plt.scatter(Roots_x,Roots_y, marker = 'o',color = 'black', label = 'm and n values rise equally from 0.1 to 1.0')
plt.grid()
plt.title('HH Phase portrait (V-h) - Intersection of nullclines')
plt.xlabel (r'Voltage$(\mu V)$')
plt.ylabel('recovery variable h')
plt.legend()
plt.show()
