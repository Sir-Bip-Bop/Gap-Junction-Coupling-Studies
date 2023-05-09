
                                                                                        
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

m = 0.1
m_values = np.ones(50) * m
h_values = np.linspace(0,1,50)
Roots_x = np.zeros(50)
Roots_y = np.zeros(50)

for p in range(len(m_values)):
      X = []
      Y = []
      ii = np.linspace(0,1,1000)
      for i in ii:
        solve_x = sp.optimize.root_scalar(HHx,args=(i,m_values[p],h_values[p]), x0 = -70, x1 = -50)
        X.append(solve_x.root)

      X = np.array(X)

      for i in X:
        solve_y = sp.optimize.root_scalar(HHy,args=(i), x0 = 0, x1 = 0.1)
        Y.append(solve_y.root)

      Y = np.array(Y)


      idx = np.argwhere(np.diff(np.sign(ii- Y))).flatten()
      Roots_x[p] = X[idx]
      Roots_y[p] = Y[idx]

m = 0.3
m_values = np.ones(50) * m
h_values = np.linspace(0,1,50)
Roots_x_2 = np.zeros(50)
Roots_y_2 = np.zeros(50)

for p in range(len(m_values)):
      X = []
      Y = []
      ii = np.linspace(0,1,1000)
      for i in ii:
        solve_x = sp.optimize.root_scalar(HHx,args=(i,m_values[p],h_values[p]), x0 = -70, x1 = -50)
        X.append(solve_x.root)

      X = np.array(X)

      for i in X:
        solve_y = sp.optimize.root_scalar(HHy,args=(i), x0 = 0, x1 = 0.1)
        Y.append(solve_y.root)

      Y = np.array(Y)


      idx = np.argwhere(np.diff(np.sign(ii- Y))).flatten()
      Roots_x_2[p] = X[idx]
      Roots_y_2[p] = Y[idx]

m = 0.5
m_values = np.ones(50) * m
h_values = np.linspace(0,1,50)
Roots_x_3 = np.zeros(50)
Roots_y_3 = np.zeros(50)

for p in range(len(m_values)):
      X = []
      Y = []
      ii = np.linspace(0,1,1000)
      for i in ii:
        solve_x = sp.optimize.root_scalar(HHx,args=(i,m_values[p],h_values[p]), x0 = -70, x1 = -50)
        X.append(solve_x.root)

      X = np.array(X)

      for i in X:
        solve_y = sp.optimize.root_scalar(HHy,args=(i), x0 = 0, x1 = 0.1)
        Y.append(solve_y.root)

      Y = np.array(Y)


      idx = np.argwhere(np.diff(np.sign(ii- Y))).flatten()
      Roots_x_3[p] = X[idx]
      Roots_y_3[p] = Y[idx]
m = 0.8
m_values = np.ones(50) * m
h_values = np.linspace(0,1,50)
Roots_x_4 = np.zeros(50)
Roots_y_4 = np.zeros(50)

for p in range(len(m_values)):
      X = []
      Y = []
      ii = np.linspace(0,1,1000)
      for i in ii:
        solve_x = sp.optimize.root_scalar(HHx,args=(i,m_values[p],h_values[p]), x0 = -70, x1 = -50)
        X.append(solve_x.root)

      X = np.array(X)

      for i in X:
        solve_y = sp.optimize.root_scalar(HHy,args=(i), x0 = 0, x1 = 0.1)
        Y.append(solve_y.root)

      Y = np.array(Y)


      idx = np.argwhere(np.diff(np.sign(ii- Y))).flatten()
      Roots_x_4[p] = X[idx]
      Roots_y_4[p] = Y[idx]

fig, axs = plt.subplots(2, 2)

plot_1 = axs[0,0].scatter(Roots_x,Roots_y, marker = 'o',c = h_values, cmap=plt.cm.get_cmap('cool'))
plt.colorbar(plot_1,ax=axs[0,0])
axs[0,0].grid()
axs[0,0].set_title('HH Phase portrait (V-n) - Intersection of nullclines for m = 0.1')
axs[0,0].set_xlabel (r'Voltage$(\mu V)$')
axs[0,0].set_ylabel('Recovery variable n')


plot_2 = axs[0,1].scatter(Roots_x_2,Roots_y_2, marker = 'o',c = h_values, cmap=plt.cm.get_cmap('cool'))
plt.colorbar(plot_2,ax=axs[0,1])
axs[0,1].grid()
axs[0,1].set_title('HH Phase portrait (V-n) - Intersection of nullclines for m = 0.3')
axs[0,1].set_xlabel (r'Voltage$(\mu V)$')
axs[0,1].set_ylabel('Recovery variable n')

plot_3 = axs[1,0].scatter(Roots_x_3,Roots_y_3, marker = 'o',c = h_values, cmap=plt.cm.get_cmap('cool'))
plt.colorbar(plot_3,ax=axs[1,0])
axs[1,0].grid()
axs[1,0].set_title('HH Phase portrait (V-n) - Intersection of nullclines for m = 0.5')
axs[1,0].set_xlabel (r'Voltage$(\mu V)$')
axs[1,0].set_ylabel('Recovery variable n')


plot_4 = axs[1,1].scatter(Roots_x_4,Roots_y_4, marker = 'o',c = h_values, cmap=plt.cm.get_cmap('cool'))
plt.colorbar(plot_4,ax=axs[1,1])
axs[1,1].grid()
axs[1,1].set_title('HH Phase portrait (V-n) - Intersection of nullclines for m = 0.8')
axs[1,1].set_xlabel (r'Voltage$(\mu V)$')
axs[1,1].set_ylabel('Recovery variable n')


plt.show()
