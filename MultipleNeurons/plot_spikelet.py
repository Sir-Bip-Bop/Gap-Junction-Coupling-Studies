import numpy as np 
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.animation as animation
from scipy.sparse import dok_matrix
import multiprocessing as mp
import scipy as sp
import matplotlib.ticker as ticker
import project

#General plot parameters and size definition
plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,7)
plt.rcParams.update({"axes.grid" : True})
plt.rcParams.update({"axes.titlesize": 18})
plt.rcParams.update({"axes.labelsize": 15})

k = np.loadtxt('MultipleNeurons/k_plus_0.0005/k_data.txt')
num = np.loadtxt('MultipleNeurons/k_plus_0.0005/num_data.txt')
diff_HH = np.loadtxt('MultipleNeurons/k_plus_0.0005/diff_HH_data.txt')

#discarding the values where diff_HH is zero
diff_HH[diff_HH<1e-2] = np.nan

#Plot and fit of the resulting data
K, NUM = np.meshgrid(k,num)

def fit_fun(k, num, k_coef, num_coef,base):
    return (k * k_coef * (num*num)*num_coef + base*k)
    #return 2

fig = plt.figure()
ax = fig.add_subplot(projection= '3d')
ax.plot_surface(K, NUM, diff_HH,cmap = 'cool')
plt.show()

def _fit_fun(M, *args):
    k, num = M
    arr = np.zeros(k.shape)
    arr += fit_fun(k,num, *args)

    return arr 

guess_prms = [3, 0.2,2]
kdata = np.vstack((K.ravel(),NUM.ravel()))
popt, pcov = sp.optimize.curve_fit(_fit_fun,kdata,diff_HH.ravel(),guess_prms,nan_policy = 'omit')



fig = plt.figure()
ax = fig.add_subplot(projection= '3d')
ax.plot_surface(K, NUM, fit_fun(K,NUM,*popt),cmap = 'cool')
cset = ax.contourf(K, NUM, diff_HH-fit_fun(K,NUM,*popt), zdir='z', offset=0, cmap='cool')
plt.show()

print('The final parameters:', popt)

#plt.grid()
#plt.pcolor(k,num,diff_HH)
#plt.colorbar()
#plt.show()


#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.contour3D(k,num,diff_HH)
#plt.show()