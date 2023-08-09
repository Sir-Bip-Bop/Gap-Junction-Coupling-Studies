import numpy as np 
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.animation as animation
from scipy.sparse import dok_matrix
import multiprocessing as mp
import scipy as sp
import matplotlib.ticker as ticker
import project
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#General plot parameters and size definition
plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,7)
plt.rcParams.update({"axes.grid" : True})
plt.rcParams.update({"axes.titlesize": 18})
plt.rcParams.update({"axes.labelsize": 18})

k = np.loadtxt('MultipleNeurons/k_plus_0.0007/k_data.txt')
num = np.loadtxt('MultipleNeurons/k_plus_0.0007/num_data.txt')
diff_HH = np.loadtxt('MultipleNeurons/k_plus_0.0007/diff_HH_data.txt')

#k = np.loadtxt('MultipleNeurons/k_plus_divided/k_data.txt')
#num = np.loadtxt('MultipleNeurons/k_plus_divided/num_data.txt')
#diff_HH = np.loadtxt('MultipleNeurons/k_plus_divided/diff_HH_data.txt')


#discarding the values where diff_HH is zero
diff_HH[diff_HH<1e-2] = np.nan

def fit_linear(x,a,b):
    return a*x + b

def fit_quadratic(x,a,b,c):
    return a*x**2 + b*x + c

def fit_exponential(x,a,b,c):
    return a*np.exp(b*x) + c
p0_linear = [0,0]
p0_quadratic = [0,0,0]
p0_exponential = [0,0,0]

mask = ~np.isnan(diff_HH[:,0])
popt_linear_0, pcov_linear_0 = sp.optimize.curve_fit(fit_linear,num[mask],diff_HH[mask,0],p0_linear)
popt_quadratic_0, pcov_quadratic_0 = sp.optimize.curve_fit(fit_quadratic,num[mask],diff_HH[mask,0],p0_quadratic)
popt_exponential_0, pcov_exponential_0 = sp.optimize.curve_fit(fit_exponential,num[mask],diff_HH[mask,0],p0_exponential)

mask = ~np.isnan(diff_HH[:,5])
popt_linear_1, pcov_linear_1 = sp.optimize.curve_fit(fit_linear,num[mask],diff_HH[mask,5],p0_linear)
popt_quadratic_1, pcov_quadratic_1 = sp.optimize.curve_fit(fit_quadratic,num[mask],diff_HH[mask,5],p0_quadratic)
#popt_exponential_1, pcov_exponential_1 = sp.optimize.curve_fit(fit_exponential,num[mask],diff_HH[mask,5],p0_exponential) #does not converge

mask = ~np.isnan(diff_HH[:,10])
popt_linear_2, pcov_linear_2 = sp.optimize.curve_fit(fit_linear,num[mask],diff_HH[mask,10],p0_linear)
popt_quadratic_2, pcov_quadratic_2 = sp.optimize.curve_fit(fit_quadratic,num[mask],diff_HH[mask,10],p0_quadratic)
popt_exponential_2, pcov_exponential_2 = sp.optimize.curve_fit(fit_exponential,num[mask],diff_HH[mask,10],p0_exponential)

mask = ~np.isnan(diff_HH[:,15])
popt_linear_3, pcov_linear_3 = sp.optimize.curve_fit(fit_linear,num[mask],diff_HH[mask,15],p0_linear)
popt_quadratic_3, pcov_quadratic_3 = sp.optimize.curve_fit(fit_quadratic,num[mask],diff_HH[mask,15],p0_quadratic)
popt_exponential_3, pcov_exponential_3 = sp.optimize.curve_fit(fit_exponential,num[mask],diff_HH[mask,15],p0_exponential)

mask = ~np.isnan(diff_HH[:,20])
popt_linear_4, pcov_linear_4 = sp.optimize.curve_fit(fit_linear,num[mask],diff_HH[mask,20],p0_linear)
popt_quadratic_4, pcov_quadratic_4 = sp.optimize.curve_fit(fit_quadratic,num[mask],diff_HH[mask,20],p0_quadratic)
popt_exponential_4, pcov_exponential_4 = sp.optimize.curve_fit(fit_exponential,num[mask],diff_HH[mask,20],p0_exponential)

mask = ~np.isnan(diff_HH[:,25])
popt_linear_5, pcov_linear_5 = sp.optimize.curve_fit(fit_linear,num[mask],diff_HH[mask,25],p0_linear)
popt_quadratic_5, pcov_quadratic_5 = sp.optimize.curve_fit(fit_quadratic,num[mask],diff_HH[mask,25],p0_quadratic)
popt_exponential_5, pcov_exponential_5 = sp.optimize.curve_fit(fit_exponential,num[mask],diff_HH[mask,25],p0_exponential)

mask = ~np.isnan(diff_HH[:,29])
popt_linear_6, pcov_linear_6 = sp.optimize.curve_fit(fit_linear,num[mask],diff_HH[mask,29],p0_linear)
popt_quadratic_6, pcov_quadratic_6 = sp.optimize.curve_fit(fit_quadratic,num[mask],diff_HH[mask,29],p0_quadratic)
popt_exponential_6, pcov_exponential_6 = sp.optimize.curve_fit(fit_exponential,num[mask],diff_HH[mask,29],p0_exponential)


plt.scatter(num,diff_HH[:,0], label = 'k = %.2e' %k[0], color = '#28168b')
plt.plot(num,fit_linear(num,*popt_linear_0), color = 'red')
plt.plot(num,fit_quadratic(num,*popt_quadratic_0), color = 'green')
plt.plot(num,fit_exponential(num,*popt_exponential_0), color = 'purple')
plt.scatter(num,diff_HH[:,5], label = 'k = %.2e' %k[5] ,color = '#37397d')
plt.plot(num,fit_linear(num,*popt_linear_1), color = 'red')
plt.plot(num,fit_quadratic(num,*popt_quadratic_1), color ='green')
#plt.plot(num,fit_exponential(num,*popt_exponential_1)) does not converge
plt.scatter(num,diff_HH[:,10], label = 'k = %.2e' %k[10], color = '#4858c4')
plt.plot(num,fit_linear(num,*popt_linear_2), color = 'red')
plt.plot(num,fit_quadratic(num,*popt_quadratic_2), color ='green')
plt.plot(num,fit_exponential(num,*popt_exponential_2), color = 'purple')
plt.scatter(num,diff_HH[:,15], label = 'k = %.2e' %k[15], color = '#8c62f1')
plt.plot(num,fit_linear(num,*popt_linear_3), color = 'red')
plt.plot(num,fit_quadratic(num,*popt_quadratic_3), color = 'green')
plt.plot(num,fit_exponential(num,*popt_exponential_3), color = 'purple')
plt.scatter(num,diff_HH[:,20], label = 'k = %.2e' %k[20], color = '#54abeb')
plt.plot(num,fit_linear(num,*popt_linear_4), color = 'red')
plt.plot(num,fit_quadratic(num,*popt_quadratic_4), color = 'green')
plt.plot(num,fit_exponential(num,*popt_exponential_4), color = 'purple')
plt.scatter(num,diff_HH[:,25], label = 'k = %.2e' %k[25], color = '#96e8ff')
plt.plot(num,fit_linear(num,*popt_linear_5), color = 'red')
plt.plot(num,fit_quadratic(num,*popt_quadratic_5), color = 'green')
plt.plot(num,fit_exponential(num,*popt_exponential_5), color = 'purple')
plt.scatter(num,diff_HH[:,29], label = 'k = %.2e' %k[29], color = '#cebdb2')
plt.plot(num,fit_linear(num,*popt_linear_6), color = 'red')
plt.plot(num,fit_quadratic(num,*popt_quadratic_6), color = 'green')
plt.plot(num,fit_exponential(num,*popt_exponential_6), color = 'purple')
plt.xlabel('Number of Neurons')
plt.ylabel('Spikelet heigth (mV)')
plt.legend(frameon=True, loc = 'upper left')
plt.title('Spikelet Height in function of neuron number for multiple values of gap junction strength')
plt.show()