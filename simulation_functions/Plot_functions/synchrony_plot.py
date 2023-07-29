import numpy as np 
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.animation as animation
import project
import scipy as sp

#General plot parameters and size definition
plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,7)
plt.rcParams.update({"axes.grid" : True})
plt.rcParams.update({"axes.titlesize": 16})
plt.rcParams.update({"axes.labelsize": 14})


chi_HH_k = np.loadtxt('simulation_functions/saved_data/sync_noise_chemical/chi_HH_noise.txt',skiprows=1)
chi_IF_k = np.loadtxt('simulation_functions/saved_data/sync_noise_chemical/chi_IF_noise.txt',skiprows=1)
chi_IZH_k = np.loadtxt('simulation_functions/saved_data/sync_noise_chemical/chi_IZH_noise.txt',skiprows=1)
chi_ML_k = np.loadtxt('simulation_functions/saved_data/sync_noise_chemical/chi_ML_noise.txt',skiprows=1)

rossum_HH_k = np.loadtxt('simulation_functions/saved_data/sync_noise_chemical/rossum_HH_noise.txt',skiprows=1)
rossum_IF_k = np.loadtxt('simulation_functions/saved_data/sync_noise_chemical/rossum_IF_noise.txt',skiprows=1)
rossum_IZH_k = np.loadtxt('simulation_functions/saved_data/sync_noise_chemical/rossum_IZH_noise.txt',skiprows=1)
rossum_ML_k = np.loadtxt('simulation_functions/saved_data/sync_noise_chemical/rossum_ML_noise.txt',skiprows=1)

rel_HH_k = np.loadtxt('simulation_functions/saved_data/sync_noise_chemical/rel_HH_noise.txt',skiprows=1)
rel_IF_k = np.loadtxt('simulation_functions/saved_data/sync_noise_chemical/rel_IF_noise.txt',skiprows=1)
rel_IZH_k = np.loadtxt('simulation_functions/saved_data/sync_noise_chemical/rel_IZH_noise.txt',skiprows=1)
rel_ML_k = np.loadtxt('simulation_functions/saved_data/sync_noise_chemical/rel_ML_noise.txt',skiprows=1)


def chi_fit(x,x_coef,cte):
    return x*x*x_coef + cte

def ross_fit(x,x_coef,cte):
    return x*x*x_coef + cte

def rel_fit(x,x_coef,cte):
    return x*x*x_coef + cte


#k = np.array([0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5])
k = np.array([2.5,5,7.5,10,12.5,15,17.5,20])
fig, (ax1, ax2, ax3) = plt.subplots(3,1)

HH_popt, pcov = sp.optimize.curve_fit(chi_fit,xdata=k,ydata=chi_HH_k[:,0])
IF_popt, pcov = sp.optimize.curve_fit(chi_fit,xdata=k,ydata=chi_IF_k[:,0])
IZH_popt, pcov = sp.optimize.curve_fit(chi_fit,xdata=k,ydata=chi_IZH_k[:,0])
ML_popt, pcov = sp.optimize.curve_fit(chi_fit,xdata=k,ydata=chi_ML_k[:,0])

HH_popt_2, pcov = sp.optimize.curve_fit(ross_fit,xdata=k,ydata=rossum_HH_k[:,0])
IF_popt_2, pcov = sp.optimize.curve_fit(ross_fit,xdata=k,ydata=rossum_IF_k[:,0])
IZH_popt_2, pcov = sp.optimize.curve_fit(ross_fit,xdata=k,ydata=rossum_IZH_k[:,0])
ML_popt_2, pcov = sp.optimize.curve_fit(ross_fit,xdata=k,ydata=rossum_ML_k[:,0])

HH_popt_3, pcov = sp.optimize.curve_fit(rel_fit,xdata=k,ydata=rel_HH_k[:,0])
IF_popt_3, pcov = sp.optimize.curve_fit(rel_fit,xdata=k,ydata=rel_IF_k[:,0])
IZH_popt_3, pcov = sp.optimize.curve_fit(rel_fit,xdata=k,ydata=rel_IZH_k[:,0])
ML_popt_3, pcov = sp.optimize.curve_fit(rel_fit,xdata=k,ydata=rel_ML_k[:,0])


plt.suptitle(r'Synchrony measurements in function of Noise Variance  with $I_{syn} = 0.3$',fontsize = 18)
#plt.suptitle(r'Synchrony measurements fit to a Quadratic Function with $I_{syn} = 0.3$',fontsize = 18)
fig.subplots_adjust(hspace=0.5)
#ax1.set_xlabel('Transmission Coefficient')
#ax2.set_xlabel('Transmission Coefficient')
#ax3.set_xlabel('Transmission Coefficient')
ax3.set_xlabel('Noise Variance')
ax1.set_ylabel(r'$\chi^2$')
ax2.set_ylabel(r'$D_R$')
ax3.set_ylabel(r'$\mathcal{R}$')
ax1.tick_params(axis = 'y')
ax2.tick_params(axis = 'y')
ax3.tick_params(axis = 'y')

#ax1.title.set_text(r' $\chi^2$ in function of transmission coefficient')
#ax2.title.set_text(r' $D_R$ in function of transmission coefficient')
#ax3.title.set_text(r' $\mathcal{R}$ in function of transmission coefficient')

ax1.title.set_text(r' $\chi^2$ in function of noise variance')
ax2.title.set_text(r' $D_R$ in function of noise variance')
ax3.title.set_text(r' $\mathcal{R}$ in function of noise variance')

ln1 = ax1.scatter(k,chi_HH_k[:,0],color = 'r', label = 'HH')
ax1.errorbar(k,chi_HH_k[:,0],yerr=chi_HH_k[:,1], fmt='none',capsize=3,color = 'r')
#ax1.plot(k,chi_fit(k,*HH_popt),color = 'r')
ln2 = ax1.scatter(k,chi_IF_k[:,0],color = 'b', label = 'IF')
ax1.errorbar(k,chi_IF_k[:,0],yerr=chi_IF_k[:,1], fmt='none',capsize=3,color = 'b')
#ax1.plot(k,chi_fit(k,*IF_popt),color = 'b')
ln3 = ax1.scatter(k,chi_IZH_k[:,0],color = 'g', label = 'IZH')
ax1.errorbar(k,chi_IZH_k[:,0],yerr=chi_IZH_k[:,1], fmt='none',capsize=3,color = 'g')
#ax1.plot(k,chi_fit(k,*IZH_popt),color = 'g')
ln4 = ax1.scatter(k,chi_ML_k[:,0],color = 'orange', label = 'ML')
ax1.errorbar(k,chi_ML_k[:,0],yerr=chi_ML_k[:,1], fmt='none',capsize=3,color = 'orange')
#ax1.plot(k,chi_fit(k,*ML_popt),color = 'orange')

ax2.scatter(k,rossum_HH_k[:,0],color = 'r', label = 'HH')
ax2.errorbar(k,rossum_HH_k[:,0],yerr=rossum_HH_k[:,1], fmt='none',capsize=3,color = 'r')
#ax2.plot(k,ross_fit(k,*HH_popt_2),color = 'r')
ax2.scatter(k,rossum_IF_k[:,0],color = 'b', label = 'IF')
ax2.errorbar(k,rossum_IF_k[:,0],yerr=rossum_IF_k[:,1], fmt='none',capsize=3,color = 'b')
#ax2.plot(k,ross_fit(k,*IF_popt_2),color = 'b')
ax2.scatter(k,rossum_IZH_k[:,0],color = 'g', label = 'IZH')
ax2.errorbar(k,rossum_IZH_k[:,0],yerr=rossum_IZH_k[:,1], fmt='none',capsize=3,color = 'g')
#ax2.plot(k,ross_fit(k,*IZH_popt_2),color = 'g')
ax2.scatter(k,rossum_ML_k[:,0],color = 'orange', label = 'ML')
ax2.errorbar(k,rossum_ML_k[:,0],yerr=rossum_ML_k[:,1], fmt='none',capsize=3,color = 'orange')
#ax2.plot(k,ross_fit(k,*ML_popt_2),color = 'orange')

ax3.scatter(k,rel_HH_k[:,0],color = 'r', label = 'HH')
ax3.errorbar(k,rel_HH_k[:,0],yerr=rel_HH_k[:,1], fmt='none',capsize=3,color = 'r')
#ax3.plot(k,ross_fit(k,*HH_popt_3),color = 'r')
ax3.scatter(k,rel_IF_k[:,0],color = 'b', label = 'IF')
ax3.errorbar(k,rel_IF_k[:,0],yerr=rel_IF_k[:,1], fmt='none',capsize=3,color = 'b')
#ax3.plot(k,ross_fit(k,*IF_popt_3),color = 'b')
ax3.scatter(k,rel_IZH_k[:,0],color = 'g', label = 'IZH')
ax3.errorbar(k,rel_IZH_k[:,0],yerr=rel_IZH_k[:,1], fmt='none',capsize=3,color = 'g')
#ax3.plot(k,ross_fit(k,*IZH_popt_3),color = 'g')
ax3.scatter(k,rel_ML_k[:,0],color = 'orange', label = 'ML')
ax3.errorbar(k,rel_ML_k[:,0],yerr=rel_ML_k[:,1], fmt='none',capsize=3,color = 'orange')
#ax3.plot(k,ross_fit(k,*ML_popt_3),color = 'orange')


leg  = ax1.legend(bbox_to_anchor =( 0.1,1.4), ncols = 2 , mode = 'expand',prop={'size':12})
leg._legend_box.align = 'center'
plt.show()