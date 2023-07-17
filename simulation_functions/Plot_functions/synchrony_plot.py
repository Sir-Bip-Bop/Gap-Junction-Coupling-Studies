import numpy as np 
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.animation as animation
import project

#General plot parameters and size definition
plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,7)
plt.rcParams.update({"axes.grid" : True})
plt.rcParams.update({"axes.titlesize": 14})
plt.rcParams.update({"axes.labelsize": 12})


chi_HH_k = np.loadtxt('simulation_functions/saved_data/sync_k/chi_HH_k.txt')
chi_IF_k = np.loadtxt('simulation_functions/saved_data/sync_k/chi_IF_k.txt')
chi_IZH_k = np.loadtxt('simulation_functions/saved_data/sync_k/chi_IZH_k.txt')
chi_ML_k = np.loadtxt('simulation_functions/saved_data/sync_k/chi_ML_k.txt')

rossum_HH_k = np.loadtxt('simulation_functions/saved_data/sync_k/rossum_HH_k.txt')
rossum_IF_k = np.loadtxt('simulation_functions/saved_data/sync_k/rossum_IF_k.txt')
rossum_IZH_k = np.loadtxt('simulation_functions/saved_data/sync_k/rossum_IZH_k.txt')
rossum_ML_k = np.loadtxt('simulation_functions/saved_data/sync_k/rossum_ML_k.txt')

rel_HH_k = np.loadtxt('simulation_functions/saved_data/sync_k/rel_HH_k.txt')
rel_IF_k = np.loadtxt('simulation_functions/saved_data/sync_k/rel_IF_k.txt')
rel_IZH_k = np.loadtxt('simulation_functions/saved_data/sync_k/rel_IZH_k.txt')
rel_ML_k = np.loadtxt('simulation_functions/saved_data/sync_k/rel_ML_k.txt')

k = [0,0.01,0.02,0.05,0.1,0.2,0.3,0.4,0.5]
fig, (ax1, ax2, ax3) = plt.subplots(3,1)

plt.suptitle('Synchrony measurements in function of Transmission Coefficient for a pair of coupled neurons',fontsize = 14)
fig.subplots_adjust(hspace=0.5)
#ax1.set_xlabel('Transmission Coefficient')
#ax2.set_xlabel('Transmission Coefficient')
ax3.set_xlabel('Transmission Coefficient')
ax1.set_ylabel('Chi measurement')
ax2.set_ylabel('Van Rossum distance')
ax3.set_ylabel('Reliability mesaurement')
ax1.tick_params(axis = 'y')
ax2.tick_params(axis = 'y')
ax3.tick_params(axis = 'y')

ax1.title.set_text('Chi measurement results')
ax2.title.set_text('Van Rossum distance results')
ax3.title.set_text('Reliability measurement results')

ln1 = ax1.scatter(k,chi_HH_k[:,0],color = 'r', label = 'HH')
ax1.errorbar(k,chi_HH_k[:,0],yerr=chi_HH_k[:,1], fmt='none',capsize=3,color = 'r')
ln2 = ax1.scatter(k,chi_IF_k[:,0],color = 'b', label = 'IF')
ax1.errorbar(k,chi_IF_k[:,0],yerr=chi_IF_k[:,1], fmt='none',capsize=3,color = 'b')
ln3 = ax1.scatter(k,chi_IZH_k[:,0],color = 'g', label = 'Izhikevich')
ax1.errorbar(k,chi_IZH_k[:,0],yerr=chi_IZH_k[:,1], fmt='none',capsize=3,color = 'g')
ln4 = ax1.scatter(k,chi_ML_k[:,0],color = 'orange', label = 'ML')
ax1.errorbar(k,chi_ML_k[:,0],yerr=chi_ML_k[:,1], fmt='none',capsize=3,color = 'orange')


ax2.scatter(k,rossum_HH_k[:,0],color = 'r', label = 'HH')
ax2.errorbar(k,rossum_HH_k[:,0],yerr=rossum_HH_k[:,1], fmt='none',capsize=3,color = 'r')
ax2.scatter(k,rossum_IF_k[:,0],color = 'b', label = 'IF')
ax2.errorbar(k,rossum_IF_k[:,0],yerr=rossum_IF_k[:,1], fmt='none',capsize=3,color = 'b')
ax2.scatter(k,rossum_IZH_k[:,0],color = 'g', label = 'Izhikevich')
ax2.errorbar(k,rossum_IZH_k[:,0],yerr=rossum_IZH_k[:,1], fmt='none',capsize=3,color = 'g')
ax2.scatter(k,rossum_ML_k[:,0],color = 'orange', label = 'ML')
ax2.errorbar(k,rossum_ML_k[:,0],yerr=rossum_ML_k[:,1], fmt='none',capsize=3,color = 'orange')

ax3.scatter(k,rel_HH_k[:,0],color = 'r', label = 'HH')
ax3.errorbar(k,rel_HH_k[:,0],yerr=rel_HH_k[:,1], fmt='none',capsize=3,color = 'r')
ax3.scatter(k,rel_IF_k[:,0],color = 'b', label = 'IF')
ax3.errorbar(k,rel_IF_k[:,0],yerr=rel_IF_k[:,1], fmt='none',capsize=3,color = 'b')
ax3.scatter(k,rel_IZH_k[:,0],color = 'g', label = 'Izhikevich')
ax3.errorbar(k,rel_IZH_k[:,0],yerr=rel_IZH_k[:,1], fmt='none',capsize=3,color = 'g')
ax3.scatter(k,rel_ML_k[:,0],color = 'orange', label = 'ML')
ax3.errorbar(k,rel_ML_k[:,0],yerr=rel_ML_k[:,1], fmt='none',capsize=3,color = 'orange')


leg  = ax1.legend(title='Models',bbox_to_anchor =( 0.1,1.5), ncols = 2 , mode = 'expand')
leg._legend_box.align = 'right'
plt.show()