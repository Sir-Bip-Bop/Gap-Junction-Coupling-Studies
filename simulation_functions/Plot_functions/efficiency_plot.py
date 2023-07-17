import numpy as np 
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.animation as animation
import project

#General plot parameters and size definition
plt.style.use('science')
plt.rcParams["figure.figsize"] = (12,7)
plt.rcParams.update({"axes.grid" : True})
plt.rcParams.update({"axes.titlesize": 22})
plt.rcParams.update({"axes.labelsize": 19})

#Loading the frequencies
fIF = np.loadtxt('simulation_functions/saved_data/efficiency/fIF.txt')
fHH = np.loadtxt('simulation_functions/saved_data/efficiency/fHH.txt')
fIZH  = np.loadtxt('simulation_functions/saved_data/efficiency/fIZH.txt')
fML  = np.loadtxt('simulation_functions/saved_data/efficiency/fML.txt')

#Loading the phases
phasesIF = np.loadtxt('simulation_functions/saved_data/efficiency/phasesIF.txt')
phasesHH = np.loadtxt('simulation_functions/saved_data/efficiency/phasesHH.txt')
phasesIZH = np.loadtxt('simulation_functions/saved_data/efficiency/phasesIZH.txt')
phasesML = np.loadtxt('simulation_functions/saved_data/efficiency/phasesML.txt')

#Loading the transmission coefficients
coefIF = np.loadtxt('simulation_functions/saved_data/efficiency/coeffIF.txt')
coefHH = np.loadtxt('simulation_functions/saved_data/efficiency/coeffHH.txt')
coefIZH = np.loadtxt('simulation_functions/saved_data/efficiency/coeffIZH.txt')
coefML = np.loadtxt('simulation_functions/saved_data/efficiency/coeffML.txt')


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.set_title('Efficiency of the electrical synapse - Phase delay')
ax1.set_xscale('log')

ax1.set_xlabel('Frequency (Hz)')
ax1.set_ylabel('Coupling coefficient (\%)')
ax1.set_xlim(1,100)

ax2.set_ylabel('Phase delay (deg)')

ax2.plot(fIF,phasesIF, 's', color = 'tab:blue')
ax2.plot(fHH,phasesHH, 's', color = 'tab:red')
ax2.plot(fIZH,phasesIZH, 's', color = 'tab:green')
ax2.plot(fML,phasesML, 's', color = 'tab:orange')

ax1.plot(fIF,coefIF, color = 'tab:blue',label = 'LIF')
ax1.plot(fHH,coefHH, color = 'tab:red',label= 'HH')
ax1.plot(fIZH, coefIZH,color = 'tab:green', label = 'IZH')
ax1.plot(fML,coefML, color = 'tab:orange', label = 'ML')

ax1.legend(title='Models',ncols=2, loc = (0.0,0.5),frameon=True, fontsize = 20)