import numpy as np 
import matplotlib.pyplot as plt
import scienceplots
import matplotlib.animation as animation
import project

#Loading all of the parameters needed for the simulation of all four models
params_HH = project.utils.load_HH()
params_LIF = project.utils.load_LIF()
params_IZH = project.utils.load_ISH()
params_ML = project.utils.load_ML()


k = 0.012
k_izh = 0.04
k_ML = 0.008


Isyn = [[0, 0.05], [0.05, 0]]
Is = np.array(Isyn)
tau = 1
order = 2


y0 = [-70,-70]
u0 = [0.0,0.0]
w0 = [0.0007223855976593603, 0.0007223855976593603]
V0 = -70
n0 =0.2
m0 = 0.1
h0 = 0.6 


#time parameters
t_final = 4000
dt = 0.01

#HH model
#Defining the base intensity to then modify into a sine wave function
I = np.zeros((int(t_final/dt),2))
t = np.linspace(0,t_final,int(t_final/dt))


#defining the frequencies, we have the initial set of frequencies to correspond the experiment, then we'll add additional frequencies
f_log = np.linspace(-3,-1,int(2/0.1))
f = 10**f_log
#extra_f = np.linspace(0.15,0.9,16)
#f = np.append(f,extra_f)
extra_f = np.linspace(0.15,0.9,2)
f = np.append(f,extra_f)


#variables used to store the resulting coefficient and delay
coef = np.zeros((len(f),1))
phases_test = np.zeros((len(f),1))


#variables used to store the data
data_HH_total_1 = np.zeros((len(f),len(t)))
data_HH_total_2 = np.zeros((len(f),len(t)))


#simulation to obtain the efficiency
for i in range(0,len(f)):
    #defining the intensity in function of frequency
    I[:,0] = 0.4 + 0.35*np.sin(2 * np.pi * f[i] * t)


    #computing and storing the simulation
    data4b_HH, completeHH_4b, a = project.models.HH_Neuron_Pairs(dt,t_final,1,[V0,V0],[n0,n0],[m0,m0],[h0,h0],*params_HH,I,Isyn,k,tau)
    data_HH_total_1[i] = data4b_HH[:,0]
    data_HH_total_2[i] = data4b_HH[:,1]
    end = len(data4b_HH)


    #computing the efficiency, chunking off the initial part of the simulation to ensure that we are on the steady state
    if i < 9:
        coef[i] = ( np.max(data4b_HH[10000:end-1,1]) - np.min(data4b_HH[10000:end-1,1]) ) / ( np.max(data4b_HH[10000:end-1,0]) - np.min(data4b_HH[10000:end-1,0]) )


        time_diff = project.utils.phases(data4b_HH[10000:end-1,:],dt)
        phases_test[i] = 360 * f[i] * time_diff
        if phases_test[i] > 90:
            phases_test[i] = 90 - abs(phases_test[i]-90)

    else:
        coef[i] = ( np.max(data4b_HH[80000:end-1,1]) - np.min(data4b_HH[80000:end-1,1]) ) / ( np.max(data4b_HH[80000:end-1,0]) - np.min(data4b_HH[80000:end-1,0]) )


        time_diff = project.utils.phases(data4b_HH[80000:end-1,:],dt)
        phases_test[i] = 360 * f[i] * time_diff
        if phases_test[i] > 90:
            phases_test[i] = 90 - abs(phases_test[i]-90)


#recomputing the coefficients, normalise by their maximum (which corresponds to the one at the lowest frequency)
f = 1000*f 
coef_HH_og = coef
coef = coef * 100 
for i in range(1,len(f)):
    coef[i] = coef[i] * 100 / coef[0]
coef[0] = 100


#storing the values
np.savetxt('simulation_functions/saved_data/efficiency/data_HH_total_1.txt',data_HH_total_1)
np.savetxt('simulation_functions/saved_data/efficiency/data_HH_total_2.txt',data_HH_total_2)
np.savetxt('simulation_functions/saved_data/efficiency/fHH.txt',f)
np.savetxt('simulation_functions/saved_data/efficiency/coeffHH.txt',coef)
np.savetxt('simulation_functions/saved_data/efficiency/phasesHH.txt',phases_test)

#IF Model
#Defining the base intensity to then modify into a sine wave function
I = np.zeros((int(t_final/dt),2))
t = np.linspace(0,t_final,int(t_final/dt))

#defining the frequencies, we have the initial set of frequencies to correspond the experiment, then we'll add additional frequencies
f_log = np.linspace(-3,-1,int(2/0.1))
f = 10**f_log
#extra_f = np.linspace(0.15,0.9,16)
#f = np.append(f,extra_f)
extra_f = np.linspace(0.15,0.9,2)
f = np.append(f,extra_f)


#variables used to store the resulting coefficient and delay
coef = np.zeros((len(f),1))
phases_test = np.zeros((len(f),1))


#variables used to store the data
data_IF_total_1 = np.zeros((len(f),len(t)))
data_IF_total_2 = np.zeros((len(f),len(t)))


#simulation to obtain the efficiency
for i in range(0,len(f)):
    #defining the intensity in function of frequency
    I[:,0] = 0.4+ 0.35*np.sin(2 * np.pi * f[i] * t)

    
    #computing and storing the simulation
    data4b_IF, complete_IF_4b, a = project.models.LIF_Neuron_Pairs(dt, t_final,order,y0,*params_LIF,I,Is,k,tau,1)
    end = len(data4b_IF)
    data_IF_total_1[i] = data4b_IF[:,0]
    data_IF_total_2[i] = data4b_IF[:,1]


    #computing the efficiency, chunking off the initial part of the simulation to ensure that we are on the steady state
    if i < 9:
        coef[i] = ( np.max(data4b_IF[10000:,1]) - np.min(data4b_IF[10000:,1]) ) / ( np.max(data4b_IF[10000:,0]) - np.min(data4b_IF[10000:,0]) )


        time_diff = project.utils.phases(data4b_IF[10000:end-1,:],dt)
        phases_test[i] = 360 * f[i] * time_diff
        if phases_test[i] > 90:
            phases_test[i] = 90 - abs(phases_test[i]-90)
            
    else:
        coef[i] = ( np.max(data4b_IF[80000:end-1,1]) - np.min(data4b_IF[80000:end-1,1]) ) / ( np.max(data4b_IF[80000:end-1,0]) - np.min(data4b_IF[80000:end-1,0]) )


        time_diff = project.utils.phases(data4b_IF[80000:end-1,:],dt)
        phases_test[i] = 360 * f[i] * time_diff
        if phases_test[i] > 90:
            phases_test[i] = 90 - abs(phases_test[i]-90)


#recomputing the coefficients, normalise by their maximum (which corresponds to the one at the lowest frequency)
f = 1000*f 
coef_IF_og = coef
coef = coef * 100 
for i in range(1,len(f)):
    coef[i] = coef[i] * 100 / coef[0]
coef[0] = 100


#storing the values
np.savetxt('simulation_functions/saved_data/efficiency/data_IF_total_1.txt',data_IF_total_1)
np.savetxt('simulation_functions/saved_data/efficiency/data_IF_total_2.txt',data_IF_total_2)
np.savetxt('simulation_functions/saved_data/efficiency/fIF.txt',f)
np.savetxt('simulation_functions/saved_data/efficiency/coeffIF.txt',coef)
np.savetxt('simulation_functions/saved_data/efficiency/phasesIF.txt',phases_test)

#IZH model
#Defining the base intensity to then modify into a sine wave function
I = np.zeros((int(t_final/dt),2))
t = np.linspace(0,t_final,int(t_final/dt))


#defining the frequencies, we have the initial set of frequencies to correspond the experiment, then we'll add additional frequencies
f_log = np.linspace(-3,-1,int(2/0.1))
f = 10**f_log
#extra_f = np.linspace(0.15,0.9,16)
#f = np.append(f,extra_f)
extra_f = np.linspace(0.15,0.9,2)
f = np.append(f,extra_f)


#variables used to store the resulting coefficient and delay
coef = np.zeros((len(f),1))
phases_test = np.zeros((len(f),1))


#variables used to store the data
data_IZH_total_1 = np.zeros((len(f),len(t)))
data_IZH_total_2 = np.zeros((len(f),len(t)))


#simulation to obtain the efficiency
for i in range(0,len(f)):
    #defining the intensity in function of frequency
    I[:,0] = 1.0 + 0.5*np.sin(2 * np.pi * f[i] * t)


    #computing and storing the simulation
    data4b_Ish, complete_Ish_4b, a = project.models.IZH_Neuron_Pairs(dt,t_final,order,y0,u0,I,Is,*params_IZH,k_izh,tau)
    end = len(data4b_Ish)
    data_IZH_total_1[i] = data4b_Ish[:,0]
    data_IZH_total_2[i] = data4b_Ish[:,1]
    
    
    #computing the efficiency, chunking off the initial part of the simulation to ensure that we are on the steady state
    if i < 9:
        coef[i] = ( np.max(data4b_Ish[10000:end-1,1]) - np.min(data4b_Ish[10000:end-1,1]) ) / ( np.max(data4b_Ish[10000:end-1,0]) - np.min(data4b_Ish[10000:end-1,0]) )


        time_diff  = project.utils.phases(data4b_Ish[10000:end-1],dt)
        phases_test[i] = 360 * f[i] * time_diff
        if phases_test[i] > 90:
            phases_test[i] = 90 - abs(phases_test[i]-90)
    else:
        coef[i] = ( np.max(data4b_Ish[10000:end-1,1]) - np.min(data4b_Ish[10000:end-1,1]) ) / ( np.max(data4b_Ish[10000:end-1,0]) - np.min(data4b_Ish[10000:end-1,0]) )


        time_diff  = project.utils.phases(data4b_Ish[80000:end-1,:],dt)
        phases_test[i] = 360 * f[i] * time_diff
        if phases_test[i] > 90:
            phases_test[i] = 90 - abs(phases_test[i]-90)


#recomputing the coefficients, normalise by their maximum (which corresponds to the one at the lowest frequency)
f = 1000*f 
coef_IZH_og = coef
coef = coef * 100 
for i in range(1,len(f)):
    coef[i] = coef[i] * 100 / coef[0]
coef[0] = 100


#storing the values
np.savetxt('simulation_functions/saved_data/efficiency/data_IZH_total_1.txt',data_IZH_total_1)
np.savetxt('simulation_functions/saved_data/efficiency/data_IZH_total_2.txt',data_IZH_total_2)
np.savetxt('simulation_functions/saved_data/efficiency/fIZH.txt',f)
np.savetxt('simulation_functions/saved_data/efficiency/coeffIZH.txt',coef)
np.savetxt('simulation_functions/saved_data/efficiency/phasesIZH.txt',phases_test)

#ML model
#Defining the base intensity to then modify into a sine wave function
I = np.zeros((int(t_final/dt),2))
t = np.linspace(0,t_final,int(t_final/dt))


#defining the frequencies, we have the initial set of frequencies to correspond the experiment, then we'll add additional frequencies
f_log = np.linspace(-3,-1,int(2/0.1))
f = 10**f_log
#extra_f = np.linspace(0.15,0.9,16)
#f = np.append(f,extra_f)
extra_f = np.linspace(0.15,0.9,2)
f = np.append(f,extra_f)


#variables used to store the resulting coefficient and delay
coef = np.zeros((len(f),1))
phases_test = np.zeros((len(f),1))


#variables used to store the data
data_ML_total_1 = np.zeros((len(f),len(t)))
data_ML_total_2 = np.zeros((len(f),len(t)))


#simulation to obtain the efficiency
for i in range(0,len(f)):
    #defining the intensity in function of frequency
    I[:,0] = 0.3 + 0.4*np.sin(2 * np.pi * f[i] * t)


    #computing and storing the simulation
    data4b_ML, complete_ML_4b, a = project.models.ML_Neuron_Pairs(dt,t_final,2,y0,w0,*params_ML,I,Isyn,k_ML,tau)
    end = len(data4b_ML)
    data_ML_total_1[i] = data4b_ML[:,0]
    data_ML_total_2[i] = data4b_ML[:,1]

    
    #computing the efficiency, chunking off the initial part of the simulation to ensure that we are on the steady state
    if i < 9:
        coef[i] = ( np.max(data4b_ML[10000:,1]) - np.min(data4b_ML[10000:,1]) ) / ( np.max(data4b_ML[10000:,0]) - np.min(data4b_ML[10000:,0]) )


        time_diff = project.utils.phases(data4b_ML[5000:end-1],dt)
        phases_test[i] = 360 * f[i] * time_diff
        if phases_test[i] > 90:
            phases_test[i] = 90 - abs(phases_test[i]-90)
    elif i == 9:
        coef[i] = ( np.max(data4b_ML[10000:,1]) - np.min(data4b_ML[10000:,1]) ) / ( np.max(data4b_ML[10000:,0]) - np.min(data4b_ML[10000:,0]) )


        time_diff = project.utils.phases(data4b_ML[70000:end-1],dt)
        phases_test[i] = 360 * f[i] * time_diff
        if phases_test[i] > 90:
            phases_test[i] = 90 - abs(phases_test[i]-90)
    else:
        coef[i] = ( np.max(data4b_ML[10000:end-1,1]) - np.min(data4b_ML[10000:end-1,1]) ) / ( np.max(data4b_ML[10000:end-1,0]) - np.min(data4b_ML[10000:end-1,0]) )


        time_diff = project.utils.phases(data4b_ML[80000:end-1,:],dt)
        phases_test[i] = 360 * f[i] * time_diff
        if phases_test[i] > 90:
            phases_test[i] = 90 - abs(phases_test[i]-90)


#recomputing the coefficients, normalise by their maximum (which corresponds to the one at the lowest frequency)
f = 1000*f 
coef_ML_og = coef
coef = coef * 100 
for i in range(1,len(f)):
    coef[i] = coef[i] * 100 / coef[0]
coef[0] = 100


#storing the values
np.savetxt('simulation_functions/saved_data/efficiency/data_ML_total_1.txt',data_ML_total_1)
np.savetxt('simulation_functions/saved_data/efficiency/data_ML_total_2.txt',data_ML_total_2)
np.savetxt('simulation_functions/saved_data/efficiency/fML.txt',f)
np.savetxt('simulation_functions/saved_data/efficiency/coeffML.txt',coef)
np.savetxt('simulation_functions/saved_data/efficiency/phasesML.txt',phases_test)