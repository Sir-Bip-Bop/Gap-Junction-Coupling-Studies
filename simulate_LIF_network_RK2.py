# Network of leaky integrate-and-fire neurons
# Exploring the role of gap junctions
# 26-08-2022

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy import sparse

# Set seed
np.random.seed(101)

# Numerical parameters
N = 100             # Number of neurons
T = 500             # Final time in ms
dt = 0.005         # time step in ms
M = int(T / dt)     # number of time steps
t = np.linspace(0, T, M + 1)

# Model parameters
pE = 0            # percent excitatory
pI = 1 - pE         # percent excitatory
Cm = 1
g_gap = 0.01
gL_E = 0.025
gL_I = 0.1
gE = 0.5
gI = 5
vL = -70
vE = 0
vI = -80
tauE = 1
tauI = 5
vthE = -36.4
vthI = -50.2#-49.2
vresetE = -51.1
vresetI = -66.5#-66.9

spikelet = 50#100

## To use Gaussian noise rather than Poisson
# # For noise
# sigma = 5
# sigma_bis_V = np.sqrt(dt) * sigma * np.sqrt(2. / Cm)
# sigma_bis_E = np.sqrt(dt) * sigma * np.sqrt(2. / tauE)
# sigma_bis_I = np.sqrt(dt) * sigma * np.sqrt(2. / tauI)

# Poisson noise
fnu = 30       # combined firing = r*f
r =  250         # firing rate for external spikes per neuron (Hz)
f =  fnu/r 
external_spikes = sparse.csr_matrix(stats.poisson.rvs(r/1000*dt, size=((M + 1, N))))

# probability of each type of chemical synapses
prob_ee = 0#.25          # E to E
prob_ei = 0#.25          # E to I
prob_ie = 0#.5          # I to E
prob_ii = 0#.5          # I to I
N_e = int(pE*N)
N_i = N - N_e

# Set up synaptic coupling matrices
# Matrix of random numbers between 0 and 1. Heavisided such that prob_nm of 
# the entries are 1 for each connection type
w_ee = np.heaviside(prob_ee-stats.uniform.rvs(size=(N_e,N_e)),0)
w_ei = np.heaviside(prob_ei-stats.uniform.rvs(size=(N_e,N_i)),0)
w_ie = np.heaviside(prob_ie-stats.uniform.rvs(size=(N_i,N_e)),0)
w_ii = np.heaviside(prob_ii-stats.uniform.rvs(size=(N_i,N_i)),0)

# Combine synaptic coupling matrices
W = np.concatenate([np.concatenate([w_ee, w_ie]),np.concatenate([w_ei, w_ii])],axis=1)
# Remove sefl-coupling
np.fill_diagonal(W, 0)
# Convert to sparse matrix to increase efficiency
W = sparse.csr_matrix(W)

# probability of each type of electrical synapses
prob_gap_ee = 0             # E to E
prob_gap_ie = 0             # I to E / I to E
prob_gap_ii = 0.3           # I to I

# Set up gap junction coupling matrix
# Same idea as above. Coupling reciprocal so w_gap_ei = w_gap_ei^T
w_gap_ee = np.heaviside(prob_gap_ee-stats.uniform.rvs(size=(N_e,N_e)),0)
w_gap_ei = np.heaviside(prob_gap_ie-stats.uniform.rvs(size=(N_e,N_i)),0)
w_gap_ii = np.heaviside(prob_gap_ii-stats.uniform.rvs(size=(N_i,N_i)),0)

# Combine gap junction coupling matrices
W_gap = np.concatenate([np.concatenate([w_gap_ee, w_gap_ei.T]),np.concatenate([w_gap_ei, w_gap_ii])],axis=1)
# Reciprocal coupling - project upper triangular entries to lower half
i_lower = np.tril_indices(len(W_gap), -1)
W_gap[i_lower] = W_gap.T[i_lower]
# Remove sefl-coupling
np.fill_diagonal(W_gap, 0)
# Convert to sparse matrix to increase efficiency
W_gap = sparse.csr_matrix(W_gap)

# Initialise variables
v = np.zeros((M + 1, N))
v[0, :] = stats.norm.rvs(loc=-60, scale=2, size=(1, N))
s_E = np.zeros((M + 1, N))
s_E[0,:] = stats.uniform.rvs(scale=0.1,size=(1, N))
s_I = np.zeros((M + 1, N))
s_I[0,:] = stats.uniform.rvs(scale=0.0001,size=(1, N))

# Set up spike time vector
spike_timesE = np.zeros((M, 2))
spike_timesI = np.zeros((M, 2))
spikeCounterE = 0
spikeCounterI = 0


def dxdt(x):
    # Network equations
    
    # Compute the sum (v_i-v_j) and multiple by coupling matrix 
    I_gap = np.ravel((W_gap.multiply(np.subtract.outer(x[0], x[0]))).sum(axis=0))
    
    # Voltage update
    dvE =  dt * ( gL_E * ( vL - x[0][:N_e] ) + g_gap * I_gap[:N_e]
                                    +  x[1][:N_e] * ( vE - x[0][:N_e] ) 
                                    +  x[2][:N_e] * ( vI - x[0][:N_e] ) ) / Cm \
    
    dvI = dt * ( gL_I * ( vL - x[0][N_e:] ) + g_gap * I_gap[N_e:]
                                    +  x[1][N_e:] * ( vE - x[0][N_e:] ) 
                                    +  x[2][N_e:] * ( vI - x[0][N_e:] ) ) / Cm \
        
        
    dv = np.append(dvE,dvI)
    
    # Synaptic conductance upadtes (no spike)                        
    dsE = dt * ( -x[1] / tauE )
    dsI = dt * ( -x[2] / tauI )
    
    return [dv, dsE, dsI, I_gap]


# Loop over time steps
# 2nd order Rumge-Kutta method
for i in range(M):
    
    k1 = dxdt([v[i, :],s_E[i, :],s_I[i, :]])                           
    k2 = dxdt([v[i, :] + k1[0]/2, s_E[i, :] + k1[1]/2, s_I[i, :]] + k1[2]/2)
    
    # integrate LIF volatge equations
    v[i + 1, :] = v[i, :] +  k2[0] #+  sigma_bis_V * np.random.randn(N)
    
    # integrate synaptic equations
    s_E[i + 1, :] = s_E[i, :] + k2[1] +  f*external_spikes[i, :]/tauE
    s_I[i + 1, :] = s_I[i, :] + k2[2] #+  sigma_bis_I * np.random.randn(N)

    # Determine which, if any, neurons has reached threshold
    spikeE = np.where(v[i + 1, :N_e] > vthE)
    spikeI = np.where(v[i + 1, N_e:] > vthI)
    
    # Excitatory spikes
    if len(spikeE[0]) > 0:
        
        # loop over those neurons that spike
        for spikeInd in spikeE[0]:           
            
            # Find spike time (linear interpolation)
            tspike = t[i] + dt * (vthE - v[i, spikeInd]) / (v[i+1, spikeInd] - v[i, spikeInd])
            
            if W[spikeInd,:].sum() > 0:
                # Increase excitatory conductances
                s_E[i + 1, :] = s_E[i + 1, :] + (gE / tauE) * W[spikeInd,:] / W[spikeInd,:].sum()
                
            # put spike times into the matrix
            spike_timesE[spikeCounterE,:] = [tspike, spikeInd]
            spikeCounterE = spikeCounterE + 1
            
            # Reset voltage of neurons that spiked
            v[i + 1, spikeInd] = vresetE
    
    # Inhibitory spikes            
    if len(spikeI[0]) > 0:
        
        # loop over those neurons that spike
        for spikeInd in spikeI[0]:           
            
            # Find spike time (linear interpolation)
            tspike = t[i] + dt * (vthI - v[i, N_e + spikeInd]) / (v[i+1, N_e + spikeInd] - v[i, N_e + spikeInd])
            
           
            # Only update if neuron has outgoing synaptic connections
            if W[N_e + spikeInd,:].sum() > 0:
                # Increase inhibitory conductances
                s_I[i + 1, :] = s_I[i + 1, :] + ( gI / tauI ) * W[N_e + spikeInd,:] / W[N_e + spikeInd,:].sum()
                
            # If neuron is gap junction coupled to other neurons
            if W_gap[N_e + spikeInd,:].sum() > 0:
                # Add spikelet
                v[i + 1, :] = v[i + 1, :] + spikelet * g_gap * W_gap[N_e + spikeInd,:] 
                    
            # put spike times into the matrix
            spike_timesI[spikeCounterI,:] = [tspike, N_e + spikeInd]
            spikeCounterI = spikeCounterI + 1

            # Reset voltage of neurons that spiked
            v[i + 1, N_e + spikeInd] = vresetI
            
# Plot voltages vs time
plt.figure()   
plt.plot(t, v)

# Plot avergae voltage vs time (axis=1 finds average across columns)
plt.figure()   
plt.plot(t, v.mean(axis=1))

# Plot excitatory synaptic conductances
plt.figure()  
plt.plot(t, s_E)

# Plot inhibitory synaptic conductances
plt.figure()  
plt.plot(t, s_I)

# Remove zeros after final spike times
spike_timesE = spike_timesE[:spikeCounterE,:]
spike_timesI = spike_timesI[:spikeCounterI,:]

# Create raster plot
plt.figure()  
plt.scatter(spike_timesE[:,0],spike_timesE[:,1],s=1)
plt.scatter(spike_timesI[:,0],spike_timesI[:,1],s=1)
plt.axis([0,T,0,100])

# plt.figure()  
# plt.scatter(t,external_spikes[:,9].T.toarray()[0] )
# plt.axis([0,100,0.9,1.1])
