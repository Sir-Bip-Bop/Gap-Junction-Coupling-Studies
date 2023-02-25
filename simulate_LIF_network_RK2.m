% Network of leaky integrate-and-fire neurons
% Exploring the role of gap junctions
% 03-01-2023
tic
close all
clear all

% Set seed
rng(101);

%% Setting up problem

% Numerical parameters
N = 100;             % Number of neurons
T = 1000;             % Final time in ms
dt = 0.005;         % time step in ms
M = round(T / dt);     % number of time steps
t = linspace(0, T, M + 1);

% Model parameters
pE = 0;            % fraction excitatory
pI = 1 - pE;         % fraction excitatory
Cm = 1;
g_gap = 0.02;
gL_E = 0.025;
gL_I = 0.1;
gE = 0.5;
gI = 5;
vL = -70;
vE = 0;
vI = -80;
tauE = 1;
tauI = 5;
vthE = -36.4;
vthI = -50.2;
vresetE = -51.1;
vresetI = -66.5;
spikelet = 20;

% Poisson noise
fnu = 34;       % combined firing = r*f
r = 500;         % firing rate for external spikes per neuron (Hz)
f = fnu/r;
external_spikes = sparse(poissrnd(r/1000*dt,[M + 1, N]));

% probability of each type of chemical synapses
prob_ee = 0;%.25          % E to E
prob_ei = 0;%.25          % E to I
prob_ie = 0;%.5          % I to E
prob_ii = 0;%.5          % I to I
N_e = ceil(pE*N);
N_i = N - N_e;

% Set up synaptic coupling matrices
% Matrix of random numbers between 0 and 1. Heavisided such that prob_nm of 
% the entries are 1 for each connection type
w_ee = heaviside(prob_ee-unifrnd(0,1,[N_e,N_e]));
w_ei = heaviside(prob_ei-unifrnd(0,1,[N_e,N_i]));
w_ie = heaviside(prob_ie-unifrnd(0,1,[N_i,N_e]));
w_ii = heaviside(prob_ii-unifrnd(0,1,[N_i,N_i]));

% Combine synaptic coupling matrices
if isempty(w_ee)
    W = w_ii;
elseif isempty(w_ii)
    W = w_ee;
else
    W = [w_ee, w_ie; w_ei, w_ii];
end
% Remove sefl-coupling
W = W - diag(diag(W));
% Convert to sparse matrix to increase efficiency
W = sparse(W);

% probability of each type of electrical synapses
prob_gap_ee = 0;             % E to E
prob_gap_ie = 0;             % I to E / I to E
prob_gap_ii = 0.3;           % I to I

% Set up gap junction coupling matrix
% Same idea as above. Coupling reciprocal so w_gap_ei = w_gap_ei^T
w_gap_ee = heaviside(prob_gap_ee-unifrnd(0,1,[N_e,N_e]));
w_gap_ei = heaviside(prob_gap_ie-unifrnd(0,1,[N_e,N_i]));
w_gap_ii = heaviside(prob_gap_ii-unifrnd(0,1,[N_i,N_i]));

% Combine gap junction coupling matrices
if isempty(w_gap_ee)
    W_gap = w_gap_ii;
elseif isempty(w_gap_ii)
    W_gap = w_gap_ee;
else
    W_gap = [w_gap_ee, w_gap_ei.T; w_gap_ei, w_gap_ii];
end
% Reciprocal coupling - project upper triangular entries to lower half
[ i, j ] = find(tril(ones(N), 1));
W_gap( j + N*(i-1) )= W_gap( i + N*(j-1) );
% Remove sefl-coupling
W_gap =  W_gap - diag(diag(W_gap));
% Convert to sparse matrix to increase efficiency
W_gap = sparse(W_gap);

% Initialise variables
v = zeros(M + 1, N);
v(1, :) = normrnd(-60, 2, [1, N]);
s_E = zeros(M + 1, N);
s_E(1,:) = unifrnd(0,0.1,[1, N]);
s_I = zeros(M + 1, N);
s_I(1,:) = unifrnd(0,0.01,[1, N]);

% Set up spike time vector
% spike_timesE = zeros(M, 2);
% spike_timesI = zeros(M, 2);
% spikeCounterE = 1;
% spikeCounterI = 1;
spike_matrix = sparse(N,M+1);

%% Evolve and solve equations
% Loop over time steps
% 2nd order Rumge-Kutta method
for i=1:M
    
    [k1_v,k1_E,k1_I] = dxdt([v(i, :);s_E(i, :);s_I(i, :)],dt,vL,vE,vI,gL_E,gL_I,Cm,g_gap,W_gap,tauE,tauI,N_e);    
    
    [k2_v,k2_E,k2_I,I_gap] = dxdt([v(i, :) + k1_v/2; s_E(i, :) + k1_E/2; s_I(i, :) + k1_I/2],dt,vL,vE,vI,gL_E,gL_I,Cm,g_gap,W_gap,tauE,tauI,N_e);
    
    % integrate LIF volatge equations
    v(i + 1, :) = v(i, :) +  k2_v; %+  sigma_bis_V * np.random.randn(N)
    
    % integrate synaptic equations
    s_E(i + 1, :) = s_E(i, :) + k2_E +  f*external_spikes(i, :)/tauE;
    s_I(i + 1, :) = s_I(i, :) + k2_I;

    % Determine which, if any, neurons has reached threshold
    spikeE = find(v(i + 1, 1:N_e) > vthE);
    spikeI = find(v(i + 1, N_e+1:end) > vthI);
    
    % Excitatory spikes
    if ~isempty(spikeE)
        
        % loop over those neurons that spike
        for spikeInd=spikeE           
            
            % Find spike time (linear interpolation)
            tspike = t(i) + dt * (vthE - v(i, spikeInd)) / (v(i+1, spikeInd) - v(i, spikeInd));
            
            if sum(W(spikeInd,:)) > 0
                % Increase excitatory conductances
                s_E(i + 1, :) = s_E(i + 1, :) + (gE / tauE) * W(spikeInd,:) / sum(W(spikeInd,:));
            end
                
            % put spike times into the matrix
%             spike_timesE(spikeCounterE,:) = [tspike, spikeInd];
%             spikeCounterE = spikeCounterE + 1;
            spike_matrix(spikeInd,i) = 1;
            
            % Reset voltage of neurons that spiked
            v(i + 1, spikeInd) = vresetE;
        end
    end
    
    % Inhibitory spikes            
    if ~isempty(spikeI)
        
        % loop over those neurons that spike
        for spikeInd = spikeI          
            
            % Find spike time (linear interpolation)
            tspike = t(i) + dt * (vthI - v(i, N_e + spikeInd)) / (v(i+1, N_e + spikeInd) - v(i, N_e + spikeInd));
            
           
            % Only update if neuron has outgoing synaptic connections
            if sum(W(N_e + spikeInd,:)) > 0
                % Increase inhibitory conductances
                s_I(i + 1, :) = s_I(i + 1, :) + ( gI / tauI ) * W(N_e + spikeInd,:) / sum(W(N_e + spikeInd,:));
            end
                
            % If neuron is gap junction coupled to other neurons
            if sum(W_gap(N_e + spikeInd,:)) > 0
                % Add spikelet
                v(i + 1, :) = v(i + 1, :) + spikelet * g_gap * W_gap(N_e + spikeInd,:);
            end
                    
%             % put spike times into the matrix
%             spike_timesI(spikeCounterI,:) = [tspike, N_e + spikeInd];
%             spikeCounterI = spikeCounterI + 1;
            spike_matrix(N_e+spikeInd,i) = 1;
            
            % Reset voltage of neurons that spiked
            v(i + 1, N_e + spikeInd) = vresetI;
        end
    end
end

% Find spike times to compute firing rate and creating raster plot later
[neuron_num, spike_time] = find(spike_matrix>0);
network_firing_rate = length(neuron_num)/N*1000/T;
network_firing_rate_E = length(find(neuron_num<=N_e))/N_e*1000/T;
network_firing_rate_I = length(find(neuron_num>N_e))/(N-N_e)*1000/T;

fprintf('\n<strong>Network firing rate</strong>\n')
fprintf('Firing rate of all cells: %.2f Hz\n',network_firing_rate)
fprintf('Firing rate of excitatory cells: %.2f Hz\n',network_firing_rate_E)
fprintf('Firing rate of inhibitory cells: %.2f Hz\n\n',network_firing_rate_I)


%% Synchrony measures
% Peaks in mean voltage
% Compute average voltages
mean_voltage = mean(v,2);
mean_voltageE = mean(v(:,1:N_e),2);
mean_voltageI = mean(v(:,N_e+1:end),2);

% Find peaks
% Peaks - decreasing through half-way point between reset and threshold
peaksE = find((mean_voltageE(1:end-1)>(vresetE+vthE)/2)&(mean_voltageE(2:end)<(vresetE+vthE)/2));
peaksI = find((mean_voltageI(1:end-1)>(vresetI+vthI)/2)&(mean_voltageI(2:end)<(vresetI+vthI)/2));

% Count number of peaks
num_peaks_E = length(peaksE);
num_peaks_I = length(peaksI);
peak_rate_E = num_peaks_E*1000/T;
peak_rate_I = num_peaks_I*1000/T;

fprintf('<strong>Peaks in mean voltage</strong>\n')
fprintf('%i peaks in the excitatory voltage (%i peaks per second)\n', num_peaks_E, peak_rate_E)
fprintf('%i peaks in the inhibitory voltage (%i peaks per second)\n\n', num_peaks_I, peak_rate_I)

% Chi measure
chi_all = compute_chi(v);
chi_E = compute_chi(v(:,1:N_e));
chi_I = compute_chi(v(:,N_e+1:end));

fprintf('<strong>Chi synchrony measure</strong>\n')
fprintf('Chi for the network is %i\n', chi_all)
fprintf('Chi for the excitatory cells is %i\n', chi_E)
fprintf('Chi for the inhibitory cells is %i\n\n', chi_I)

% van Rossum distance
van_Rossum = compute_van_Rossum_distance(spike_matrix,t,10);

% Find average van Rossum distance across all neurons in network
total_van_Rossum = mean(van_Rossum, 'all');
total_van_Rossum_E = mean(van_Rossum(1:N_e,1:N_e), 'all');
total_van_Rossum_I = mean(van_Rossum(N_e+1:end,N_e+1:end), 'all');

fprintf('<strong>van Rossum distance</strong>\n')
fprintf('Full network: %i\n', total_van_Rossum)
fprintf('Excitatory cells: %i\n', total_van_Rossum_E)
fprintf('Inhibitory cells: %i\n\n', total_van_Rossum_I)

fprintf('<strong>van Rossum distance normalised by firing rate</strong>\n')
fprintf('Full network: %i\n', total_van_Rossum/network_firing_rate)
fprintf('Excitatory cells: %i\n', total_van_Rossum_E/network_firing_rate_E)
fprintf('Inhibitory cells: %i\n\n', total_van_Rossum_I/network_firing_rate_I)

%% Visualise solutions             
% Plot voltages vs time
figure()   
plot(t, v)

% Plot avergae voltage vs time (axis=1 finds average across columns)
figure()   
plot(t, mean_voltage)

% Plot excitatory synaptic conductances
figure()  
plot(t, s_E)

% Plot inhibitory synaptic conductances
figure()  
plot(t, s_I)

% % Remove zeros after final spike times
% spike_timesE = spike_timesE(1:spikeCounterE,:);
% spike_timesI = spike_timesI(1:spikeCounterI,:);

% Create raster plot
figure()  
% plot(spike_timesE(:,1),spike_timesE(:,2),'.')
% plot(spike_timesI(:,1),spike_timesI(:,2),'.')
plot(t(spike_time), neuron_num,'.')
axis([0,T,1,100])
xlabel('Time')
ylabel('Neuron')

toc