function van_Rossum = compute_van_Rossum_distance(spike_matrix,t,t_R)
% Computes the van Rossum distance between a set of spike trains
% spike_matrix - matrix containing spike trains, each row corresponds to 
% different neuron/spike train
% t - time vector (time points for columns of spike_matrix)
% t_R - time constant of exponential kernel

N = size(spike_matrix,1);
dt = (t(end)-t(1))/(length(t)-1);
van_Rossum = zeros(N);
waveforms = zeros(size(spike_matrix));
 
% Construct kernel 
kernel = exp(-t/t_R);

% Convolve spike trains with kernel 
% (2D convolution with 1 as column convolution, i.e. no convolution)
for j=1:N
    waveforms(j,:) = conv(full(spike_matrix(j,:)), kernel, 'valid');
end

% Compute van Rossum distance between each pair of spike trains
for j=1:N
    waveform_differences = waveforms - waveforms(j,:);
    van_Rossum(j,:) = sqrt(trapz(dt,waveform_differences'.^2)./t_R);
end

end

