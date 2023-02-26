function chi = compute_chi(v)
% Computes the synchrony measure chi for a given set of voltage traces
% v - voltage traces, each row is a different neuron

% Calculate the average voltage as a function of time
mean_voltage = mean(v,2);

% Calculate the variance of each trace and the average voltage
ind_variance = mean(v.^2) - mean(v).^2;
total_variance = mean(mean_voltage.^2) - mean(mean_voltage)^2;

% Calculate chi 
chi = sqrt(total_variance^2 / mean(ind_variance.^2));

end

