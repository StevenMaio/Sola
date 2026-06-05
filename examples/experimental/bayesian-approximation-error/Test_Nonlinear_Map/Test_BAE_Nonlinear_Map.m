clear;
close all;
clc;

addpath('../');
%% Test the simple forward problem
rng(192);

param_dim = 1;
state_dim = 100;
noise_lvl = 5;

m0 = 1.36;
prior_mean = 1;
prior_std = 0.5;
prior = Gaussian_Distribution(prior_mean, prior_std^2);

x = linspace(0, 1, state_dim)';
obs_vec = (5:5:100)';
data_dim = numel(obs_vec);

full_cons = Nonlinear_Constraint(x);
approx_cons = Linearized_Constraint(x);

u0 = full_cons.State_Solve(m0);
d0 = u0(obs_vec);
sigma = noise_lvl / 100 * (max(d0) - min(d0));

d0 = d0 + sigma * randn(size(d0));

likelihood = BAE_Test_Likelihood(state_dim, obs_vec, sigma);

figure
plot(x, u0);
hold
plot(x, full_cons.State_Solve(m0));
scatter(x(obs_vec), d0);

for i = 1:10
    plot(x, full_cons.State_Solve(prior.Sample()), '--');
end

legend({'True State', 'Approximate State', 'Data'})
title('Truth and Samples');

%% BAE
num_samples = 100;

bae_likelihood = BAE_Params_Only_Likelihood(...
    full_cons, approx_cons, likelihood, prior, num_samples, d0);

f = likelihood.Observation_Operator_Apply(approx_cons.State_Solve(1));
f_tilde = f + bae_likelihood.Apply_Linear_Correction(1);

map_rhs = bae_likelihood.Get_Observed_Data();
map_rhs = f_tilde' * bae_likelihood.Noise_Precision_Apply(map_rhs) + prior_mean / prior_std^2;

bae_post_var = 1 / (f_tilde' * bae_likelihood.Noise_Precision_Apply(f_tilde) + prior_std^(-2));
bae_map_estimate = bae_post_var * map_rhs;

fprintf('BAE - m_map=%.4f, var=%.4f\n', bae_map_estimate, bae_post_var);

%% No BAE
map_rhs = f' * d0 / sigma^2;

no_bae_post_var = 1 / (f' * f / sigma^2 + prior_std^(-2));
no_bae_map_estimate = no_bae_post_var * map_rhs;

fprintf('No BAE - m_map=%.4f, var=%.4f\n', no_bae_map_estimate, no_bae_post_var);

% plot the supports
pdf_support = linspace(0.5, 2, 500);
bae_post_rho = normpdf(pdf_support, bae_map_estimate, bae_post_var);
no_bae_post_rho = normpdf(pdf_support, no_bae_map_estimate, no_bae_post_var);

figure
hold
plot(pdf_support, no_bae_post_rho, '-.');
plot(pdf_support, bae_post_rho);
xline(m0, '--');
legend({'No BAE', 'BAE', 'True m'})
