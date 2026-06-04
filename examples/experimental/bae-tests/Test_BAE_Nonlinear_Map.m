clear;
close all;
clc;

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

solution_op = @(m) exp(-m*x);
% solution_op = @(m) m*exp(-1*x);
approx_model = @(m) m*exp(-x);

u0 = solution_op(m0);
d0 = u0(obs_vec);
sigma = noise_lvl / 100 * (max(d0) - min(d0));

d0 = d0 + sigma * randn(size(d0));

likelihood = Adv_Diff_Likelihood_Model(state_dim, obs_vec, sigma);

figure
plot(x, u0);
hold
scatter(x(obs_vec), d0);

for i = 1:10
    plot(x, solution_op(prior.Sample()), '--');
end

legend({'True State', 'Data'})
title('Truth and Samples');

%% BAE
num_samples = 100;
full_F = @(m) likelihood.Observation_Operator_Apply(solution_op(m));
approx_F = @(m) likelihood.Observation_Operator_Apply(approx_model(m));

bae_likelihood = BAE_Params_Only_Likelihood(...
    full_F, approx_F, likelihood, prior, num_samples, d0);

f_tilde = approx_F(1) + bae_likelihood.Apply_Linear_Correction(1);

map_rhs = bae_likelihood.Get_Observed_Data();
map_rhs = f_tilde' * bae_likelihood.Noise_Precision_Apply(map_rhs) + prior_mean / prior_std^2;

bae_post_var = 1 / (f_tilde' * bae_likelihood.Noise_Precision_Apply(f_tilde) + prior_std^(-2));
bae_map_estimate = bae_post_var * map_rhs;

fprintf('BAE - m_map=%.4f, var=%.4f\n', bae_map_estimate, bae_post_var);

%% No BAE
f = approx_F(1);
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
