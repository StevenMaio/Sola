clear;
close all;
clc;

% Incorrect Model test. Demonstrating the 
rng(192);

param_dim = 1;
state_dim = 100;
diff_coeff = 1e-1;
vel_coeff = 0.6;    % true value = 1

x = linspace(0, 1, state_dim);

prior_mean = 0;
prior_var = 1;
prior = Adv_Diff_Prior_Model(prior_mean, prior_var);

% true constraint
c0 = Adv_Diff_Constraint(param_dim, state_dim, diff_coeff, vel_coeff);

M = c0.M;

m0 = 1.0;
u0 = c0.State_Solve(m0);

obs_vec = 9:5:95;
noise_lvl = 10;
sigma = noise_lvl / 100 * sqrt(u0' * M * u0);

likelihood = Adv_Diff_Likelihood_Model(state_dim, obs_vec, sigma);

d0 = likelihood.Observation_Operator_Apply(u0);
d0 = d0 + sigma * randn(numel(obs_vec), 1);

figure
plot(x, u0);
hold
scatter(x(obs_vec), d0);

f = likelihood.Observation_Operator_Apply(u0); % coincidence here

post_var = 1 / (prior_var^(-2) + f' * likelihood.Noise_Precision_Apply(f));
map_rhs = f' * likelihood.Noise_Precision_Apply(d0);
map_estimate = post_var * map_rhs;

figure;

pdf_support = linspace(0.5, 1.2, 500);
post_rho = normpdf(pdf_support, map_estimate, post_var);
plot(pdf_support, post_rho);
hold;
xline(m0);
