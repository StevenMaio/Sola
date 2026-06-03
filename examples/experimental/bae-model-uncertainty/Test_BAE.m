clear;
close all;
clc;

% Inversion with true value of vel_coeff known
rng(192);

param_dim = 1;
state_dim = 100;
diff_coeff = 1e-1;
vel_coeff = 1.0;    % true value = 1
vel_coeff_nom = 0.6;

x = linspace(0, 1, state_dim);

prior_mean = 0;
prior_var = 1;
prior = Adv_Diff_Prior_Model(prior_mean, prior_var);

% true constraint
c = Adv_Diff_Constraint(param_dim, state_dim, diff_coeff, vel_coeff_nom);

M = c.M;

m0 = 1.0;
u0 = c.Parametric_State_Solve(m0, vel_coeff);

obs_vec = 9:5:95;
data_dim = numel(obs_vec);
noise_lvl = 10;
sigma = noise_lvl / 100 * sqrt(u0' * M * u0);

likelihood = Adv_Diff_Likelihood_Model(state_dim, obs_vec, sigma);

% actual observed data
d0 = likelihood.Observation_Operator_Apply(u0);
d0 = d0 + sigma * randn(data_dim, 1);

% Compute approximation error sample statistics
num_samples = 1000;

% sample interesting and auxiliary parameters
m_samples = randn(num_samples, 1);
l_samples = .4 + .8 * rand(num_samples, 1);
samples = zeros(num_samples, data_dim);

for t = 1:num_samples
    % compute accurate forward model for sample
    u_acc = c.Parametric_State_Solve(m_samples(t), l_samples(t));
    d_acc = likelihood.Observation_Operator_Apply(u_acc);

    % compute forward of approx model
    u_approx = c.State_Solve(m_samples(t));
    d_approx = likelihood.Observation_Operator_Apply(u_approx);

    model_err = d_acc - d_approx;
    samples(t, :) = model_err;
end

% compute error sample statistics
error_mean = mean(samples);
error_cov = cov(samples);

bae_likelihood = BAE_Likelihood_Model(likelihood, error_mean, error_cov);

u_nom = c.State_Solve(1.0);
f_nom = likelihood.Observation_Operator_Apply(u_nom);

% Compute posterior statistics w/o BAE
map_rhs = f_nom' * likelihood.Noise_Precision_Apply(d0);
nom_post_var = 1 / (prior_var^(-2) + f_nom' * likelihood.Noise_Precision_Apply(f_nom));
nom_post_mean = map_rhs * nom_post_var;

% Compute posterior statistics w/ BAE
map_rhs = f_nom' * bae_likelihood.Noise_Precision_Apply(d0);
bae_post_var = 1 / (prior_var^(-2) + f_nom' * bae_likelihood.Noise_Precision_Apply(f_nom));
bae_post_mean = map_rhs * bae_post_var;

% Plot posterior densities
pdf_support = linspace(0.5, 1.2, 500);
prior_rho = normpdf(pdf_support, prior_mean, prior_var);
bae_post_rho = normpdf(pdf_support, bae_post_mean, bae_post_var);
nom_post_rho = normpdf(pdf_support, nom_post_mean, nom_post_var);

figure
hold
plot(pdf_support, nom_post_rho);
plot(pdf_support, bae_post_rho, '-.');
xline(m0, '--');
legend({'No BAE', 'BAE', 'True m'})
