clear;
close all;
clc;

addpath('../');

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

prior = Gaussian_Distribution(prior_mean, prior_var);
aux_distr = Uniform_Distribution(0.4, 1.2);

% Parametric constraint with params set to nominal params
c = Adv_Diff_Constraint(param_dim, state_dim, diff_coeff, vel_coeff_nom);

M = c.M;

m0 = 1.0;
u0 = c.Parametric_State_Solve(m0, vel_coeff);   % true state
u_nom = c.State_Solve(m0);

obs_vec = 5:5:95;
data_dim = numel(obs_vec);
noise_lvl = 10;
sigma = noise_lvl / 100 * sqrt(u0' * M * u0);

likelihood = BAE_Test_Likelihood(state_dim, obs_vec, sigma);

% actual observed data
d0 = likelihood.Observation_Operator_Apply(u0);
d0 = d0 + sigma * randn(data_dim, 1);

figure;
plot(x, u0);
hold;
plot(x, u_nom, '--');
scatter(x(obs_vec), d0);
legend({'True State', 'Approx State', 'Data'});

% Compute approximation error sample statistics
num_samples = 1000;

bae_likelihood = BAE_Aux_Params_Only_Likelihood( ...
                                                c, c, likelihood, prior, aux_distr, num_samples, d0);

u_nom = c.State_Solve(1.0);
f_nom = likelihood.Observation_Operator_Apply(u_nom);

% Compute posterior statistics w/o BAE
map_rhs = f_nom' * likelihood.Noise_Precision_Apply(d0);
nom_post_var = 1 / (prior_var^(-2) + f_nom' * likelihood.Noise_Precision_Apply(f_nom));
nom_post_mean = map_rhs * nom_post_var;

% Compute posterior statistics w/ BAE
map_rhs = f_nom' * (bae_likelihood.Noise_Precision_Apply(d0) - bae_likelihood.Get_Error_Mean());
bae_post_var = 1 / (prior_var^(-2) + f_nom' * bae_likelihood.Noise_Precision_Apply(f_nom));
bae_post_mean = map_rhs * bae_post_var;

% Plot posterior densities
pdf_support = linspace(0.5, 1.2, 500);
prior_rho = normpdf(pdf_support, prior_mean, prior_var);
bae_post_rho = normpdf(pdf_support, bae_post_mean, bae_post_var);
nom_post_rho = normpdf(pdf_support, nom_post_mean, nom_post_var);

figure;
hold;
plot(pdf_support, nom_post_rho);
plot(pdf_support, bae_post_rho, '-.');
xline(m0, '--');
legend({'No BAE', 'BAE', 'True m'});

%% Try using Sola tools
inversion_problem = Bayesian_Inversion(bae_likelihood, prior, c);
inversion_problem.opt.Gauss_Newton_Hess = true;
[u_map, m_map] = inversion_problem.Compute_MAP_Point(0);
