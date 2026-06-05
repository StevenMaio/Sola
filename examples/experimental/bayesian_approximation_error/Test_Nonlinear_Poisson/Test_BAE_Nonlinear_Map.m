clear;
close all;
clc;

addpath('../');
%% Test the simple forward problem
rng(192);

dim = 100;
x = linspace(0, 1, dim)';

m_nom = x .* (1 - x) * 0.1 + 1;

cons = Nonlinear_Poisson_Constraint(dim);
M = cons.M;
u_nom = cons.State_Solve(m_nom);
linearized_cons = Linearized_Poisson_Constraint(dim, m_nom, u_nom, M);

figure
plot(x, u_nom);

% Compare linearization
h = 1e-1;
delta_m = h * randn(dim, 1);

u = cons.State_Solve(m_nom + delta_m);
u_tilde = linearized_cons.State_Solve(m_nom + delta_m);

figure
plot(x, u_nom)
hold
plot(x, u)
plot(x, u_tilde)

diff = u - u_tilde;
disp(diff' * M * diff)

% Bayesian Inversion Stuff

m0 = m_nom + h * sin(pi*x);
obs_vec = (5:5:95)';
u0 = cons.State_Solve(m0);
d0 = u0(obs_vec);
noise_lvl = 5;

figure
hold
plot(x, u0)
plot(x, u_nom)
scatter(x(obs_vec), d0)
legend({'True State', 'Nominal State', 'Data'})

figure
plot(x, m_nom)
hold
plot(x, m0)
legend({'True m', 'Nominal m'})


sigma = (noise_lvl / 100) * (max(d0) - min(d0));
d0 = d0 + sigma * randn(size(d0));

likelihood = BAE_Test_Likelihood(dim, obs_vec, sigma);

num_samples = 1000;

prior = Poisson_Prior_Model(cons, 0.05, 0.2);
% prior = Gaussian_Distribution(zeros(dim, 1), (.2)^2 * eye(dim));

bae_likelihood = BAE_Params_Only_Likelihood(...
  cons, linearized_cons, likelihood, prior, num_samples, d0);

inversion_problem = Bayesian_Inversion(bae_likelihood, prior, linearized_cons);
inversion_problem.opt.iteration_limit = 100;
inversion_problem.opt.max_cg_iter = 100;

[u_map, m_map] = inversion_problem.Compute_MAP_Point(m_nom);

figure
hold
plot(x, m0)
plot(x, m_map)
legend({'Truth', 'MAP'})