% Test 
% 
%
clear;
close all;

addpath('../');
%
rng(1003)
dim = 100;
x = linspace(0, 1, dim)';
h = x(2) - x(1);
f = 4 * x - 1;

cons = Nonlinear_Poisson_Constraint(dim, f);

M = cons.M;
S = cons.S;
K = cons.Construct_Stiffness_Matrix(ones(dim, 1));

m_nom = x;
u_nom = x .* (1 - x);

linearized_cons = Linearized_Poisson_Constraint(cons, m_nom, u_nom);
obs_vec = (3:3:99)';

m0 = m_nom + .1 * sin(pi * x) + 1e-3;
u0 = cons.State_Solve(m0);
d0 = u0(obs_vec);

noise_lvl = 5;
sigma = noise_lvl / 100 * (max(d0) - min(d0));
d0 = d0 + sigma * randn(size(d0));
data_dim = numel(d0);

likelihood = BAE_Test_Likelihood(dim, obs_vec, sigma);
likelihood.d = d0;

figure
hold
plot(x, m0);
plot(x, m_nom)
plot(x, u0);
scatter(x(obs_vec), d0);
legend({'$m_0$', '$m_\mathrm{nom}$', '$u_0$', '$d_0$'}, 'Interpreter', 'latex')

% initialize prior
scale = 1;
prior = Poisson_Prior_Model(cons, scale * 2e-2, scale);
prior.mean = m_nom;
num_samples = 1000;

bae_likelihood = BAE_Likelihood_Model(cons, linearized_cons, likelihood, prior, ...
    num_samples, d0, dim, dim, data_dim);
bae_cons = BAE_Correction_Constraint(linearized_cons, bae_likelihood);

inversion_problem = Bayesian_Inversion(likelihood, prior, linearized_cons);
bae_inversion_problem = Bayesian_Inversion(bae_likelihood, prior, bae_cons);
bae_inversion_problem.opt.use_trust_region = false;

inversion_problem.opt.iteration_limit = 100;
bae_inversion_problem.opt.iteration_limit = 100;

[u_map, m_map] = inversion_problem.Compute_MAP_Point(m_nom);
[u_bae, m_bae] = bae_inversion_problem.Compute_MAP_Point(m_nom);

figure
hold
plot(x, m0)
plot(x, m_nom)
plot(x, m_bae)
plot(x, m_map)
legend({'$m_0$', '$m_\mathrm{nom}$', ...
    '$m_\mathrm{BAE}$', '$m_\mathrm{No BAE}$'}, ...
    'Interpreter','latex')

figure
hold
plot(x, u0)
plot(x, u_nom)
plot(x, u_bae(1:dim))
plot(x, u_map)
legend({'$u_0$', '$u_\mathrm{nom}$', ...
    '$u_\mathrm{BAE}$', '$u_\mathrm{No BAE}$'}, ...
    'Interpreter','latex')