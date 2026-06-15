%%
% Test Nonlinear Poisson Inversion w/ BAE
%   Test inversion of nonlinear Poisson problem using the BAE approach.
%
%   author: Steven Maio
clear;
close all;

addpath('../');

rng(100);

dim = 100;
x = linspace(0, 1, dim)';

cons = Nonlinear_Poisson_Constraint(dim);
M = cons.M;

m_nom = (x.^2) .* (1 - x).^2 + 2;
u_nom = x .* (1 - x);

diff = u_nom - (x .* (1 - x));
disp(diff' * M * diff);

linearized_cons = Linearized_Poisson_Constraint(cons, m_nom, u_nom);
scale = 3.5;
prior = Poisson_Prior_Model(cons, scale * 9e-2, scale);

% Compare linearization
h = 1e-1;
m0 = m_nom + h * sin(pi / 2 * x);

u0 = cons.State_Solve(m0);
u_tilde = linearized_cons.State_Solve(m0);

figure;
plot(x, u_nom);
hold;
plot(x, u0);
plot(x, u_tilde);
legend({'$u(m_\mathrm{nom})$', '$u(m_\mathrm{nom}+\delta m)$', '$\tilde{u}(m_\mathrm{nom}+\delta m)$'}, ...
       'Interpreter', 'latex');

diff = u0 - u_tilde;
disp(diff' * M * diff)

obs_vec = (5:5:95)';
sigma = 1e-2 * (max(u0) - min(u0));
d0 = u0(obs_vec) + sigma * randn(size(obs_vec));
data_dim = numel(d0);

figure
hold
plot(x, u0)
scatter(x(obs_vec), d0)
legend({'$u$', '$d$'}, 'Interpreter', 'latex')

likelihood = BAE_Test_Likelihood(dim, obs_vec, sigma);
likelihood.d = d0;

num_samples = 1000;

bae_likelihood = BAE_Params_Only_Likelihood( ...
                                            cons, linearized_cons, likelihood, ...
                                            prior, num_samples, d0, ...
                                            dim, dim, data_dim);

bae_cons = BAE_Correction_Constraint(linearized_cons, bae_likelihood);

bae_inversion_problem = Bayesian_Inversion(bae_likelihood, prior, bae_cons);
bae_inversion_problem.opt.iteration_limit = 250;
bae_inversion_problem.opt.max_cg_iter = 100;
bae_inversion_problem.opt.Gauss_Newton_Hess = false;

inversion_problem = Bayesian_Inversion(likelihood, prior, linearized_cons);
inversion_problem.opt.iteration_limit = 100;
inversion_problem.opt.max_cg_iter = 100;
inversion_problem.opt.Gauss_Newton_Hess = false;

[~, bae_m_map] = bae_inversion_problem.Compute_MAP_Point(ones(dim, 1));
[~, m_map] = inversion_problem.Compute_MAP_Point(ones(dim, 1));

figure;
hold;
plot(x, m0);
plot(x, bae_m_map);
plot(x, m_map);
legend({'$m_\mathrm{true}$', '$m_\mathrm{BAE}$', '$m_\mathrm{No BAE}$'}, 'Interpreter', 'latex');

diff = m_map - m0;
fprintf('No BAE Error: %.4e\n', diff' * M * diff);

diff = bae_m_map - m0;
fprintf('BAE Error: %.4e\n', diff' * M * diff);

figure;
plot(x, u0);
hold;
plot(x, cons.State_Solve(bae_m_map));
plot(x, cons.State_Solve(m_map));
scatter(x(obs_vec), d0);
legend({'$u(m_\mathrm{true})$', '$u(m_\mathrm{BAE})$', '$u(m_\mathrm{No BAE})$'}, 'Interpreter', 'latex');

diff = m_map - bae_m_map;
fprintf('MAP Estimate Diff %.4e\n', diff' * M * diff);
