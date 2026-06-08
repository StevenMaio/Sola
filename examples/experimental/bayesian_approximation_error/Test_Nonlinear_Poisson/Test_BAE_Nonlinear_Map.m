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

scale = 0.05;
prior = Poisson_Prior_Model(cons, scale * 0.05, scale);

% Compare linearization
h = 1e-1;
delta_m = h * prior.Sample();

u = cons.State_Solve(m_nom + delta_m);
u_tilde = linearized_cons.State_Solve(m_nom + delta_m);

figure;
plot(x, u_nom);
hold;
plot(x, u);
plot(x, u_tilde);
legend({'$u(m_\mathrm{nom})$', '$u(m_\mathrm{nom}+\delta m)$', '$\tilde{u}(m_\mathrm{nom}+\delta m)$'}, ...
       'Interpreter', 'latex');

diff = u - u_tilde;
disp(diff' * M * diff);

% Bayesian Inversion Stuff

m0 = m_nom + h * sin(pi * x);
obs_vec = (5:5:95)';
u0 = cons.State_Solve(m0);
d0 = u0(obs_vec);
noise_lvl = 5;

sigma = (noise_lvl / 100) * (max(d0) - min(d0));
d0 = d0 + sigma * randn(size(d0));

figure;
hold;
plot(x, u0);
plot(x, u_nom);
scatter(x(obs_vec), d0);
legend({'$u_\mathrm{true}$', '$u_\mathrm{nom}$', '$\mathbf{d}$'}, ...
       'Interpreter', 'latex');

figure;
plot(x, m_nom);
hold;
plot(x, m0);
legend({'$m_\mathrm{true}$', '$m_\mathrm{nom}$'}, ...
       'Interpreter', 'latex');

likelihood = BAE_Test_Likelihood(dim, obs_vec, sigma);
likelihood.d = d0;

num_samples = 1000;

bae_likelihood = BAE_Params_Only_Likelihood( ...
                                            cons, linearized_cons, likelihood, prior, num_samples, d0);

inversion_problem = Bayesian_Inversion(bae_likelihood, prior, linearized_cons);
inversion_problem.opt.iteration_limit = 100;
inversion_problem.opt.max_cg_iter = 100;

[u_map, m_map] = inversion_problem.Compute_MAP_Point(m_nom);

figure;
hold;
plot(x, m0);
plot(x, m_map);
legend({'$m_\mathrm{true}$', '$m_\mathrm{MAP}$'}, 'Interpreter', 'latex');

diff = m_map - m0;
disp(diff' * M * diff);

figure;
plot(x, u0);
hold;
plot(x, cons.State_Solve(m_map));
scatter(x(obs_vec), d0);
legend({'$u(m_\mathrm{true})$', '$u(m_\mathrm{MAP})$'}, 'Interpreter', 'latex');
