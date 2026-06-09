clear;
close all;

addpath('../');

dim = 100;
x = linspace(0, 1, 100)';
l = 1e-2;

cons = Nonlinear_Poisson_Constraint(dim, l);
M = cons.M;
m_nom = l^2 * (x.^2) .* (1 - x).^2 + 1;

u_nom = cons.State_Solve(m_nom);

figure;
hold;
plot(x, x .* (1 - x) / 2);
plot(x, u_nom);
legend({'True u', 'Estimate'});

diff = u_nom - (x .* (1 - x));
disp(diff' * M * diff);

linearized_cons = Linearized_Poisson_Constraint(cons, m_nom, u_nom);

scale = .8;
prior = Poisson_Prior_Model(cons, scale * 3e-4, scale * 1);

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

m0 = m_nom + h * sin(4 * pi * x);
obs_vec = (5:5:95)';
u0 = cons.State_Solve(m0);
d0 = u0(obs_vec);
noise_lvl = 1;

sigma = (noise_lvl / 100) * (max(d0) - min(d0));
d0 = d0 + sigma * randn(size(d0));

likelihood = BAE_Test_Likelihood(dim, obs_vec, sigma);
likelihood.d = d0;

num_samples = 1000;

bae_likelihood = BAE_Params_Only_Likelihood( ...
                                            cons, linearized_cons, likelihood, prior, num_samples, d0);

bae_inversion_problem = Bayesian_Inversion(bae_likelihood, prior, linearized_cons);
bae_inversion_problem.opt.iteration_limit = 100;
bae_inversion_problem.opt.max_cg_iter = 100;
bae_inversion_problem.opt.Gauss_Newton_Hess = true;

inversion_problem = Bayesian_Inversion(likelihood, prior, linearized_cons);
inversion_problem.opt.iteration_limit = 100;
inversion_problem.opt.max_cg_iter = 100;
inversion_problem.opt.Gauss_Newton_Hess = true;

[~, bae_m_map] = bae_inversion_problem.Compute_MAP_Point(m_nom);
[~, m_map] = inversion_problem.Compute_MAP_Point(m_nom);

figure;
hold;
plot(x, m0);
plot(x, bae_m_map);
plot(x, m_map);
legend({'$m_\mathrm{true}$', '$m_\mathrm{BAE}$', '$m_\mathrm{No BAE}$'}, 'Interpreter', 'latex');

diff = m_map - m0;
disp(diff' * M * diff);

diff = bae_m_map - m0;
disp(diff' * M * diff);

figure;
plot(x, u0);
hold;
plot(x, cons.State_Solve(bae_m_map));
plot(x, cons.State_Solve(m_map));
scatter(x(obs_vec), d0);
legend({'$u(m_\mathrm{true})$', '$u(m_\mathrm{BAE})$', '$u(m_\mathrm{No BAE})$'}, 'Interpreter', 'latex');

diff = m_map - bae_m_map;
disp(diff' * M * diff);
