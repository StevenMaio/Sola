clear;
close all;
clc;

addpath('../');
addpath('../Test_Recover_Linear_Map/');
%% Test the simple forward problem
rng(192);

dim = 100;
x = linspace(0, 1, dim)';

m_nom = x .* (1 - x) * 0.2 + 1;

cons = Nonlinear_Poisson_Constraint(dim);
f = cons.f;
M = cons.M;
S = cons.S;
u_nom = cons.State_Solve(m_nom);
linearized_cons = Linearized_Poisson_Constraint(dim, m_nom, u_nom, ...
    cons.Construct_Matrix(m_nom), f, M);


scale = 0.5;   
prior = Poisson_Prior_Model(cons, scale * (1/30), scale * 1);

% Compare linearization
h = 0.5e-1;
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

m0 = m_nom - 0.2 * x;
obs_vec = (5:5:95)';
u0 = cons.State_Solve(m0);
d0 = u0(obs_vec);
noise_lvl = 5;

sigma = (noise_lvl / 100) * (max(d0) - min(d0));
d0 = d0 + sigma * randn(size(d0));
data_dim = numel(d0);

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
legend({'$m_\mathrm{nom}$', '$m_\mathrm{true}$'}, ...
       'Interpreter', 'latex');

likelihood = BAE_Test_Likelihood(dim, obs_vec, sigma);
likelihood.d = d0;

num_samples = 10000;

bae_likelihood = BAE_Params_Only_Likelihood(cons, linearized_cons, ...
                                            likelihood, prior, num_samples, ...
                                            d0, dim, dim, data_dim);
bae_cons = BAE_Correction_Constraint(linearized_cons, bae_likelihood);

bae_inversion_problem = Bayesian_Inversion(bae_likelihood, prior, bae_cons);
bae_inversion_problem.opt.iteration_limit = 100;
bae_inversion_problem.opt.max_cg_iter = 100;
bae_inversion_problem.opt.Gauss_Newton_Hess = true;


inversion_problem = Bayesian_Inversion(likelihood, prior, linearized_cons);
inversion_problem.opt.iteration_limit = 100;
inversion_problem.opt.max_cg_iter = 100;

[~, bae_m_map] = bae_inversion_problem.Compute_MAP_Point(m_nom);
[~, m_map] = inversion_problem.Compute_MAP_Point(m_nom);

figure;
hold;
plot(x, m0);
plot(x, bae_m_map);
plot(x, m_map);
legend({'True $m$', '$m_\mathrm{BAE}$', '$m_\mathrm{No BAE}$'}, 'Interpreter', 'latex');

diff = bae_m_map - m0;
disp(diff' * M * diff);
diff = m_map - m0;
disp(diff' * M * diff);

figure;
plot(x, u0);
hold;
plot(x, cons.State_Solve(bae_m_map));
plot(x, cons.State_Solve(m_map));
plot(x, linearized_cons.State_Solve(bae_m_map));
plot(x, linearized_cons.State_Solve(m_map));
scatter(x(obs_vec), d0);
legend({'$u(m_\mathrm{true})$', '$u(m_\mathrm{BAE})$', '$u(m_\mathrm{No BAE})$', ...
    '$\tilde{u}(m_\mathrm{BAE})$', '$\tilde{u}(m_\mathrm{No BAE})$'}, 'Interpreter', 'latex');