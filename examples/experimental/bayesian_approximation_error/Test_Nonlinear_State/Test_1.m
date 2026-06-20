%%
% Test Nonlinear Poisson Inversion
%   Test inversion of nonlinear Poisson problem using accurate model
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
m0 = (x.^2) .* (1 - x).^2 + 2;

u0 = cons.State_Solve(m0);

diff = u0 - (x .* (1 - x));
disp(diff' * M * diff);

obs_vec = (5:5:95)';
sigma = (max(u0) - min(u0)) * 1e-2;
d0 = u0(obs_vec) + sigma * randn(size(obs_vec));

figure;
hold;
plot(x, u0);
scatter(x(obs_vec), d0);
legend({'$u$', '$d$'}, 'Interpreter', 'latex');

likelihood = BAE_Test_Likelihood(dim, obs_vec, sigma);
likelihood.d = d0;
scale = 3.5;
prior = Poisson_Prior_Model(cons, scale * 8e-2, scale);

inversion_problem = Bayesian_Inversion(likelihood, prior, cons);
% something is going wrong here -- why do I need to turn this off?
inversion_problem.opt.use_trust_region = false;

[u_map, m_map] = inversion_problem.Compute_MAP_Point(ones(dim, 1));

error = m0 - m_map;
disp(error' * M * error)

figure
hold
plot(x, m0)
plot(x, m_map)
legend({'$m_0$', '$m_\mathrm{MAP}$'}, 'Interpreter', 'latex')

figure
hold
plot(x, u0)
plot(x, u_map)
legend({'$u(m_0)$', '$u(m_\mathrm{MAP})$'}, 'Interpreter', 'latex')
