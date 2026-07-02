%%
% Test Nonlinear Poisson Inversion
%   Test inversion of nonlinear Poisson problem using accurate model
%
%   author: Steven Maio
clear;
close all;
mkdir figures;

addpath('../');

rng(100);

dim = 100;
x = linspace(0, 1, dim)';

m_nom = 0 * x;
u_nom = 0 * x;

cons = Nonlinear_Poisson_Constraint(dim);
M = cons.M;
h = 5e-1;
m0 = h * x;

linearized_cons = Linearized_Poisson_Constraint(cons, m_nom, u_nom);

u0 = cons.State_Solve(m0);

diff = u0 - (x .* (1 - x));
disp(diff' * M * diff);

obs_vec = (5:5:95)';
sigma = (max(u0) - min(u0)) * 1e-2;
d0 = u0(obs_vec) + sigma * randn(size(obs_vec));
data_dim = numel(d0);

figure(1);
hold;
plot(x, u0);
scatter(x(obs_vec), d0);
legend({'$u$', '$d$'}, 'Interpreter', 'latex', 'FontSize', 20);

likelihood = BAE_Test_Likelihood(dim, obs_vec, sigma);
likelihood.d = d0;
scale = 2;
prior = Poisson_Prior_Model(cons, scale * 8e-2, scale);

num_samples = 1000;

bae_likelihood = BAE_Likelihood_Model(cons, linearized_cons, likelihood, ...
                                      prior, num_samples, d0, ...
                                      dim, dim, data_dim);

bae_cons = BAE_Correction_Constraint(linearized_cons, bae_likelihood);

bae_inversion_problem = Bayesian_Inversion(bae_likelihood, prior, bae_cons);
bae_inversion_problem.opt.iteration_limit = 100;
bae_inversion_problem.opt.max_cg_iter = 500;

inversion_problem = Bayesian_Inversion(likelihood, prior, cons);
inversion_problem.opt.use_trust_region = false;
inversion_problem.opt.iteration_limit = 100;
inversion_problem.opt.max_cg_iter = 500;

[u_map, m_map] = inversion_problem.Compute_MAP_Point(ones(dim, 1));
[u_bae, m_bae] = bae_inversion_problem.Compute_MAP_Point(ones(dim, 1));

diff = m_map - m0;
fprintf('No BAE Error: abs=%.4e; rel=%.4e\n', diff' * M * diff, diff' * M * diff / (m0' * M * m0));

diff = m_bae - m0;
fprintf('BAE Error: %.4e; rel= %.4e\n', diff' * M * diff,  diff' * M * diff / (m0' * M * m0));

figure(2)
hold
plot(x, m0)
plot(x, m_bae, '--')
plot(x, m_map, '-.')
legend({'$m_0$', '$m_\mathrm{BAE}$', '$m_\mathrm{Acc}$'}, ...
    'Interpreter', 'latex', 'FontSize', 24, 'Location', 'southeast')

figure(3)
hold
plot(x, u0)
plot(x, u_bae(1:dim), '--')
plot(x, u_map, '-.')
legend({'$u(m_0)$', '$u(m_\mathrm{BAE})$', '$u(m_\mathrm{Acc})$'}, ...
    'Interpreter', 'latex', 'FontSize', 24, 'Location', 'south')

fig_names = {
    'data'
    'parameter_comparison'
    'state_comparison'
};
problem = 'Test1';

for i = 1:3
    fig = figure(i);
    filename = sprintf('figures/%s_%s.pdf', problem, fig_names{i});
    set(gca, 'fontsize', 20)
    exportgraphics(gcf, filename);
end
