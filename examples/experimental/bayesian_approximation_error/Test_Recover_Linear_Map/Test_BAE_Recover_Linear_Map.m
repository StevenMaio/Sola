clear;
close all;
clc;

addpath('../');
addpath('../Test_Nonlinear_State/')

mkdir figures;
%% Test the simple forward problem
rng(192);

param_dim = 100;
h = 1 / (param_dim - 1);
state_dim = 500;
noise_lvl = 1;
num_samples = 1000;

S = diag(2 * ones(1, param_dim)) + (-1) * diag(ones(1, param_dim - 1), 1) + (-1) * diag(ones(1, param_dim - 1), -1);
S(1, 1) = .5 * S(1, 1);
S(end, end) = .5 * S(end, end);
S = (1 / h) * S;

A = randn(state_dim, param_dim);

cons = Nonlinear_Poisson_Constraint(param_dim);
prior = Poisson_Prior_Model(cons, 2e-1, 1);

x = linspace(0, 1, param_dim)';
m0 = sin(pi*x);
u0 = A * m0;
sigma = (noise_lvl / 100) * (max(u0) - min(u0));
d0 = u0 + sigma * randn(size(u0));

full_cons = Linear_Constraint(A);
approx_cons = Zero_Constraint(param_dim, state_dim);

likelihood = BAE_Test_Likelihood(state_dim, 1:state_dim, sigma);
likelihood.d = d0;

bae_likelihood = BAE_Likelihood_Model(full_cons, approx_cons, likelihood, ...
                                      prior, num_samples, d0, ...
                                      param_dim, state_dim, state_dim);

F_tilde = zeros(state_dim, param_dim);
for i = 1:param_dim
    v = zeros(param_dim, 1);
    v(i) = 1;
    F_tilde(:, i) = bae_likelihood.Apply_Linear_Correction(v);
end

fprintf('BAE Diff: l2=%.4e max=%.4e\n', norm(F_tilde - A), max(max(abs(F_tilde-A))));
fprintf('BAE Diff: rel_l2=%.4e rel_max=%.4e\n', norm(F_tilde - A)/norm(A), max(max(abs(F_tilde-A)))/max(max(abs(A))));

bae_cons = BAE_Correction_Constraint(approx_cons, bae_likelihood);

bae_inversion_problem = Bayesian_Inversion(bae_likelihood, prior, bae_cons);
bae_inversion_problem.opt.Gauss_Newton_Hess = false;

inversion_problem = Bayesian_Inversion(likelihood, prior, approx_cons);
inversion_problem.opt.Gauss_Newton_Hess = false;

[~, bae_m_map] = bae_inversion_problem.Compute_MAP_Point(zeros(param_dim, 1));
[~, m_map] = inversion_problem.Compute_MAP_Point(zeros(param_dim, 1));

figure(1)
plot(x, m0)
hold
plot(x, bae_m_map)
plot(x, m_map)
legend({'$m_\mathrm{true}$', '$m_\mathrm{BAE}$', '$m_\mathrm{No BAE}$'}, ...
    'interpreter', 'latex', 'FontSize', 20)

set(gca, 'fontsize', 20)


exportgraphics(gcf, 'figures/recover_linear_map.pdf');