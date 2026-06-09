clear;
close all;
clc;

addpath('../');
addpath('../Test_Nonlinear_Poisson/');
%% Test the simple forward problem
rng(192);

param_dim = 100;
state_dim = 500;
noise_lvl = 1;
num_samples = 1000;

poisson_cons = Nonlinear_Poisson_Constraint(param_dim);

A = randn(state_dim, param_dim);
%prior = Gaussian_Distribution(zeros(param_dim, 1), eye(param_dim));
prior = Poisson_Prior_Model(poisson_cons, 1e-3, 3);

m0 = rand(param_dim, 1);
u0 = A * m0;
sigma = (noise_lvl / 100) * (max(u0) - min(u0));
d0 = u0 + sigma * randn(size(u0));

full_cons = Linear_Constraint(A);
approx_cons = Zero_Constraint(state_dim);

likelihood = BAE_Test_Likelihood(state_dim, 1:state_dim, sigma);

bae_likelihood = BAE_Params_Only_Likelihood( ...
                                            full_cons, approx_cons, likelihood, prior, num_samples, d0);

F_tilde = zeros(state_dim, param_dim);
for i = 1:param_dim
    x = zeros(param_dim, 1);
    x(i) = 1;
    F_tilde(:, i) = bae_likelihood.Apply_Linear_Correction(x);
end

fprintf('BAE Diff: l2=%.4e max=%.4e\n', norm(F_tilde - A), max(max(abs(F_tilde-A))));
