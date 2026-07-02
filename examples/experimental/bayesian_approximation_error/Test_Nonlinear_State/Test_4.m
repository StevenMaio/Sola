%%
% Test Nonlinear Poisson OED w/ BAE
%   In this experiment, we utilize BAE in order to simplify the OED problem.
%
%   author: Steven Maio
clear;
close all;
mkdir figures;

addpath('../');

rng(100);

dim = 100;
x = linspace(0, 1, dim)';

cons = Nonlinear_Poisson_Constraint(dim);
M = cons.M;

% zero surrogate
m_nom = 0 * x;
u_nom = 0 * x;
linearized_cons = Linearized_Poisson_Constraint(cons, m_nom, u_nom);

scale = 2;
prior = Poisson_Prior_Model(cons, scale * 8e-2, scale);

obs_vec = (5:5:95)';
data_dim = numel(obs_vec);
d0 = 0 * obs_vec;
sigma = 1e-2;
likelihood = BAE_Test_Likelihood(dim, obs_vec, sigma);

% set up BAE likelihood and constraint
num_samples = 1000;
bae_likelihood = BAE_Likelihood_Model(cons, linearized_cons, likelihood, ...
                                      prior, num_samples, d0, ...
                                      dim, dim, data_dim);
bae_cons = BAE_Correction_Constraint(linearized_cons, bae_likelihood);

% Solve OED problem -- can't use lazy evaluation
inversion_problem = Bayesian_Inversion(bae_likelihood, prior, bae_cons);
oed = Linear_OED_D_Opt(inversion_problem, 5, false);

[sensors, eig] = oed.Optimize_Design();

disp(sensors)
