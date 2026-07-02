% Test 
% 
%
clear;
close all;
mkdir figures;

addpath('../');
addpath('../Test_Nonlinear_Poisson/');
% Initialize problem
dim = 100;
x = linspace(0, 1, dim)';
h = x(2) - x(1);
f = 4 * x - 1;

cons = Nonlinear_Poisson_Constraint(dim, f);

M = cons.M;
S = cons.S;
K = cons.Construct_Stiffness_Matrix(ones(dim, 1));

m0 = 0.9 * x.^2 + 1e-1;
u0 = cons.State_Solve(m0);

obs_vec = (3:4:99)';

% initialize data
d0 = u0(obs_vec);
noise_lvl = 1;
sigma = noise_lvl * 1e-2 * (max(u0) - min(u0));
d0 = d0 + sigma * randn(size(d0));

figure
plot(x, m0);
hold
plot(x, u0);
scatter(x(obs_vec), d0);
legend({'$m_0$', '$u_0$', '$d_0$'}, 'Interpreter', 'latex')

likelihood = BAE_Test_Likelihood(dim, obs_vec, sigma);
likelihood.d = d0;

% initialize prior
scale = 1.5;
prior = Poisson_Prior_Model(cons, scale * 1e-1, scale);

% initialize inversion problem
inversion_problem = Bayesian_Inversion(likelihood, prior, cons);
inversion_problem.opt.Gauss_Newton_Hess = false;
inversion_problem.opt.iteration_limit = 250;
inversion_problem.opt.max_cg_iter = 150;

init_guess = .7 * x + .2;
[u_map, m_map] = inversion_problem.Compute_MAP_Point(init_guess);

figure
hold
plot(x, m0)
plot(x, m_map)
legend({'$m_0$', '$m_\mathrm{MAP}$'}, 'Interpreter', 'latex');

diff = m0 - m_map;
fprintf('m_map err: %.4e\n', diff' * M * diff);

diff = m0 - init_guess;
fprintf('Init_guess error: %.4e\n', diff' * M * diff);

figure
u_init = cons.State_Solve(init_guess);
hold
plot(x, u0)
plot(x, u_map)
legend({'$u_0$', '$u_\mathrm{MAP}$'}, ...
    'interpreter', 'latex');


fig_names = {
    'problem_setup'
    'parameter_comparison'
    'state_comparison'
    };
problem = 'No_BAE';

for i = 1:3
    fig = figure(i);
    filename = sprintf('figures/%s_%s.pdf', problem, fig_names{i});
    exportgraphics(gcf, filename);
end
