% Test 
% 
%
dim = 100;
x = linspace(0, 1, dim)';
h = x(2) - x(1);
f = 4 * x - 1;

cons = Nonlinear_Poisson_Constraint(dim, f);

M = cons.M;
S = cons.S;
K = cons.Construct_Stiffness_Matrix(ones(dim, 1));

m0 = x;

u0 = x .* (1 - x);
u = cons.State_Solve(m0);

figure
plot(x, u0);
hold
plot(x, u);
legend({'Truth', 'Estimate'})

diff = u0 - u;
disp(diff' * M * diff)

% Test approximate solve
linearized_cons = Linearized_Poisson_Constraint(cons, m0, u0);

dm = 5e-2 * sin(pi * x);

u = cons.State_Solve(m0 + dm);
u_approx = linearized_cons.State_Solve(m0 + dm);

figure
hold
plot(x, u0)
plot(x, u)
plot(x, u_approx)
legend({'Nominal', 'HF', 'LF'})