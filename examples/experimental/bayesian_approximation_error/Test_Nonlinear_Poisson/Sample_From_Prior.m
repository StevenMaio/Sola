%% Test the simple forward problem
rng(192);

dim = 100;
x = linspace(0, 1, dim)';

m_nom = x .* (1 - x) * 0.1 + 1;

cons = Nonlinear_Poisson_Constraint(dim);

scale = 3;   
prior = Poisson_Prior_Model(cons, scale * (1/30), scale * (5/6));

figure
hold
for i = 1:10
    z = prior.Sample();
    plot(x, z)
end