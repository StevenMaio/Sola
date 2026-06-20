dim = 100;
x = linspace(0, 1, 100)';
cons = Nonlinear_Poisson_Constraint(dim);
scale = 3;
prior = Poisson_Prior_Model(cons, scale * 5e-2, scale);

figure
hold

for i = 1:5
    m = prior.Sample();
    plot(x, m);
end