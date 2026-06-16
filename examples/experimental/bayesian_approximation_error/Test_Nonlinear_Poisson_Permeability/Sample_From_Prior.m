dim = 100;
x = linspace(0, 1, 100)';
m_nom = x;

cons = Nonlinear_Poisson_Constraint(dim, ones(dim, 1));
scale = 1;
prior = Poisson_Prior_Model(cons, scale * 2e-2, scale);
prior.mean = m_nom;

figure
hold

N = 100;
m_samples = zeros(N, dim);

for i = 1:N
    m = prior.Sample();
    plot(x, m);
    m_samples(i, :) = m;
end

figure
plot(x, mean(m_samples))