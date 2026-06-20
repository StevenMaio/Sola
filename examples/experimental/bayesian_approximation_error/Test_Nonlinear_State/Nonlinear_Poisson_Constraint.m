classdef Nonlinear_Poisson_Constraint < Constraint

    properties
        dim     % dimension of state and parameter
        x       % domain discretization
        M       % mass matrix
        M0      % mass matrix with lifting
        S       % H1 inner product matrix
        S0      % H1 inner product matrix with lifting
        l
    end

    methods (Access = public)
        % abstract methods that need to be implemented

        function [u] = State_Solve(this, m)
            u = m;
            u(1) = 0;
            u(end) = 0;
            for i = 1:15
                res = this.S0 * u + this.M * (u.^2) - this.M0 * m;
                d = this.c_u_Inverse_Apply(-res, u, m);
                u = u + d;
            end
        end

        function [u_out] = c_u_Transpose_Inverse_Apply(this, u_in, u, z)
            L = this.S0 + 2 * this.M * diag(u);
            u_out = linsolve(L', u_in);
        end

        function [z_out] = c_z_Transpose_Apply(this, u_in, u, z)
            z_out = -this.M * u_in;
        end

        function [u_out] = c_u_Inverse_Apply(this, u_in, u, z)
            L = this.S0 + 2 * this.M * diag(u);
            u_out = linsolve(L, u_in);
        end

        function [u_out] = c_z_Apply(this, z_in, u, z)
            u_out = -this.M0 .* z_in;
        end

        function [con] = c(this, u, z)
            con = this.S0 * u + this.M * (u.^2) - this.M0 * z;
        end

        function [u_out] = c_uu_Apply(this, u_in, u, z, lambda)
            u_out = 2 * this.M * (u_in .* lambda);
        end

        function [u_out] = c_uz_Apply(this, z_in, u, z, lambda)
            u_out = zeros(this.dim, size(z_in, 2));
        end

        function [z_out] = c_zu_Apply(this, u_in, u, z, lambda)
            z_out = zeros(this.dim, size(u_in, 2));
        end

        function [z_out] = c_zz_Apply(this, z_in, u, z, lambda)
            z_out = zeros(this.dim, size(z_in, 2));
        end

    end

    methods

        function this = Nonlinear_Poisson_Constraint(dim, l)
            arguments
                dim
                l = 1
            end
            this@Constraint();
            this.x = linspace(0, 1, dim);
            this.dim = dim;

            h = this.x(2) - this.x(1);

            M = diag(4 * ones(1, dim)) + diag(ones(1, dim - 1), 1) + diag(ones(1, dim - 1), -1);
            M(1, 1) = .5 * M(1, 1);
            M(end, end) = .5 * M(end, end);
            M = (1 / 6) * h * M;
            this.M = M;

            M0 = M;
            M0(1, :) = 0 * M0(1, :);
            M0(end, :) = 0 * M0(end, :);
            this.M0 = M0;

            S = diag(2 * ones(1, dim)) + (-1) * diag(ones(1, dim - 1), 1) + (-1) * diag(ones(1, dim - 1), -1);
            S(1, 1) = .5 * S(1, 1);
            S(end, end) = .5 * S(end, end);
            S = (1 / h) * S;
            this.S = S;

            S0 = S;
            S0(1, :) = 0 * S0(1, :);
            S0(end, :) = 0 * S0(end, :);
            S0(:, 1) = 0 * S0(:, 1);
            S0(:, end) = 0 * S0(:, end);
            S0(1, 1) = 1;
            S0(end, end) = 1;
            this.S0 = S0;
        end

    end

end
