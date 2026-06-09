classdef Linearized_Poisson_Constraint < Constraint

    properties
        dim
        M       % mass matrix
        M0      % mass matrix with lifting
        S0      % H1 inner product matrix with lifting
        L
        m0      % linearization point
        u0      % state at linearization point
    end

    methods (Access = public)
        % abstract methods that need to be implemented

        function [u] = State_Solve(this, m)
            rhs = this.M0 * (m - this.m0);
            u = this.u0 + linsolve(this.L,  rhs);
        end

        function [u_out] = c_u_Transpose_Inverse_Apply(this, u_in, u, z)
            rhs = this.L' * this.M * u_in;
            u_out = linsolve(this.M, rhs);
        end

        function [z_out] = c_z_Transpose_Apply(this, u_in, u, z)
            z_out = -this.M0' * u_in;
        end

        function [u_out] = c_u_Inverse_Apply(this, u_in, u, z)
            u_out = linsolve(this.L, u_in);
        end

        function [u_out] = c_z_Apply(this, z_in, u, z)
            u_out = -this.M0 * z_in;
        end

        % Hess vec applies
        function [u_out] = c_uu_Apply(this, u_in, u, z, lambda)
            u_out = zeros(this.dim, 1);
        end

        function [u_out] = c_uz_Apply(this, z_in, u, z, lambda)
            u_out = zeros(this.dim, 1);
        end

        function [z_out] = c_zu_Apply(this, u_in, u, z, lambda)
            z_out = zeros(this.dim, 1);
        end

        function [z_out] = c_zz_Apply(this, z_in, u, z, lambda)
            z_out = zeros(this.dim, 1);
        end

    end

    methods

        function this = Linearized_Poisson_Constraint(nonlinear_cons, m0, u0)
            this.dim = nonlinear_cons.dim;
            this.M = nonlinear_cons.M;
            this.M0 = nonlinear_cons.M0;
            this.S0 = nonlinear_cons.S0;
            this.L = this.S0 + 2 * this.M * diag(u0);
            this.m0 = m0;
            this.u0 = u0;
        end

    end

end
