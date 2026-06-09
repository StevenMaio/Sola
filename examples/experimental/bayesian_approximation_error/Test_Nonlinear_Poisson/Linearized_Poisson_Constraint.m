classdef Linearized_Poisson_Constraint < Constraint

    properties
        dim     % dimension of state and parameter
        z0      % point of linearization
        u0      % state at point of linearization
        A       % bilinear form at linearization point
        f       % source term
        M       % Mass matrix
        M0      % lifted mass matrix
    end

    methods (Access = public)
        % abstract methods that need to be implemented

        function [u] = State_Solve(this, z)
            rhs = this.M0 * (this.f .* (z - this.z0));
            du = -linsolve(this.A, rhs);
            u = this.u0 + du;
        end

        function [u_out] = c_u_Transpose_Inverse_Apply(this, u_in, u, z)
            u_out = this.A * u_in;
        end

        function [z_out] = c_z_Transpose_Apply(this, u_in, u, z)
            temp = this.M0 * u_in;
            z_out = this.f .* temp;
        end

        function [u_out] = c_u_Inverse_Apply(this, u_in, u, z)
            u_out = linsolve(this.A, u_in);
        end

        function [u_out] = c_z_Apply(this, z_in, u, z)
            temp = this.f .* z;
            u_out = this.M0 * temp;
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

        function this = Linearized_Poisson_Constraint(dim, z0, u0, A, f, M)
            this@Constraint();
            this.dim = dim;
            this.z0 = z0;
            this.u0 = u0;
            this.A = A;
            this.f = f;
            this.M = M;
            M0 = M;
            M0(:, 1) = 0 * M0(:, 1);
            M0(:, end) = 0 * M0(:, end);
            % Maybe incorporate these?
            % M0(1, 1) = 1;
            % M0(end, end);
            this.M0 = M0;
        end

    end

end
