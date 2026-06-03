%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%      Sola - Sandbox for Outer Loop Analysis         %%%%%%%%%
%%%%%%%%% Questions? Contact Joseph Hart (joshart@sandia.gov) %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef Adv_Diff_Constraint < Constraint

    properties
        param_dim
        state_dim
        diff_coeff
        vel_coeff
        x
        M
        M0
        S
        A
        V
    end

    methods (Access = public)

        function [u] = State_Solve(this, z)
            z = z * ones(this.state_dim, 1);
            b = this.M0 * z;
            u = linsolve(this.A, b);
        end

        function [Mv] = c_u_Transpose_Inverse_Apply(this, v, u, z)
            Mv = linsolve(this.A', v);
        end

        function [Mv] = c_z_Apply(this, v, u, z)
            v = v * ones(this.state_dim, 1);
            Mv = this.M0 * v;
        end

        function [Mv] = c_z_Transpose_Apply(this, v, u, z)
            v = v * ones(this.state_dim, 1);
            Mv = this.M0' * v;
        end

        function [Mv] = c_u_Inverse_Apply(this, v, u, z)
            Mv = linsolve(this.A, v);
        end

        function [Mv] = c_uu_Apply(this, v, u, z, lambda)
            % TODO: don't use this. It's probably wrong
            Mv = zeros(this.state_dim, 1);
        end

        function [Mv] = c_uz_Apply(this, v, u, z, lambda)
            % TODO: don't use this. It's probably wrong
            Mv = zeros(this.state_dim, 1);
        end

        function [Mv] = c_zu_Apply(this, v, u, z, lambda)
            % TODO: don't use this. It's probably wrong
            Mv = zeros(this.state_dim, 1);
        end

        function [Mv] = c_zz_Apply(this, v, u, z, lambda)
            % TODO: don't use this. It's probably wrong
            Mv = zeros(this.state_dim, 1);
        end

    end

    methods (Access = public)

        function this = Adv_Diff_Constraint(param_dim, state_dim, diff_coeff, vel_coeff)
            this = this@Constraint();
            this.param_dim = param_dim;
            this.state_dim = state_dim;
            this.diff_coeff = diff_coeff;
            this.vel_coeff = vel_coeff;
            this.x = linspace(0, 1, state_dim)';

            h = this.x(2) - this.x(1);

            M = diag(4 * ones(1, state_dim)) + diag(ones(1, state_dim - 1), 1) + diag(ones(1, state_dim - 1), -1);
            M(1, 1) = .5 * M(1, 1);
            M(end, end) = .5 * M(end, end);
            M = (1 / 6) * h * M;
            this.M = M;

            M0 = M;
            M0(1, :) = 0 * M0(1, :);
            M0(end, :) = 0 * M0(end, :);
            this.M0 = M0;

            S = diag(2 * ones(1, state_dim)) + (-1) * diag(ones(1, state_dim - 1), 1) + (-1) * diag(ones(1, state_dim - 1), -1);
            S(1, 1) = .5 * S(1, 1);
            S(end, end) = .5 * S(end, end);
            S = (1 / h) * S;
            this.S = S;

            V = diag(0 * ones(1, state_dim)) + (1 / 2) * diag(ones(1, state_dim - 1), 1) + (-1 / 2) * diag(ones(1, state_dim - 1), -1);
            V(1, 1) = -1 / 2;
            V(end, end) = 1 / 2;
            this.V = V;

            A = this.diff_coeff * this.S + this.vel_coeff * this.V;
            A(1, :) = 0 * A(1, :);
            A(end, :) = 0 * A(end, :);
            A(1, 1) = 1;
            A(end, end) = 1;
            this.A = A;
        end

    end
end

