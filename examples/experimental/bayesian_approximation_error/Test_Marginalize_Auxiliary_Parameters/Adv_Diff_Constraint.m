%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%      Sola - Sandbox for Outer Loop Analysis         %%%%%%%%%
%%%%%%%%% Questions? Contact Joseph Hart (joshart@sandia.gov) %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef Adv_Diff_Constraint < Parametric_Constraint

    properties
        param_dim
        state_dim
        diff_coeff
        vel_coeff
        x
        M
        M0
        S
        V
    end

    methods (Access = public)

        function [u] = Parametric_State_Solve(this, z, theta)
            A = this.Construct_Matrix(theta);
            z = z * ones(this.state_dim, 1);
            b = this.M0 * z;
            u = linsolve(A, b);
        end

        function [Mv] = Parametric_c_u_Transpose_Inverse_Apply(this, v, u, z, theta)
            A = -this.Construct_Matrix(theta);
            Mv = linsolve(A', v);
        end

        function [Mv] = Parametric_c_z_Apply(this, v, u, z, theta)
            v = v * ones(this.state_dim, 1);
            Mv = this.M0 * v;
        end

        function [Mv] = Parametric_c_z_Transpose_Apply(this, u_in, u, z, theta)
            v = this.M0' * ones(this.state_dim, 1);
            Mv = v' * u_in;
        end

        function [Mv] = Parametric_c_u_Inverse_Apply(this, v, u, z, theta)
            A = this.Construct_Matrix(theta);
            Mv = -linsolve(A, v);
        end

    end

    methods (Access = public)

        function this = Adv_Diff_Constraint(param_dim, state_dim, diff_coeff, vel_coeff)
            this@Parametric_Constraint(vel_coeff);
            this.param_dim = param_dim;
            this.state_dim = state_dim;
            this.diff_coeff = diff_coeff;
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
        end

    end

    methods (Access = private)

        function [A] = Construct_Matrix(this, theta)
              A = this.diff_coeff * this.S + theta * this.V;
              A(1, :) = 0 * A(1, :);
              A(end, :) = 0 * A(end, :);
              A(1, 1) = 1;
              A(end, end) = 1;
        end

    end
end

