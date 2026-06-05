classdef Nonlinear_Poisson_Constraint < Constraint

    properties
        dim     % dimension of state and parameter
        f       % given source term
        x       % domain discretization
        M       % mass matrix
        M0      % mass matrix with BCs applied
        S       % H1 inner product matrix
    end

    methods (Access = public)
        % abstract methods that need to be implemented

        function [u] = State_Solve(this, z)
            A = this.Construct_Matrix(z);
            b = this.M0 * this.f;
            u = linsolve(A, b);
        end

        function [u_out] = c_u_Transpose_Inverse_Apply(this, u_in, u, z)
            % Ignore this
            u_out = - u_in;
        end

        function [z_out] = c_z_Transpose_Apply(this, u_in, u, z)
            % Ignore this
            z_out = - u_in .* this.u0;
        end

        function [u_out] = c_u_Inverse_Apply(this, u_in, u, z)
            % Ignore this
            u_out = - u_in;
        end

        function [u_out] = c_z_Apply(this, z_in, u, z)
            % Ignore this
            u_out = - this.u0 .* z_in;
        end
        
    end

    methods

        function this = Nonlinear_Poisson_Constraint(dim)
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

            this.f = ones(dim, 1);
        end

    end

    methods (Access = private)
          
        function [A] = Construct_Matrix(this, z)
            A = this.S;
            v = exp(z/2);
            A = diag(v) * A * diag(v);
            % Apply lifting to maintain BCs
            A(1, :) = 0 * A(1, :);
            A(end, :) = 0 * A(end, :);
            A(1, 1) = 1;
            A(end, end) = 1;
        end

    end
end
