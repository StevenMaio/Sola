%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%      Sola - Sandbox for Outer Loop Analysis         %%%%%%%%%
%%%%%%%%% Questions? Contact Joseph Hart (joshart@sandia.gov) %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef Nonlinear_Poisson_Constraint < Constraint

    properties
        dim
        x
        f
        M
        S
        K_adj
    end

    methods (Access = public)

        function [u] = State_Solve(this, z)
            K = this.Construct_Stiffness_Matrix(z);
            rhs = this.M * this.f;
            rhs(1) = 0;
            rhs(end) = 0;

            u = linsolve(K, rhs);
        end

        function [u_out] = c_u_Transpose_Inverse_Apply(this, u_in, u, z)
            K = this.Construct_Stiffness_Matrix(z);
            u_out = linsolve(K', u_in);
        end

        function [z_out] = c_z_Transpose_Apply(this, u_in, u, z)
            z_out = zeros(this.dim, size(u_in, 2));
            for j = 1:size(u_in, 2)
                uv = u_in(:, j) * u';
                % for i = 1:this.dim
                %     z_out(i, j) = sum(this.K_adj(i, :, :) .* uv, 'all');
                % end
                z_out(:, j) = this.K_adj' * reshape(uv, [this.dim*this.dim, 1]);
            end
        end

        function [u_out] = c_u_Inverse_Apply(this, u_in, u, z)
            K = this.Construct_Stiffness_Matrix(z);
            u_out = linsolve(K, u_in);
        end

        function [u_out] = c_z_Apply(this, z_in, u, z)
            K = this.Construct_Stiffness_Matrix(z_in);
            u_out = K * u;
        end

        function [Mv] = c_uu_Apply(this, v, u, z, lambda)
            Mv = zeros(this.dim, size(v, 2));
        end

        function [Mv] = c_uz_Apply(this, v, u, z, lambda)
            Mv = zeros(this.dim, size(v, 2));
        end

        function [Mv] = c_zu_Apply(this, v, u, z, lambda)
            Mv = zeros(this.dim, size(v, 2));
        end

        function [Mv] = c_zz_Apply(this, v, u, z, lambda)
            Mv = zeros(this.dim, size(v, 2));
        end

    end

    methods (Access = public)

        function this = Nonlinear_Poisson_Constraint(dim, f)
            this = this@Constraint();
            this.dim = dim;
            this.x = linspace(0, 1, dim)';
            this.f = f;

            h = this.x(2) - this.x(1);

            M = diag(4 * ones(1, dim)) + diag(ones(1, dim - 1), 1) + diag(ones(1, dim - 1), -1);
            M(1, 1) = .5 * M(1, 1);
            M(end, end) = .5 * M(end, end);
            M = (1 / 6) * h * M;
            this.M = M;

            % central difference difference approximation -- 
            S = diag(2 * ones(1, dim)) + (-1) * diag(ones(1, dim - 1), 1) + (-1) * diag(ones(1, dim - 1), -1);
            S(1, 1) = .5 * S(1, 1);
            S(end, end) = .5 * S(end, end);
            S = (1 / h) * S;
            this.S = S;

            K_adj = zeros(dim*dim, dim);
            for i = 1:dim
                m = zeros(dim, 1);
                m(i) = 1.0;
                K = this.Construct_Stiffness_Matrix(m, false);
                K_adj(:, i) = reshape(K, [dim*dim, 1]);
            end
            this.K_adj = sparse(K_adj);
        end

        function [K] = Construct_Stiffness_Matrix(this, z, apply_lifting)
            arguments
                this
                z
                apply_lifting = true
            end
            % Returns matrix with lifting applied
            h = this.x(2) - this.x(1);

            K = zeros(this.dim, this.dim);
            K(1, 1) = (z(1) + z(2)) / (2 * h);
            K(end, end) = (z(end-1) + z(end)) / (2 * h);
            for i = 2:this.dim-1
                K(i, i) = (z(i-1) + 2*z(i) + z(i+1))/ (2 * h);
                if i < this.dim
                    K(i, i+1) = -(z(i) + z(i+1)) / (2 * h);
                end
                if i > 1
                    K(i, i-1) = -(z(i) + z(i-1)) / (2 * h);
                end
            end
            % apply lifting
            if apply_lifting
                K(1, :) = 0 * K(1, :);
                K(:, 1) = 0 * K(:, 1);
                K(end, :) = 0 * K(end, :);
                K(:, end) = 0 * K(:, end);
                K(1, 1) = 1;
                K(end, end) = 1;
            end
        end

    end
end
