%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%      Sola - Sandbox for Outer Loop Analysis         %%%%%%%%%
%%%%%%%%% Questions? Contact Joseph Hart (joshart@sandia.gov) %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef Linearized_Poisson_Constraint < Constraint

    properties
        dim
        u0
        m0
        K0
        nonlinear_cons
    end

    methods (Access = public)

        function [u] = State_Solve(this, m)
            dm = m - this.m0;
            K = this.nonlinear_cons.Construct_Stiffness_Matrix(dm);
            rhs = -K*this.u0;
            du = linsolve(this.K0, rhs);
            u = this.u0 + du;
        end

        function [u_out] = c_u_Transpose_Inverse_Apply(this, u_in, u, z)
            u_out = linsolve(this.K0', u_in);
        end

        function [z_out] = c_z_Transpose_Apply(this, u_in, u, z)
            z_out = zeros(this.dim, size(u_in, 2));
            for j = 1:size(u_in, 2)
                uv = u_in(:, j) * this.u0';
                z_out(:, j) = this.nonlinear_cons.K_adj' * reshape(uv, [this.dim*this.dim, 1]);
            end
        end

        function [u_out] = c_u_Inverse_Apply(this, u_in, u, z)
            u_out = linsolve(this.K0, u_in);
        end

        function [u_out] = c_z_Apply(this, z_in, u, z)
            u_out = zeros(this.dim, size(z_in, 2));
            for i = 1:size(z_in, 2)
                K = this.nonlinear_cons.Construct_Stiffness_Matrix(z_in(:, i));
                u_out(:, i) = K * this.u0;
            end
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

        function this = Linearized_Poisson_Constraint(nonlinear_cons, m0, u0)
            this = this@Constraint();
            this.dim = nonlinear_cons.dim;
            this.nonlinear_cons = nonlinear_cons;
            this.m0 = m0;
            this.u0 = u0;
            this.K0 = nonlinear_cons.Construct_Stiffness_Matrix(m0);
        end

    end
end
