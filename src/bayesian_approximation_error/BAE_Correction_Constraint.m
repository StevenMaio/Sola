classdef BAE_Correction_Constraint < Constraint
    %BAE_CORRECTION_CONSTRAINT undefined
    %   undefined

    properties
      cons
      G_mm
      G_me
      state_dim
      param_dim
      data_dim
    end

    methods (Access = public)

        function this = BAE_Correction_Constraint(cons, bae_likelihood)
            this.cons = cons;
            this.G_mm = bae_likelihood.G_mm;
            this.G_me = bae_likelihood.G_me;
            this.state_dim = bae_likelihood.state_dim;
            this.param_dim = bae_likelihood.param_dim;
            this.data_dim = bae_likelihood.data_dim;
        end

        function [u] = State_Solve(this, m)
            u1 = this.cons.State_Solve(m);
            u2 = this.G_me' * linsolve(this.G_mm, m);
            u = [u1; u2];
        end

        function [u_out] = c_u_Transpose_Inverse_Apply(this, u_in, u, z)
            u1 = u(1:this.state_dim);
            u1_in = u_in(1:this.state_dim, :);
            u2_in = u_in(this.state_dim+1:end, :);
            u1_out = this.cons.c_u_Transpose_Inverse_Apply(u1_in, u1, z);
            u_out = [u1_out; u2_in];
        end

        function [z_out] = c_z_Transpose_Apply(this, u_in, u, m)
            u1 = u(1:this.state_dim);
            u1_in = u_in(1:this.state_dim, :);
            u2_in = u_in(this.state_dim+1:end, :);
            z_out = this.cons.c_z_Transpose_Apply(u1_in, u1, m);
            temp = zeros(this.param_dim, size(u_in, 2));
            for i = 1:size(u_in, 2)
                contrib = this.G_me * u2_in(:, i);
                contrib = linsolve(this.G_mm, contrib);
                temp(:, i) = temp(:, i) + contrib;
            end
            z_out = z_out - temp;
        end

        function [u_out] = c_u_Inverse_Apply(this, u_in, u, z)
            u1 = u(1:this.state_dim);
            u1_in = u_in(1:this.state_dim, :);
            u2_in = u_in(this.state_dim+1:end, :);
            u1_out = this.cons.c_u_Inverse_Apply(u1_in, u1, z);
            u_out = [u1_out; u2_in];
        end

        function [u_out] = c_z_Apply(this, m_in, u, m)
            u1 = u(1:this.state_dim);
            u1_out = this.cons.c_z_Apply(m_in, u1, m);
            u2_out = zeros(this.data_dim, size(m_in, 2));
            for i = 1:size(m_in, 2)
                u2_out(:, i) = -this.G_me' * linsolve(this.G_mm, m_in(:, i));
            end
            u_out = [u1_out; u2_out];
        end

        % Hess vec applies
        function [u_out] = c_uu_Apply(this, u_in, u, m, lambda)
            u1 = u(1:this.state_dim);
            u1_in = u_in(1:this.state_dim, :);
            lambda1 = lambda(1:this.state_dim, :);
            u1_out = this.cons.c_uu_Apply(u1_in, u1, m, lambda1);
            u2_out = zeros(this.data_dim, size(u_in, 2));
            u_out = [u1_out; u2_out];
        end

        function [u_out] = c_uz_Apply(this, z_in, u, m, lambda)
            u1 = u(1:this.state_dim);
            lambda1 = lambda(1:this.state_dim, :);
            u1_out = this.cons.c_uz_Apply(z_in, u1, m, lambda1);
            u2_out = zeros(this.data_dim, size(z_in, 2));
            u_out = [u1_out; u2_out];
        end

        function [z_out] = c_zu_Apply(this, u_in, u, m, lambda)
            u1 = u(1:this.state_dim);
            u1_in = u_in(1:this.state_dim, :);
            lambda1 = lambda(1:this.state_dim, :);
            z_out = this.cons.c_zu_Apply(u1_in, u1, m, lambda1);
        end

        function [z_out] = c_zz_Apply(this, z_in, u, m, lambda)
            u1 = u(1:this.state_dim);
            lambda1 = lambda(1:this.state_dim, :);
            z_out = this.cons.c_zz_Apply(z_in, u1, m, lambda1);
        end

    end
end
