classdef BAE_Correction_Constraint < Constraint
    %BAE_CORRECTION_CONSTRAINT undefined
    %   undefined

    properties
      cons
      M         % mass matrix
      G_mm
      G_me
      state_dim
      param_dim
      data_dim
    end

    methods (Access = public)

        function this = BAE_Correction_Constraint(cons, bae_likelihood)
            this.cons = cons;
            this.M = cons.M;
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

        function [u_out] = c_u_Transpose_Inverse_Apply(this, u_in, u, m)
            u_out = zeros(size(u_in));
            u1 = u(1:this.state_dim);
            for i = 1:size(u_in, 2)
                u1_in = u_in(1:this.state_dim, i);
                u2_in = u_in(this.state_dim+1:end, i);
                u1_out = this.cons.c_u_Transpose_Inverse_Apply(u1_in, u1, m);
                u_out(:, i) = [u1_out; u2_in];
            end
        end

        function [z_out] = c_z_Transpose_Apply(this, u_in, u, m)
            z_out = zeros(this.param_dim, size(u_in, 2));
            u1 = u(1:this.state_dim);
            for i = 1:size(u_in, 2)
                u1_in = u_in(1:this.state_dim, i);
                u2_in = u_in(this.state_dim+1:end, i);
                z1_out = this.cons.c_z_Transpose_Apply(u1_in, u1, m);
                temp = this.G_me * u2_in;
                temp = linsolve(this.G_mm, temp);
                z2_out = linsolve(this.M, temp);
                z_out(:, i) = z1_out - z2_out;
            end
        end

        function [u_out] = c_u_Inverse_Apply(this, u_in, u, m)
            u_out = zeros(size(u_in));
            u1 = u(1:this.state_dim);
            for i = 1:size(u_in, 2)
                u1_in = u_in(1:this.state_dim, i);
                u2_in = u_in(this.state_dim+1:end, i);
                u1_out = this.cons.c_u_Inverse_Apply(u1_in, u1, m);
                u_out(:, i) = [u1_out; u2_in];
            end
        end

        function [u_out] = c_z_Apply(this, m_in, u, m)
            u_out = zeros(this.state_dim + this.data_dim, size(m_in, 2));
            u1 = u(1:this.state_dim);
            for i = 1:size(m_in, 2)
                u1_out = this.cons.c_z_Apply(m_in(:, i), u1, m);
                u2_out = -this.G_me' * linsolve(this.G_mm, m_in(:, i));
                u_out(:, i) = [u1_out; u2_out];
            end
        end

        % Hess vec applies
        function [u_out] = c_uu_Apply(this, u_in, u, m, lambda)
          % TODO: implement this
          u_out = false
        end

        function [u_out] = c_uz_Apply(this, m_in, u, m, lambda)
            % TODO: implement this
            u_out = false
        end

        function [z_out] = c_zu_Apply(this, u_in, u, m, lambda)
            % TODO: implement this
            u_out = false
        end

        function [z_out] = c_zz_Apply(this, m_in, u, m, lambda)
            % TODO: implement this
            u_out = false
        end

    end
end
