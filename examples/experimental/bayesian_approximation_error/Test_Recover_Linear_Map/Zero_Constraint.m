classdef Zero_Constraint < Constraint
    % LINEARIZED_CONSTRAINT undefined
    %   undefined

    properties
        param_dim
        state_dim
        M
    end

    methods

        function this = Zero_Constraint(param_dim, state_dim)
            this@Constraint();
            this.param_dim = param_dim;
            this.state_dim = state_dim;
            this.M = eye(param_dim);
        end

        function [u] = State_Solve(this, z)
            u = zeros(this.state_dim, 1);
        end

        function [u_out] = c_u_Transpose_Inverse_Apply(this, u_in, u, z)
            u_out = zeros(this.state_dim, size(u_in, 2));
        end

        function [z_out] = c_z_Transpose_Apply(this, u_in, u, z)
            z_out = zeros(this.param_dim, size(u_in, 2));
        end

        function [u_out] = c_u_Inverse_Apply(this, u_in, u, z)
            u_out = zeros(this.state_dim, size(u_in, 2));
        end

        function [u_out] = c_z_Apply(this, z_in, u, z)
            u_out = zeros(this.state_dim, size(z_in, 2));
        end

        function [u_out] = c_uu_Apply(this, u_in, u, z, lambda)
            u_out = zeros(this.state_dim, size(u_in, 2));
        end

        function [u_out] = c_uz_Apply(this, z_in, u, z, lambda)
            u_out = zeros(this.state_dim, size(z_in, 2));
        end

        function [z_out] = c_zu_Apply(this, u_in, u, z, lambda)
            z_out = zeros(this.param_dim, size(u_in, 2));
        end

        function [z_out] = c_zz_Apply(this, z_in, u, z, lambda)
            z_out = zeros(this.param_dim, size(z_in, 2));
        end

    end
end
