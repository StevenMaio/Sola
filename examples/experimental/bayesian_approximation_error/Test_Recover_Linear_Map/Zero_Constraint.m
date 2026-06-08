classdef Zero_Constraint < Constraint
    % LINEARIZED_CONSTRAINT undefined
    %   undefined

    properties
        state_dim
    end

    methods

        function this = Zero_Constraint(state_dim)
            this@Constraint();
            this.state_dim = state_dim;
        end

        function [u] = State_Solve(this, z)
            u = zeros(this.state_dim, 1);
        end

        function [u_out] = c_u_Transpose_Inverse_Apply(this, u_in, u, z)
            u_out = false;
        end

        function [z_out] = c_z_Transpose_Apply(this, u_in, u, z)
            z_out = false;
        end

        function [u_out] = c_u_Inverse_Apply(this, u_in, u, z)
            u_out = false;
        end

        function [u_out] = c_z_Apply(this, z_in, u, z)
            u_out = false;
        end

    end
end
