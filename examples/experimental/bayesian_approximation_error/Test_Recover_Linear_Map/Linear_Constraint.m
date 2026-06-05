classdef Linear_Constraint < Constraint
    %LINEARIZED_CONSTRAINT undefined
    %   undefined

    properties
        A   % discretization of domain
    end

    methods
        function this = Linear_Constraint(A)
            this@Constraint();
            this.A = A;
        end

        function [u] = State_Solve(this, z)
            u = this.A * z;
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

