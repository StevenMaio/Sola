%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%      Sola - Sandbox for Outer Loop Analysis         %%%%%%%%%
%%%%%%%%% Questions? Contact Joseph Hart (joshart@sandia.gov) %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef Poisson_Prior_Model < Inf_Dim_Prior_Model

    properties
        poisson_con
        L
        eigenvecs
        eigenvals
    end

    methods (Access = public)

        function [z_out] = Laplacian_Like_Apply(this, z_in)
            z_out = this.L * z_in;
        end

        function [z_out] = Laplacian_Like_Transpose_Apply(this, z_in)
            z_out = this.L' * z_in;
        end

        function [z_out] = Laplacian_Like_Inverse_Apply(this, z_in)
            z_out = linsolve(this.L, z_in);
        end

        function [z_out] = Laplacian_Like_Transpose_Inverse_Apply(this, z_in)
            z_out = linsolve(this.L', z_in);
        end

        function [z_out] = Prior_Covariance_Factor_Apply(this, z_in)
            tmp = this.Matrix_Sqrt_Apply(z_in);
            z_out = this.Laplacian_Like_Inverse_Apply(tmp);
        end

        function [z_out] = Mass_Matrix_Apply(this, z_in)
            z_out = this.poisson_con.M * z_in;
        end

        function [z_out] = Mass_Matrix_Inverse_Apply(this, z_in)
            z_out = linsolve(this.poisson_con.M, z_in);
        end

        function [z_prior_mean] = Get_Prior_Mean(this)
            z_prior_mean = zeros(this.poisson_con.dim, 1);
        end

    end

    methods (Access = public)

        function this = Poisson_Prior_Model(poisson_con, gamma, delta)
            this.dim = poisson_con.dim;
            this.poisson_con = poisson_con;
            this.L = gamma * poisson_con.S + delta * poisson_con.M;
            [V, D] = eig(poisson_con.M);
            this.eigenvecs = V;
            this.eigenvals = diag(D);
        end

    end

    methods (Access = private)

        function [z_out] = Matrix_Sqrt_Apply(this, z_in)
            temp = this.eigenvecs' * z_in;
            temp = temp .* sqrt(this.eigenvals);
            z_out = this.eigenvecs * temp;
        end

    end

end
