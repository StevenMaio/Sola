%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%      Sola - Sandbox for Outer Loop Analysis         %%%%%%%%%
%%%%%%%%% Questions? Contact Joseph Hart (joshart@sandia.gov) %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef Inf_Dim_Prior_Model < Prior_Model & Sampler_Interface

    % We assume a Bayesian inverse problem with a mean zero Gaussian noise
    % model and a linear observation operator

    properties
        mass_mat_sqrt
        dim
    end

    methods (Abstract, Access = public)

        [z_out] = Laplacian_Like_Apply(this, z_in)

        [z_out] = Laplacian_Like_Transpose_Apply(this, z_in)

        [z_out] = Laplacian_Like_Inverse_Apply(this, z_in)

        [z_out] = Laplacian_Like_Transpose_Inverse_Apply(this, z_in)

        [z_out] = Mass_Matrix_Apply(this, z_in)

        [z_out] = Mass_Matrix_Inverse_Apply(this, z_in)

        [z_prior_mean] = Get_Prior_Mean(this)

    end

    methods (Access = public)

        function this = Inf_Dim_Prior_Model(dim)
            arguments
                dim = -1    % optional to preserve compatibility.
                            % Only required if you need to use the
                            % Sampler_Interface functionality
            end
            this.mass_mat_sqrt = Mass_Matrix_Sqrt(this);
            this.dim = dim;
        end

        function [z_out] = Prior_Precision_Apply(this, z_in)
            tmp1 = this.Laplacian_Like_Apply(z_in);
            tmp2 = this.Mass_Matrix_Inverse_Apply(tmp1);
            z_out = this.Laplacian_Like_Transpose_Apply(tmp2);
        end

        function [z_out] = Prior_Covariance_Apply(this, z_in)
            tmp1 = this.Laplacian_Like_Transpose_Inverse_Apply(z_in);
            tmp2 = this.Mass_Matrix_Apply(tmp1);
            z_out = this.Laplacian_Like_Inverse_Apply(tmp2);
        end

        function [z_out] = Prior_Covariance_Factor_Apply(this, z_in)
            tmp = this.mass_mat_sqrt.Matrix_Sqrt_Apply(z_in);
            z_out = this.Laplacian_Like_Inverse_Apply(tmp);
        end

        function [z_out] = Sample(this)
            z = randn(this.dim, 1);
            z_out = this.Get_Prior_Mean() + this.Prior_Covariance_Factor_Apply(z);
        end

        function dim = Dimension(this)
            dim = this.dim;
        end

    end

end
