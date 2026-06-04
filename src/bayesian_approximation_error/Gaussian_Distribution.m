classdef Gaussian_Distribution < Sampler_Interface & Prior_Model

    properties
        mu
        sigma
        dim
        R
    end

    methods (Access = public)

        function this = Gaussian_Distribution(mu, sigma)
            this@Sampler_Interface();
            this.dim = size(mu, 1);
            this.mu = mu;
            this.sigma = sigma;
            this.R = chol(sigma);
        end

        function [x_out] = Sample(this)
            x_out = this.mu + this.Prior_Covariance_Factor_Apply(randn(this.dim, 1));
        end

        function dim = Dimension(this)
            dim = this.dim;
        end

        function [x_out] = Get_Prior_Mean(this)
            x_out = this.mu;
        end

        function [x_out] = Prior_Covariance_Apply(this, x_in)
            x_out = this.sigma * x_in;
        end

        function [x_out] = Prior_Covariance_Factor_Apply(this, x_in)
            x_out = this.R * x_in;
        end

        function [x_out] = Prior_Precision_Apply(this, x_in)
          x_out = linsolve(this.sigma, x_in);
        end

    end

end
