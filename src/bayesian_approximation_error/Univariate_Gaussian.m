classdef Univariate_Gaussian < Param_Distribution

    properties
        mu
        sigma
    end

    methods (Access = public)

        function this = Univariate_Gaussian(mu, sigma)
            this@Param_Distribution(1);
            this.mu = mu;
            this.sigma = sigma;
        end

        function [x_out] = Sample(this, num_samples)
            arguments
                this
                num_samples = 1
            end
            x_out = this.mu + this.Covariance_Factor_Apply(randn(num_samples, 1));
        end

        function [x_out] = Get_Mean(this)
            x_out = this.mu;
        end

        function [x_out] = Covariance_Apply(this, x_in)
            x_out = this.sigma^2 * x_in;
        end

        function [x_out] = Covariance_Factor_Apply(this, x_in)
            x_out = this.sigma * x_in;
        end

        function [x_out] = Precision_Apply(this, x_in)
          x_out = this.sigma^(-2) * x_in;
        end

    end

end
