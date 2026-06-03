%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%      Sola - Sandbox for Outer Loop Analysis         %%%%%%%%%
%%%%%%%%% Questions? Contact Joseph Hart (joshart@sandia.gov) %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef BAE_Likelihood_Model < Likelihood_Model

    properties
        noise_likelihood  % original likelihood model
        error_mean
        error_cov
    end

    methods (Access = public)

        function [d_out] = Noise_Precision_Apply(this, d_in)
            d_out = cgs(@(d) this.Noise_Covariance_Apply(d), d_in);
        end

        function [d_out] = Noise_Covariance_Apply(this, d_in)
            d_out = this.error_cov * d_in + this.noise_likelihood.Noise_Covariance_Apply(d_in);
        end

        function [d_out] = Observation_Operator_Apply(this, u_in)
            d_out = this.noise_likelihood.Observation_Operator_Apply(u_in);
        end

        function [u_out] = Observation_Operator_Transpose_Apply(this, d_in)
            u_out = this.noise_likelihood.Observation_Operator_Transpose_Apply(d_in);
        end

        function [d] = Get_Observed_Data(this)
            d = false;
        end

        function [d] = Get_Error_Mean(this)
            d = this.error_mean + this.noise_likelihood.Get_Error_Mean();
        end

    end

    methods (Access = public)

        function this = BAE_Likelihood_Model(noise_likelihood, error_mean, error_cov)
          this.noise_likelihood = noise_likelihood;
          this.error_mean = error_mean;
          this.error_cov = error_cov;
        end

    end

end
