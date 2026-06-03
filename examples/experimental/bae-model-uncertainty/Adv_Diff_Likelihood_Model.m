%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%      Sola - Sandbox for Outer Loop Analysis         %%%%%%%%%
%%%%%%%%% Questions? Contact Joseph Hart (joshart@sandia.gov) %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef Adv_Diff_Likelihood_Model < Likelihood_Model

    properties
        sigma
        obs_vec
        state_dim
    end

    methods (Access = public)

        function [d_out] = Noise_Precision_Apply(this, d_in)
            d_out = diag(1 / this.sigma^2) * d_in;
        end

        function [d_out] = Noise_Covariance_Apply(this, d_in)
            d_out = diag(this.sigma^2) * d_in;
        end

        function [d_out] = Observation_Operator_Apply(this, u_in)
            d_out = u_in(this.obs_vec, :);
        end

        function [u_out] = Observation_Operator_Transpose_Apply(this, d_in)
            u_out = zeros(this.state_dim, size(d_in, 2));
            u_out(this.obs_vec, :) = d_in;
        end

        function [d] = Get_Observed_Data(this)
            d = false;
        end

    end

    methods (Access = public)

        function this = Adv_Diff_Likelihood_Model(state_dim, obs_vec, sigma)
            this.sigma = sigma;
            this.obs_vec = obs_vec;
            this.state_dim = state_dim;
        end

    end

end
