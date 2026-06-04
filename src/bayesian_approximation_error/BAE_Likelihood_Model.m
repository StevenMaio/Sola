%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%      Sola - Sandbox for Outer Loop Analysis         %%%%%%%%%
%%%%%%%%% Questions? Contact Joseph Hart (joshart@sandia.gov) %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef BAE_Likelihood_Model < Likelihood_Model

    properties
        noise_likelihood % original likelihood model
        num_samples
        error_mean
        error_cov
        % only needed for model discrepancy
        param_mean 
        param_cov
        cross_cov   % param/error cross covariance
        % distributions
        param_distr
        aux_distr
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

        function this = BAE_Likelihood_Model(full_F, approximate_F, ...
                noise_likelihood, param_distr, aux_distr, num_samples, ...
                only_aux_params)
            arguments
                full_F
                approximate_F
                noise_likelihood
                param_distr
                aux_distr
                num_samples
                only_aux_params = true
            end
            % TODO: need to figure out how to simplify stuff and allow for a choice
            this.noise_likelihood = noise_likelihood;
            this.param_distr = param_distr;
            this.aux_distr = aux_distr;
            this.num_samples = num_samples;

            param_samples = zeros(num_samples, param_distr.dim);
            aux_samples = zeros(num_samples, aux_distr.dim);
            err_samples = [];

            for i = 1:num_samples
                m = param_distr.Sample();
                xi = aux_distr.Sample();
                param_samples(i, :) = m;
                aux_samples(i, :) = xi;
                err = full_F(m, xi) - approximate_F(m);
                err_samples(i, :) = err;
            end

            this.error_mean = mean(err_samples)';
            this.error_cov = cov(err_samples);

            this.only_aux_params = only_aux_params;

            if ~only_aux_params
                this.param_mean = mean(param_samples)';
                this.param_cov = cov(param_samples);
    
                this.cross_cov = zeros(size(param_samples, 2), size(err_samples, 2));
    
                for i = 1:num_samples
                    param_contrib = param_samples(i, :)' - this.param_mean;
                    error_contrib = err_samples(i, :)' - this.error_mean;
                    this.cross_cov = this.cross_cov + 1 / (num_samples - 1) * param_contrib * error_contrib';
                end
            end
        end

    end

end
