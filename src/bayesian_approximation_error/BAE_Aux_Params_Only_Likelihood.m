%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%      Sola - Sandbox for Outer Loop Analysis         %%%%%%%%%
%%%%%%%%% Questions? Contact Joseph Hart (joshart@sandia.gov) %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Author(s):
%       - Steven Maio (smaio@sandia.gov or smaio@ncsu.edu)

classdef BAE_Aux_Params_Only_Likelihood < Likelihood_Model
    %% BAE_Params_Only_Likelihood
    %   Bayesian approximation error (BAE) error that only deals with
    %   premarginalizing the approximation error due the auxiliary
    %   parameters [1]. This approach can also premarginalize over the
    %   paramter of interest [2].
    %
    %   This class is the error corrected likelihood model, and should replace
    %   the original likelihood in a Bayesian inversion problem.
    %
    %   NOTE: We assume that the initial noise error has mean zero.
    %
    %   Sources:
    %       [1]: Alen Alexanderian et al 2024 Inverse Problems 40 095001
    %       [2]: Ruanui Nicholson et al 2023 Inverse Problems 39 054001
    %
    %   Author(s):
    %       - Steven Maio (smaio@sandia.gov or smaio@ncsu.edu)

    properties
        noise_likelihood % original likelihood model
        num_samples
        e0
        G_ee
        % distributions
        param_distr
        aux_distr
        d
    end

    methods (Access = public)

        function [d_out] = Noise_Precision_Apply(this, d_in)
            [d_out, ~] = cgs(@(d) this.Noise_Covariance_Apply(d), d_in);
        end

        function [d_out] = Noise_Covariance_Apply(this, d_in)
            d_out = this.G_ee * d_in + this.noise_likelihood.Noise_Covariance_Apply(d_in);
        end

        function [d_out] = Observation_Operator_Apply(this, u_in)
            d_out = this.noise_likelihood.Observation_Operator_Apply(u_in);
        end

        function [u_out] = Observation_Operator_Transpose_Apply(this, d_in)
            u_out = this.noise_likelihood.Observation_Operator_Transpose_Apply(d_in);
        end

        function [d] = Get_Observed_Data(this)
            d = this.d - this.Get_Error_Mean();
        end

        function [d] = Get_Error_Mean(this)
            d = this.e0;
        end

    end

    methods (Access = public)

        function this = BAE_Aux_Params_Only_Likelihood(full_cons, approximate_cons, ...
                                                       noise_likelihood, param_distr, aux_distr, num_samples, d)
            arguments
                full_cons Parametric_Constraint
                approximate_cons Constraint
                noise_likelihood Likelihood_Model
                param_distr Sampler_Interface
                aux_distr Sampler_Interface
                num_samples
                d                   % data vector
            end
            this.noise_likelihood = noise_likelihood;
            this.param_distr = param_distr;
            this.aux_distr = aux_distr;
            this.num_samples = num_samples;
            this.d = d;

            param_samples = zeros(num_samples, param_distr.dim);
            aux_samples = zeros(num_samples, aux_distr.dim);
            err_samples = [];

            full_F = @(m, xi) noise_likelihood.Observation_Operator_Apply(full_cons.Parametric_State_Solve(m, xi));
            approximate_F = @(m) noise_likelihood.Observation_Operator_Apply(approximate_cons.State_Solve(m));

            for i = 1:num_samples
                m = param_distr.Sample();
                xi = aux_distr.Sample();
                param_samples(i, :) = m;
                aux_samples(i, :) = xi;
                err = full_F(m, xi) - approximate_F(m);
                err_samples(i, :) = err;
            end

            this.e0 = mean(err_samples)';
            this.G_ee = cov(err_samples);
        end

    end

end
