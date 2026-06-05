%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%      Sola - Sandbox for Outer Loop Analysis         %%%%%%%%%
%%%%%%%%% Questions? Contact Joseph Hart (joshart@sandia.gov) %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Author(s):
%       - Steven Maio (smaio@sandia.gov or smaio@ncsu.edu)

classdef BAE_Params_Only_Likelihood < Likelihood_Model
    %% BAE_Params_Only_Likelihood
    %   Bayesian approximation error (BAE) error that only deals with
    %   premarginalizing the approximation error due to the parameter of
    %   interest [1]. [2] applies this approach to OED.
    %
    %   NOTE: We assume that the initial noise error has mean zero.
    %
    %   Sources:
    %       [1]: Ruanui Nicholson et al 2023 Inverse Problems 39 054001
    %       [2]: Koval, K., Nicholson, R. Non-intrusive optimal experimental
    %            design for large-scale nonlinear Bayesian inverse problems
    %            using a Bayesian approximation error approach. J Sci Comput
    %            104, 98 (2025). https://doi.org/10.1007/s10915-025-03008-7

    properties
        noise_likelihood % original likelihood model
        num_samples
        % Empirical statistics
        % -- e: approximation error, m: parameter
        G_ee
        e0
        G_mm
        m0
        G_me
        % Other
        param_distr
        M   % parameter weighted mass matrix
        b   % conditional correction term
        d   % observed data
    end

    methods (Access = public)

        function [d_out] = Noise_Precision_Apply(this, d_in)
            [d_out, ~] = cgs(@(d) this.Noise_Covariance_Apply(d), d_in);
        end

        function [d_out] = Noise_Covariance_Apply(this, d_in)
            d_out = this.G_ee * d_in + this.noise_likelihood.Noise_Covariance_Apply(d_in);
            temp = this.G_me * d_in;
            temp = this.Apply_Linear_Correction(temp);
            d_out = d_out - temp;
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
            d = this.e0 - this.b;
        end

    end

    methods (Access = public)

        function this = BAE_Params_Only_Likelihood(full_cons, approximate_cons, ...
                noise_likelihood, param_distr, num_samples, d, M)
            arguments
                full_cons Constraint
                approximate_cons Constraint
                noise_likelihood Likelihood_Model
                param_distr Sampler_Interface
                num_samples
                d               % data vector
                M = @(m) m      % mass matrix
            end
            this.noise_likelihood = noise_likelihood;
            this.param_distr = param_distr;
            this.num_samples = num_samples;
            this.d = d;
            this.M = M;

            param_samples = zeros(num_samples, param_distr.Dimension());
            err_samples = [];

            full_F = @(m) noise_likelihood.Observation_Operator_Apply(full_cons.State_Solve(m));
            approximate_F = @(m) noise_likelihood.Observation_Operator_Apply(approximate_cons.State_Solve(m));

            for i = 1:num_samples
                m = param_distr.Sample();
                param_samples(i, :) = m;
                err = full_F(m) - approximate_F(m);
                err_samples(i, :) = err;
            end

            this.e0 = mean(err_samples)';
            this.G_ee = cov(err_samples);
            this.m0 = mean(param_samples)';
            this.G_mm = cov(param_samples);
            this.G_me = zeros(size(this.m0, 1), size(this.e0, 1));

            for i = 1:num_samples
                param_contrib = param_samples(i, :)' - this.m0;
                err_contrib = err_samples(i, :)' - this.e0;
                this.G_me = this.G_me + param_contrib * err_contrib' / (num_samples - 1);
            end

            this.b = this.Apply_Linear_Correction(this.m0);
        end

        function [m_out] = Apply_Linear_Correction(this, m_in)
            % Applying G_em G_mm^{-1} to m_in
            temp = linsolve(this.G_mm, m_in);
            temp = this.G_me' * temp;
            [m_out, ~] = cgs(this.M, temp);
        end

    end

end
