%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%      Sola - Sandbox for Outer Loop Analysis         %%%%%%%%%
%%%%%%%%% Questions? Contact Joseph Hart (joshart@sandia.gov) %%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   Author(s):
%       - Steven Maio (smaio@sandia.gov or smaio@ncsu.edu)

classdef BAE_Likelihood_Model < Likelihood_Model
    %% BAE_Likelihood_Model
    %   Bayesian approximation error (BAE) error that only 
    %   premarginalizes the approximation error due to the parameter of
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
        % information
        param_dim
        state_dim
        data_dim
    end

    methods (Access = public)

        function [d_out] = Noise_Precision_Apply(this, d_in)
            d_out = zeros(size(d_in));
            for i = 1:size(d_in, 2)
                [temp, ~] = cgs(@(d) this.Noise_Covariance_Apply(d), d_in(:, i));
                d_out(:, i) = temp;
            end
        end

        function [d_out] = Noise_Covariance_Apply(this, d_in)
            d_out = this.G_ee * d_in + this.noise_likelihood.Noise_Covariance_Apply(d_in);
            temp = this.G_me * d_in;
            temp = this.Apply_Linear_Correction(temp);
            d_out = d_out - temp;
        end

        function [d_out] = Observation_Operator_Apply(this, u_in)
            d_out = zeros(this.data_dim, size(u_in, 2));
            for i = 1:size(u_in, 2)
                u = u_in(1:this.state_dim, i);
                correction = u_in(this.state_dim+1:end, i);
                observation = this.noise_likelihood.Observation_Operator_Apply(u);
                d_out(:, i) = observation + correction;
            end
        end

        function [u_out] = Observation_Operator_Transpose_Apply(this, d_in)
            u_out = zeros(this.state_dim + this.data_dim, size(d_in, 2));
            for i = 1:size(d_in, 2)
              u1_out = this.noise_likelihood.Observation_Operator_Transpose_Apply(d_in(:, i));
              u_out(:, i) = [u1_out; d_in(:, i)];
            end
        end

        function [d] = Get_Observed_Data(this)
            d = this.d - this.Get_Error_Mean();
        end

        function [d] = Get_Error_Mean(this)
            d = this.e0 - this.b;
        end

    end

    methods (Access = public)

        function this = BAE_Likelihood_Model(full_cons, approximate_cons, ...
                noise_likelihood, prior, num_samples, d, param_dim, state_dim, data_dim)
            arguments
                full_cons Constraint
                approximate_cons Constraint
                noise_likelihood Likelihood_Model
                prior Prior_Model
                num_samples
                d               % data vector
                param_dim
                state_dim
                data_dim
            end
            this.noise_likelihood = noise_likelihood;
            this.param_distr = prior;
            this.num_samples = num_samples;
            this.d = d;
            this.param_dim = param_dim;
            this.state_dim = state_dim;
            this.data_dim = data_dim;

            param_samples = prior.Compute_Prior_Samples(num_samples)';
            err_samples = zeros(num_samples, data_dim);

            full_F = @(m) noise_likelihood.Observation_Operator_Apply(full_cons.State_Solve(m));
            approximate_F = @(m) noise_likelihood.Observation_Operator_Apply(approximate_cons.State_Solve(m));

            for i = 1:num_samples
                m = param_samples(i, :)';
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

        function [e_out] = Apply_Linear_Correction(this, m_in)
            % Applying G_em G_mm^{-1} to m_in
            temp = linsolve(this.G_mm, m_in);
            e_out = this.G_me' * temp;
        end

    end

end
