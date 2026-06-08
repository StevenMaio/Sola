classdef Adv_Diff_Prior_Model < Prior_Model

    properties
        mu
        sigma
    end

    methods (Access = public)

        function [m_out] = Prior_Precision_Apply(this, m_in)
            m_out = m_in / this.sigma^2;
        end

        function [m_out] = Prior_Covariance_Apply(this, m_in)
            m_out = m_in * this.sigma^2;
        end

        function [m_out] = Get_Prior_Mean(this)
            m_out = this.mu;
        end

        function [m_out] = Prior_Covariance_Factor_Apply(this, m_in)
            m_out = this.sigma * m_in;
        end

    end

    methods (Access = public)

        function this = Adv_Diff_Prior_Model(mu, sigma)
            this.mu = mu;
            this.sigma = sigma;
        end

    end
end
