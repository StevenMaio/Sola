classdef Param_Distribution < handle

    properties
      dim
    end

    methods (Access = public)

        function this = Param_Distribution(dim)
            this.dim = dim;
        end
    end

    %% Semi-abstract methods. Will depend on use case
    methods (Access = public)
    function [x_out] = Sample(this, num_samples)
        %% Required for Bayesian Approximation Error
        arguments
            this
            num_samples = 1
        end
        x_out = error("Sample() not implemented");
    end

    function [x_out] = Get_Mean(this)
        x_out = error("Get_Mean() not implemented");
    end

    function [x_out] = Covariance_Apply(this, x_in)
        x_out = error("Covariance() not implemented");
    end

    function [x_out] = Covariance_Factor_Apply(this, x_in)
        x_out = error("Covariance_Factor_Apply() not implemented");
    end

    function [x_out] = Precision_Apply(this, x_in)
        x_out = error("Precision_Apply() not implemented");
    end

  end
end
