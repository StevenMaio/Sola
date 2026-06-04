classdef Uniform_Distribution < Param_Distribution
    properties
      a
      b
    end

    methods (Access = public)

        function this = Uniform_Distribution(a, b)
            this@Param_Distribution(size(a, 1));
            this.dim = size(a, 1);
            this.a = a;
            this.b = b;
        end

        function [x_out] = Sample(this, num_samples)
            arguments
                this
                num_samples = 1
            end
            x_out = this.a + (this.b - this.a) .* rand(num_samples, this.dim);
        end

        function [x_out] = Get_Mean(this)
            x_out = (a + b) / 2;
        end
    end
end
