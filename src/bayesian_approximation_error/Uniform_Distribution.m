classdef Uniform_Distribution < Sampler_Interface
    properties
      a
      b
      dim
    end

    methods (Access = public)

        function this = Uniform_Distribution(a, b)
            this@Sampler_Interface();
            this.dim = size(a, 1);
            this.a = a;
            this.b = b;
        end

        function [x_out] = Sample(this)
            x_out = this.a + (this.b - this.a) .* rand(this.dim, 1);
        end

        function dim = Dimension(this)
            dim = this.dim;
        end

        function [x_out] = Get_Mean(this)
            x_out = (a + b) / 2;
        end
    end
end
