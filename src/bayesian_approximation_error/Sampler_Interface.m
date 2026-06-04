classdef Sampler_Interface < handle

    methods (Abstract, Access = public)

        [samples] = Sample(this)

        dim = Dimension(this)

    end

end
