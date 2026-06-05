%   Sampler_Interface
%
%   Author(s):
%       - Steven Maio (smaio@sandia.gov or smaio@ncsu.edu)

classdef Sampler_Interface < handle
    %% Sampler_Interface
    %   Interface for sampling from a probability distribution. This interface
    %   is used to implement BAE, which uses samples to construct a corrected
    %   likelihood function.

    methods (Abstract, Access = public)

        [samples] = Sample(this)

        dim = Dimension(this)

    end

end
