classdef GammaNull < handle
    properties
        alpha
        beta
    end
    methods
        function obj = GammaNull(alpha, beta)
            obj.alpha = alpha;
            obj.beta = beta;
        end

        function p = pvalue(obj, statistic)
            p = 1 - gamcdf(statistic, obj.alpha, obj.beta);
        end

        function s = sample(obj, n)
            s = gamrnd(obj.alpha, obj.beta, 1, n);
        end
    end
end
