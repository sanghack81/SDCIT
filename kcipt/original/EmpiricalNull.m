classdef EmpiricalNull < handle
    properties
        empirical_sample
    end
    methods
        function obj = EmpiricalNull(empirical_sample)
            obj.empirical_sample = reshape(empirical_sample, 1, []);
        end

        function p = pvalue(obj, statistic)
            SAMP = repmat(obj.empirical_sample, length(statistic), 1);
            STAT = repmat(statistic', 1, length(obj.empirical_sample));
            p = sum((SAMP > STAT) / length(obj.empirical_sample));
        end

        function s = sample(obj, n)
            s = obj.empirical_sample(randi(length(obj.empirical_sample), n, 1));
        end
    end
end
