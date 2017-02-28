function btest = bootstrap(test, boptions)
    % Parse Options
    if nargin < 2
        boptions.bootstrap_samples = 100;
        boptions.null_samples = 10000;
    else
        if ~isfield(boptions, 'bootstrap_samples')
            boptions.bootstrap_samples = 100;
        end
        if ~isfield(boptions, 'null_samples')
            boptions.null_samples = 10000;
        end
    end

    function [statistic null] = newtest(X, Y, Z, k_X, k_Y, k_Z, options)
        stats = [];
        nulls = {};
        progress = ProgressMonitor(boptions.bootstrap_samples, 'Outer Bootstrap', 10);
        for b=1:boptions.bootstrap_samples
            [statistic null] = test(X, Y, Z, k_X, k_Y, k_Z, options);
            stats(end+1) = statistic;
            nulls{end+1} = null;

            % Shuffle
            [notUsed, indperm] = sort(rand(length(X),1));
            X = X(indperm, :);
            Y = Y(indperm, :);
            Z = Z(indperm, :);

            progress.increment();
        end
        statistic = mean(stats);
        samples = [];
        for n=nulls
            samples = [samples; n{1}.sample(boptions.null_samples)];
        end
        sample = mean(samples, 1);
        null = EmpiricalNull(sample);
    end
    btest = @newtest;
end
