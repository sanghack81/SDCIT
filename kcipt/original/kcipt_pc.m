function ci_test = kcipt_pc(dataset, alpha)
    function accept = test(x, y, S)
        S = unique(S);
        X = dataset(:,x);
        Y = dataset(:,y);
        if length(S) == 0
            Z = zeros(size(X));
            k_Z = rbf(1.0);
            options.distance = 'random';
            boptions.bootstrap_samples = 10;
        else
            Z = dataset(:,S);
            k_Z = rbf(median_pdist(Z));
            options.distance = 'rkhs';
            boptions.bootstrap_samples = 10;
        end
        k_X = rbf(median_pdist(X));
        k_Y = rbf(median_pdist(Y));
        new_test = bootstrap(@kcipt, boptions);
        options.null_estimate = 'bootstrap';
        options.bootstrap_samples = 10000;
        [statistic null] = new_test(X, Y, Z, k_X, k_Y, k_Z, options);
        pvalue = null.pvalue(statistic)
        accept = (pvalue > alpha);
    end
    ci_test = @test;
end
