function [statistic null] = kcipt(X, Y, Z, k_X, k_Y, k_Z, options)
    % Performs the KCIPT with the given distance metric and null estimation

    % Parse Options
    if nargin < 7
        options.distance = 'regression';
        options.null_estimate = 'gamma';
        options.bootstrap_samples = 1000;
        options.kernel = 'supplied';
        options.split = 1;
    else
        if ~isfield(options, 'distance')
            options.distance = 'regression';
        end
        if ~isfield(options, 'null_estimate')
            options.distance = 'gamma';
        end
        if ~isfield(options, 'bootstrap_samples')
            options.bootstrap_samples = 1000;
        end
        if ~isfield(options, 'kernel')
            options.kernel = 'supplied';
        end
        if ~isfield(options, 'split')
            options.split = 1;
        end
    end

    n = length(X);
    halfn = floor(n/2);

    % Split dataset in half?
    if options.split
        X1 = X(1:halfn,:);
        X2 = X(halfn+1:2*halfn,:);
        Y1 = Y(1:halfn,:);
        Y2 = Y(halfn+1:2*halfn,:);
        Z1 = Z(1:halfn,:);
        Z2 = Z(halfn+1:2*halfn,:);
    else
        X1 = X; X2 = X;
        Y1 = Y; Y2 = Y;
        Z1 = Z; Z2 = Z;
    end

    % Compute distance
    if strcmp(options.distance, 'regression')
        D = regression_distance(Y1, Z1);
    elseif strcmp(options.distance, 'symmetric_regression')
        D = regression_distance([X1 Y1], Z1);
    elseif strcmp(options.distance, 'rkhs')
        D = rkhs_distance(Z1, k_Z);
    elseif strcmp(options.distance, 'random')
        D = [];
    else
        error(sprintf('Unknown distance metric "%s"',...
                      options.distance));
    end

    % Compute permutation
    if isempty(D)
        P = eye(halfn);
        [notUsed, indperm] = sort(rand(halfn, 1));
        P = P(indperm, :);
    else
        P = linear_permutation(D);
    end
    Y1_0 = Y1;
    Y1 = P'*Y1;

    % Compute statistic
    if strcmp(options.kernel, 'supplied')
        K = k_X(X1, X1).*k_Y(Y1, Y1).*k_Z(Z1, Z1);
        L = k_X(X2, X2).*k_Y(Y2, Y2).*k_Z(Z2, Z2);
        KL = k_X(X1, X2).*k_Y(Y1, Y2).*k_Z(Z1, Z2);

    elseif strcmp(options.kernel, 'maxmmd')
        [K L KL] = maxmmd(X1, Y1, Z1, X2, Y2, Z2);

    elseif strcmp(options.kernel, 'yzonly')
        K = k_Y(Y1, Y1).*k_Z(Z1, Z1);
        L = k_Y(Y1_0, Y1_0).*k_Z(Z2, Z2);
        KL = k_Y(Y1, Y1_0).*k_Z(Z1, Z2);

    else
        error(sprintf('Unknown kernel "%s"',...
                      options.kernel));
    end

    statistic = sum(sum(K + L - KL - KL')) / halfn;

    % Estimate Null
    if strcmp(options.null_estimate, 'gamma')
        meanMMD = (2/halfn)*(1 - trace(KL)/halfn);
        K = K - diag(diag(K));
        L = L - diag(diag(L));
        KL = KL - diag(diag(KL));
        varMMD = (2/((halfn-1)^2*halfn^2))*sum(sum((K + L - KL - KL').^2));
        alpha = meanMMD^2 / varMMD;
        beta = halfn*varMMD / meanMMD;
        null = GammaNull(alpha, beta);

    elseif strcmp(options.null_estimate, 'bootstrap')

        KK = [K KL; KL' L];
        mmds = zeros(options.bootstrap_samples, 1);
        progress = ProgressMonitor(options.bootstrap_samples, 'Bootstrapping', 10);
        for b=1:options.bootstrap_samples
            [notUsed, indperm] = sort(rand(2*halfn,1));
            KKperm = KK(indperm,indperm);
            K = KKperm(1:halfn,1:halfn);
            L = KKperm(halfn+1:2*halfn,halfn+1:2*halfn);
            KL = KKperm(1:halfn,halfn+1:2*halfn);
            mmds(b) = sum(sum(K + L - KL - KL')) / halfn;
            progress.increment();
        end
        null = EmpiricalNull(mmds);

    else
        error(sprintf('Unknown null estimation technique "%s"',...
                      options.null_estimate));
    end
end
