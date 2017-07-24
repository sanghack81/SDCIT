function data = synthetic(name, trial, N, args)
    if strcmp(name, 'henon')
        data = henon(trial, N, args);
    elseif strcmp(name, 'caseI')
        args.case = 1;
        data = zhang2012(trial, N, args);
    elseif strcmp(name, 'caseII')
        args.case = 2;
        data = zhang2012(trial, N, args);
    elseif strcmp(name, 'symmetric_caseI')
        args.case = 1;
        data = symmetric_zhang2012(trial, N, args);
    elseif strcmp(name, 'symmetric_caseII')
        args.case = 2;
        data = symmetric_zhang2012(trial, N, args);
    else
        error(sprintf('Unknown dataset "%s"', name));
    end
end

function data = henon(trial, N, args)
    if isfield(args, 'gamma')
        gamma = args.gamma;
    else
        gamma = 0.0;
    end
    if isfield(args, 'noise')
        noise = args.noise;
    else
        noise = 0;
    end

    % Seed RNG for trial number
    RandStream.setGlobalStream(RandStream('mcg16807','Seed',trial));

    X = [zeros(2*N, 2) normrnd(0, 0.5, 2*N, noise)];
    Y = [zeros(2*N, 2) normrnd(0, 0.5, 2*N, noise)];

    X(1, 1) = rand();
    X(1, 2) = rand();
    Y(1, 1) = rand();
    Y(1, 2) = rand();

    for t = 2:2*N
        X(t,1) = 1.4 - X(t-1,1)^2 + 0.3*X(t-1,2);
        X(t,2) = X(t-1,1);
        Y(t,1) = 1.4 - (gamma*X(t-1,1)*Y(t-1,1) + (1-gamma)*(Y(t-1,1)^2))...
                 + 0.1*Y(t-1,2);
        Y(t,2) = Y(t-1,1);
    end

    data.Xt1 = X(end-N+1:end,:);
    data.Yt1 = Y(end-N+1:end,:);
    data.Xt = X(end-N:end-1,:);
    data.Yt = Y(end-N:end-1,:);
end

function data = zhang2012(trial, N, args)
    if isfield(args, 'independent')
        independent = args.independent;
    else
        independent = 1;
    end
    if isfield(args, 'dimensions')
        dimensions = args.dimensions;
        if dimensions < 1
            error('Must have at least 1 dimension.')
        end
    else
        dimensions = 1;
    end
    if isfield(args, 'case')
        the_case = args.case;
        if the_case < 0 | the_case > 2
            error('Unsupported case.')
        end
    else
        the_case = 1;
    end

    % Seed RNG for trial number
    RandStream.setGlobalStream(RandStream('mcg16807','Seed',trial));

    if the_case == 1
        X = normrnd(0, 1, N, 1);
        Y = normrnd(0, 1, N, 1);
        Z = normrnd(0, 1, N, 1);
        ZZ1 = 0.7 * ((Z.^3 / 5) + (Z / 2));
        X = ZZ1 + tanh(X);
        X = X + (X.^3 / 3) + (tanh(X/3) / 2);
        ZZ2 = ((Z.^3 / 4) + Z) / 3;
        Y = Y + ZZ2;
        Y = Y + tanh(Y / 3);

        X = normalize(X);
        Y = normalize(Y);
        Z = normalize(Z);

        if dimensions > 1
            noisy_dims = normrnd(0, 1, N, dimensions - 1);
            Z = [Z noisy_dims];
        end
    else
        if dimensions > 5
            error('Between 1 and 5 dimensions supported.')
        end
        % Case II
        X = normrnd(0, 1, N, 1);
        Y = normrnd(0, 1, N, 1);
        Z = normrnd(0, 1, N, 1);
        ZZ1 = 0.7 * ((Z.^3 / 5) + (Z / 2));
        X = ZZ1 + tanh(X);
        X = X + (X.^3 / 3) + (tanh(X/3) / 2);
        ZZ2 = ((Z.^3 / 4) + Z) / 3;
        Y = Y + ZZ2;
        Y = Y + tanh(Y / 3);

        X = normalize(X);
        Y = normalize(Y);
        Z = normalize(Z);

        if dimensions > 1
            Z2 = normrnd(0, 1, N, 1);
            X = normrnd(0, 1, N, 1);
            Y = normrnd(0, 1, N, 1);
            ZZ1_2 = (ZZ1 / 2) + Z2;
            ZZ1_2 = (ZZ1_2 / 2) + 0.7*tanh(ZZ1_2);
            X = ZZ1_2 + tanh(X);
            X = X + ((X.^3) / 3) + tanh(X / 3) / 2;
            ZZ2_2 = ZZ2 / 2 + Z2;
            ZZ2_2 = ZZ2_2 / 2 + 0.7*tanh(ZZ2_2);
            Y = Y + ZZ2_2;
            Y = Y + tanh(Y / 3);

            X = normalize(X);
            Y = normalize(Y);
            Z2 = normalize(Z2);
            Z = [Z Z2];
        end

        if dimensions > 2
            Z3 = normrnd(0, 1, N, 1);
            X = normrnd(0, 1, N, 1);
            Y = normrnd(0, 1, N, 1);
            ZZ1_3 = ZZ1_2*2/3 + Z3*5/6;
            ZZ1_3 = ZZ1_3/2 + 0.7*tanh(ZZ1_3);
            X = ZZ1_3 + tanh(X);
            X = X + (X.^3)/3 + tanh(X/3)/2;
            ZZ2_3 = ZZ2_2*2/3 + Z3*5/6;
            ZZ2_3 = ZZ2_3/2 + 0.7*tanh(ZZ2_3);
            Y = Y + ZZ2_3;
            Y = Y + tanh(Y/3);

            X = normalize(X);
            Y = normalize(Y);
            Z3 = normalize(Z3);
            Z = [Z Z3];
        end

        if dimensions > 3
            Z4 = normrnd(0, 1, N, 1);
            X = normrnd(0, 1, N, 1);
            Y = normrnd(0, 1, N, 1);
            ZZ1_4 = ZZ1_3*2/3 + Z4*5/6;
            ZZ1_4 = ZZ1_4/2 + 0.7*tanh(ZZ1_4);
            X = ZZ1_4 + tanh(X);
            X = X + ((X.^3)/3) + tanh(X/3)/2;
            ZZ2_4 = ZZ2_3*2/3 + Z4*5/6;
            ZZ2_4 = ZZ2_4/2 + 0.7*tanh(ZZ2_4);
            Y = Y + ZZ2_4;
            Y = Y + tanh(Y/3);

            X = normalize(X);
            Y = normalize(Y);
            Z4 = normalize(Z4);
            Z = [Z Z4];
        end

        if dimensions > 4
            Z5 = normrnd(0, 1, N, 1);
            X = normrnd(0, 1, N, 1);
            Y = normrnd(0, 1, N, 1);
            ZZ1_5 = ZZ1_4*2/3 + Z5*5/6;
            ZZ1_5 = ZZ1_5/2 + 0.7*tanh(ZZ1_5);
            X = ZZ1_5 + tanh(X);
            X = X + ((X.^3)/3) + tanh(X/3)/2;
            ZZ2_5 = ZZ2_4*2/3 + Z5*5/6;
            ZZ2_5 = ZZ2_5/2 + 0.7*tanh(ZZ2_5);
            Y = Y + ZZ2_5;
            Y = Y + tanh(Y/3);

            X = normalize(X);
            Y = normalize(Y);
            Z5 = normalize(Z5);
            Z = [Z Z5];
        end
    end

    if ~independent
        ff = 0.5*normrnd(0, 1, N, 1);
        X = X + ff;
        Y = Y + ff;
    end

    data.X = X;
    data.Y = Y;
    data.Z = Z;
end

function data = symmetric_zhang2012(trial, N, args)
    if isfield(args, 'independent')
        independent = args.independent;
    else
        independent = 1;
    end
    if isfield(args, 'dimensions')
        dimensions = args.dimensions;
        if dimensions < 1 | dimensions > 5
            error('Between 1 and 5 dimensions supported.')
        end
    else
        dimensions = 1;
    end
    if isfield(args, 'case')
        the_case = args.case;
        if the_case < 0 | the_case > 2
            error('Unsupported case.')
        end
    else
        the_case = 1;
    end

    % Seed RNG for trial number
    RandStream.setGlobalStream(RandStream('mcg16807','Seed',trial));

    if the_case == 1
        X = normrnd(0, 1, N, 1);
        Y = normrnd(0, 1, N, 1);
        Z = normrnd(0, 1, N, 1);
        ZZ1 = 0.7 * ((Z.^3 / 5) + (Z / 2));
        X = ZZ1 + tanh(X);
        X = X + (X.^3 / 3) + (tanh(X/3) / 2);
        ZZ2 = ((Z.^3 / 4) + Z) / 3;
        Y = Y + ZZ2;
        Y = Y + tanh(Y / 3);

        X = normalize(X);
        Y = normalize(Y);
        Z = normalize(Z);

        if dimensions > 1
            noisy_dims = normrnd(0, 1, N, dimensions - 1);
            Z = [Z noisy_dims];
        end
    else
        % Case II
        X = normrnd(0, 1, N, 1);
        Y = normrnd(0, 1, N, 1);
        Z = normrnd(0, 1, N, 1);
        ZZ1 = 0.7 * ((Z.^3 / 5) + (Z / 2));
        X = ZZ1 + tanh(X);
        X = X + (X.^3 / 3) + (tanh(X/3) / 2);
        ZZ2 = ((Z.^3 / 4) + Z) / 3;
        Y = Y + ZZ2;
        Y = Y + tanh(Y / 3);

        X = normalize(X);
        Y = normalize(Y);
        Z = normalize(Z);

        if dimensions > 1
            Z2 = normrnd(0, 1, N, 1);
            X = normrnd(0, 1, N, 1);
            Y = normrnd(0, 1, N, 1);
            ZZ1_2 = (ZZ1 / 2) + Z2;
            ZZ1_2 = (ZZ1_2 / 2) + 0.7*tanh(ZZ1_2);
            X = ZZ1_2 + tanh(X);
            X = X + ((X.^3) / 3) + tanh(X / 3) / 2;
            ZZ2_2 = ZZ2 / 2 + Z2;
            ZZ2_2 = ZZ2_2 / 2 + 0.7*tanh(ZZ2_2);
            Y = Y + ZZ2_2;
            Y = Y + tanh(Y / 3);

            X = normalize(X);
            Y = normalize(Y);
            Z2 = normalize(Z2);
            Z = [Z Z2];
        end

        if dimensions > 2
            Z3 = normrnd(0, 1, N, 1);
            X = normrnd(0, 1, N, 1);
            Y = normrnd(0, 1, N, 1);
            ZZ1_3 = ZZ1_2*2/3 + Z3*5/6;
            ZZ1_3 = ZZ1_3/2 + 0.7*tanh(ZZ1_3);
            X = ZZ1_3 + tanh(X);
            X = X + (X.^3)/3 + tanh(X/3)/2;
            ZZ2_3 = ZZ2_2*2/3 + Z3*5/6;
            ZZ2_3 = ZZ2_3/2 + 0.7*tanh(ZZ2_3);
            Y = Y + ZZ2_3;
            Y = Y + tanh(Y/3);

            X = normalize(X);
            Y = normalize(Y);
            Z3 = normalize(Z3);
            Z = [Z Z3];
        end

        if dimensions > 3
            Z4 = normrnd(0, 1, N, 1);
            X = normrnd(0, 1, N, 1);
            Y = normrnd(0, 1, N, 1);
            ZZ1_4 = ZZ1_3*2/3 + Z4*5/6;
            ZZ1_4 = ZZ1_4/2 + 0.7*tanh(ZZ1_4);
            X = ZZ1_4 + tanh(X);
            X = X + ((X.^3)/3) + tanh(X/3)/2;
            ZZ2_4 = ZZ2_3*2/3 + Z4*5/6;
            ZZ2_4 = ZZ2_4/2 + 0.7*tanh(ZZ2_4);
            Y = Y + ZZ2_4;
            Y = Y + tanh(Y/3);

            X = normalize(X);
            Y = normalize(Y);
            Z4 = normalize(Z4);
            Z = [Z Z4];
        end

        if dimensions > 4
            Z5 = normrnd(0, 1, N, 1);
            X = normrnd(0, 1, N, 1);
            Y = normrnd(0, 1, N, 1);
            ZZ1_5 = ZZ1_4*2/3 + Z5*5/6;
            ZZ1_5 = ZZ1_5/2 + 0.7*tanh(ZZ1_5);
            X = ZZ1_5 + tanh(X);
            X = X + ((X.^3)/3) + tanh(X/3)/2;
            ZZ2_5 = ZZ2_4*2/3 + Z5*5/6;
            ZZ2_5 = ZZ2_5/2 + 0.7*tanh(ZZ2_5);
            Y = Y + ZZ2_5;
            Y = Y + tanh(Y/3);

            X = normalize(X);
            Y = normalize(Y);
            Z5 = normalize(Z5);
            Z = [Z Z5];
        end
    end

    ff = 0.5*normrnd(0, 1, N, 1);
    X = X + ff;
    if independent
        ff = 0.5*normrnd(0, 1, N, 1);
    end
    Y = Y + ff;

    data.X = X;
    data.Y = Y;
    data.Z = Z;
end

function X = normalize(X)
    X = X - mean(X);
    X = X / std(X);
end
