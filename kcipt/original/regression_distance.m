function D = regression_distance(Y, Z)
    % Distance || f(z_i) - f(z_j) ||_2 where Y = f(Z) is a regression function

    n = length(Z);
    dims = size(Z, 2);
    if n < 200
        width = 0.8;
    elseif n < 1200
        width = 0.5;
    else
        width = 0.3;
    end

    % Optimize GPR
    logtheta0 = [log(width * sqrt(dims))*ones(dims,1) ; 0; log(sqrt(0.1))];
    covfunc = {'covSum', {'covSEard','covNoise'}};
    [logtheta_y, fvals_y, iter_y] = minimize(...
        logtheta0, 'gpr_multi', -350, covfunc, Z, Y);
    covfunc_z = {'covSEard'};
    Kz_y = feval(covfunc_z{:}, logtheta_y, Z);
    Ry = pdinv(Kz_y + exp(2*logtheta_y(end))*eye(n));
    Fy = Y'*Ry*Kz_y;

    M = Fy'*Fy;
    O = ones(n, 1);
    N = O*diag(M)';
    D = sqrt(N + N' - 2*M);
end
