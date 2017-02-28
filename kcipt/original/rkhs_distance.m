function D = rkhs_distance(Z, kernel)
    % Pairwise RKHS Distances
    K = kernel(Z, Z);
    O = ones(length(Z), 1);
    N = O*diag(K)';
    D = sqrt(N + N' - 2*K);
end
