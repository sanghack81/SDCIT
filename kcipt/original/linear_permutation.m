function P = linear_permutation(D)
    n = length(D);

    % Rescale Distances
    D = D / max(max(D));

    % Objective Function
    f = reshape(D, [], 1);

    % Inequality contraint
    lb = zeros(n^2, 1);

    % Equality constraints
    Aeq = sparse(2*n, n^2);
    b = ones(2*n, 1);

    % Columns sum to 1
    for c = [0:n-1]
        Aeq(c + 1, c*n+1:(c+1)*n) = 1;
    end

    % Rows sum to 1 (last row constraint not necessary;
    % it is implied by other constraints)
    for r = [1:n-1]
        for c = [1:n]
            Aeq(r+n, r+(c-1)*n) = 1;
        end
    end

    % Diagonal entries zero
    for z = [1:n]
        Aeq(2*n, (z-1)*(n+1) + 1) = 1;
    end
    b(2*n, 1) = 0;

    % We use the simplex algorithm since we need a vertex solution
    % (so that the resulting matrix is a permutation)
    %options = optimset('LargeScale', 'off', 'Simplex', 'on', 'Display', 'iter');
    options = optimset('LargeScale', 'off', 'Simplex', 'on');
    vecPstar = linprog(f, [], [], Aeq, b, lb, [], [], options);
    P = reshape(vecPstar, n, n);
end
