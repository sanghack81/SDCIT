function k = rbf(sigma)
    % Returns an RBF kernel function with the given bandwidth
    constant = 1 / (2*(sigma^2));
    k = @(x,y) exp(-constant*dist2(x, y));
end
