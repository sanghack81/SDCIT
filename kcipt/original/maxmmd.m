function [K L KL] = maxmmd(X1, Y1, Z1, X2, Y2, Z2)
    n = length(X1);

    % Precompute distances
    DX1 = dist2(X1, X1);
    DY1 = dist2(Y1, Y1);
    DZ1 = dist2(Z1, Z1);

    DX2 = dist2(X2, X2);
    DY2 = dist2(Y2, Y2);
    DZ2 = dist2(Z2, Z2);

    DX12 = dist2(X1, X2);
    DY12 = dist2(Y1, Y2);
    DZ12 = dist2(Z1, Z2);

    function [f fprime] = objective(sigmas)
        if min(sigmas) < 0
            f = 0;
            fprime = [0 0 0];
            return
        end
        sigma_x = sigmas(1);
        sigma_y = sigmas(2);
        sigma_z = sigmas(3);

        K = exp(-sigma_x*DX1).*exp(-sigma_y*DY1).*exp(-sigma_z*DZ1);
        L = exp(-sigma_x*DX2).*exp(-sigma_y*DY2).*exp(-sigma_z*DZ2);
        KL = exp(-sigma_x*DX12).*exp(-sigma_y*DY12).*exp(-sigma_z*DZ12);
        f = -sum(sum(K + L - KL - KL')) / n;

        dKx = -DX1.*K;
        dLx = -DX2.*L;
        dKLx = -DX12.*KL;

        dKy = -DY1.*K;
        dLy = -DY2.*L;
        dKLy = -DY12.*KL;

        dKz = -DZ1.*K;
        dLz = -DZ2.*L;
        dKLz = -DZ12.*KL;

        dMMDx = sum(sum(dKx + dLx - dKLx - dKLx'));
        dMMDy = sum(sum(dKy + dLy - dKLy - dKLy'));
        dMMDz = sum(sum(dKz + dLz - dKLz - dKLz'));
        fprime = -[dMMDx dMMDy dMMDz] / n;
    end

    X = [X1; X2];
    Y = [Y1; Y2];
    Z = [Z1; Z2];

    sigma_0 = [median_pdist(X) median_pdist(Y) median_pdist(Z)];
    lb = sigma_0 / 2;
    ub = sigma_0 * 2;
    sigma_0
    -objective(sigma_0)

    options = optimset('GradObj', 'on');
    sigma_star = fmincon(@objective, sigma_0, [], [], [], [],...
                         lb, ub, [], options);
    sigma_star
    -objective(sigma_star)

    sigma_x = sigma_star(1);
    sigma_y = sigma_star(2);
    sigma_z = sigma_star(3);
    K = exp(-sigma_x*DX1).*exp(-sigma_y*DY1).*exp(-sigma_z*DZ1);
    L = exp(-sigma_x*DX2).*exp(-sigma_y*DY2).*exp(-sigma_z*DZ2);
    KL = exp(-sigma_x*DX12).*exp(-sigma_y*DY12).*exp(-sigma_z*DZ12);
end
