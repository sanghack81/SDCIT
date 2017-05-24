function [Sta, Cri, p_val, Cri_appr, p_appr] = CInd_test_new_withGP_Lee(Kx, Ky, Kz, alpha)
% To test if x and y are independent.
% INPUT:
%   The number of rows of x and y is the sample size.
%   alpha is the significance level (we suggest 1%).
%   width contains the kernel width.
% Output:
%   Cri: the critical point at the p-value equal to alpha obtained by bootstrapping.
%   Sta: the statistic Tr(K_{\ddot{X}|Z} * K_{Y|Z}).
%   p_val: the p value obtained by bootstrapping.
%   Cri_appr: the critical value obtained by Gamma approximation.
%   p_apppr: the p-value obtained by Gamma approximation.
% If Sta > Cri, the null hypothesis (x is independent from y) is rejected.
% Copyright (c) 2010-2011  ...
% All rights reserved.  See the file COPYING for license terms.

% Controlling parameters
IF_unbiased = 0;
IF_GP = 0;  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Approximate = 1;
% Bootstrap = 1; % Note: set to 0 to save time if you do not use simulation to generate the null !!!


T = size(Kx, 2);
Num_eig = T;
T_BS = 5000;
lambda = 1E-3; % the regularization paramter  %%%%Problem
Thresh = 1E-5;
% normalize the data
D = size(Kx, 2);
logtheta_x = []; logtheta_y = [];  df_x = []; df_y = [];
Cri = []; Sta = []; p_val = []; Cri_appr = []; p_appr = [];


H =  eye(T) - ones(T,T)/T; % for centering of the data in feature space
Kx = Kx.*Kz;
Kx = H * Kx * H;
Ky = H * Ky * H;

if IF_GP
    % learning the hyperparameters
    [eig_Kx, eix] = eigdec((Kx+Kx')/2, min(400, floor(T/4))); % /2
    [eig_Ky, eiy] = eigdec((Ky+Ky')/2, min(200, floor(T/5))); % /3
    
    covfunc = {'covSum', {'covSEard','covNoise'}};
    logtheta0 = [log(0.75 * sqrt(D))*ones(D,1) ; 0; log(sqrt(0.1))];
    fprintf('Optimizing hyperparameters in GP regression...\n');
    %
    IIx = find(eig_Kx > max(eig_Kx) * Thresh); eig_Kx = eig_Kx(IIx); eix = eix(:,IIx);
    IIy = find(eig_Ky > max(eig_Ky) * Thresh); eig_Ky = eig_Ky(IIy); eiy = eiy(:,IIy);
    [logtheta_x, fvals_x, iter_x] = minimize(logtheta0, 'gpr_multi', -350, covfunc, z, 2*sqrt(T) *eix * diag(sqrt(eig_Kx))/sqrt(eig_Kx(1)));
    [logtheta_y, fvals_y, iter_y] = minimize(logtheta0, 'gpr_multi', -350, covfunc, z, 2*sqrt(T) *eiy * diag(sqrt(eig_Ky))/sqrt(eig_Ky(1)));
    
    covfunc_z = {'covSEard'};
    Kz_x = feval(covfunc_z{:}, logtheta_x, z);
    Kz_y = feval(covfunc_z{:}, logtheta_y, z);
    
    % Note: in the conditional case, no need to do centering, as the regression
    % will automatically enforce that.
    
    % Kernel matrices of the errors
    P1_x = (eye(T) - Kz_x*pdinv(Kz_x + exp(2*logtheta_x(end))*eye(T)));
    Kxz = P1_x* Kx * P1_x';
    P1_y = (eye(T) - Kz_y*pdinv(Kz_y + exp(2*logtheta_y(end))*eye(T)));
    Kyz = P1_y* Ky * P1_y';
    % calculate the statistic
    Sta = trace(Kxz * Kyz);
    
    % degrees of freedom
    df_x = trace(eye(T)-P1_x);
    df_y = trace(eye(T)-P1_y);
else
    Kz = H * Kz * H; %*4 % as we will calculate Kz
    % Kernel matrices of the errors
    P1 = (eye(T) - Kz*pdinv(Kz + lambda*eye(T)));
    Kxz = P1* Kx * P1';
    Kyz = P1* Ky * P1';
    % calculate the statistic
    Sta = trace(Kxz * Kyz);
    % degrees of freedom
    df = trace(eye(T)-P1);
end

% calculate the eigenvalues
% Due to numerical issues, Kxz and Kyz may not be symmetric:
[eig_Kxz, eivx] = eigdec((Kxz+Kxz')/2,Num_eig);
[eig_Kyz, eivy] = eigdec((Kyz+Kyz')/2,Num_eig);

% % calculate Cri...
% calculate the product of the square root of the eigvector and the eigen
% vector
IIx = find(eig_Kxz > max(eig_Kxz) * Thresh);
IIy = find(eig_Kyz > max(eig_Kyz) * Thresh);
eig_Kxz = eig_Kxz(IIx);
eivx = eivx(:,IIx);
eig_Kyz = eig_Kyz(IIy);
eivy = eivy(:,IIy);

eiv_prodx = eivx * diag(sqrt(eig_Kxz));
eiv_prody = eivy * diag(sqrt(eig_Kyz));
clear eivx eig_Kxz eivy eig_Kyz

% calculate their product
Num_eigx = size(eiv_prodx, 2);
Num_eigy = size(eiv_prody, 2);
Size_u = Num_eigx * Num_eigy;
uu = zeros(T, Size_u);
for i=1:Num_eigx
    for j=1:Num_eigy
        uu(:,(i-1)*Num_eigy + j) = eiv_prodx(:,i) .* eiv_prody(:,j);
    end
end

if Size_u > T
    uu_prod = uu * uu';
else
    uu_prod = uu' * uu;
end

Cri=-1;
p_val=-1;
Cri_appr=-1;
p_appr=-1;

% if Bootstrap
%     eig_uu = eigdec(uu_prod,min(T,Size_u));
%     II_f = find(eig_uu > max(eig_uu) * Thresh);
%     eig_uu = eig_uu(II_f);
%     
%     % use mixture of F distributions to generate the Null dstr
%     if length(eig_uu) * T < 1E6
%         f_rand1 = chi2rnd(1,length(eig_uu),T_BS);
%         if IF_unbiased
%             Null_dstr = T^2/(T-1-df_x)/(T-1-df_y) * eig_uu' * f_rand1; %%%%Problem
%         else
%             Null_dstr = eig_uu' * f_rand1;
%         end
%     else
%         % iteratively calcuate the null dstr to save memory
%         Null_dstr = zeros(1,T_BS);
%         Length = max(floor(1E6/T),100);
%         Itmax = floor(length(eig_uu)/Length);
%         for iter = 1:Itmax
%             f_rand1 = chi2rnd(1,Length,T_BS);
%             if IF_unbiased
%                 Null_dstr = Null_dstr + T^2/(T-1-df_x)/(T-1-df_y) *... %%%%Problem
%                     eig_uu((iter-1)*Length+1:iter*Length)' * f_rand1;
%             else
%                 Null_dstr = Null_dstr + ... %%%%Problem
%                     eig_uu((iter-1)*Length+1:iter*Length)' * f_rand1;
%             end
%         end
%     end
%     
%     % use chi2 to generate the Null dstr:
%     sort_Null_dstr = sort(Null_dstr);
%     Cri = sort_Null_dstr(ceil((1-alpha)*T_BS));
%     p_val = sum(Null_dstr>Sta)/T_BS;
% end

if Approximate
    mean_appr = trace(uu_prod);
    var_appr = 2*trace(uu_prod^2);
    k_appr = mean_appr^2/var_appr;
    theta_appr = var_appr/mean_appr;
    Cri_appr = gaminv(1-alpha, k_appr, theta_appr);
    p_appr = 1-gamcdf(Sta, k_appr, theta_appr);
end
