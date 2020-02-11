function [Y,W,Z,theta_true] = Generate_ChenLee(n,tau,r_seed)
% Generates path of Chen & Lee DGP (see eq. 4.1)
%INPUT:
%n - sample size
%tau - quantile level (used to calculate true parameter values)
%OUTPUT:
%Y - nx1 dependent variable
%W - nx4 matrix of endogenous regressors
%Z - nx4 matrix of instruments
%theta_true - 4x1 vector of true parameter values

rng(r_seed);
V = eye(4);
V(1,:) = [1,0.4,0.6,-0.2];
V(:,1) = [1,0.4,0.6,-0.2];
Z = randn(n,3); %last three instruments, Z1,Z2,Z3
eps_v = mvnrnd(zeros(n,4),0.25*V); %(eps,v_1,v_2,v_3)
D = normcdf(Z+eps_v(:,2:4))*diag([1,2,1.5]); %nx3 exogenous variables
W = [ones(n,1), D]; %nx4 regressor matrix
Y = 1 + sum(D,2) + (0.5+D(:,1)+0.25*D(:,2)+0.15*D(:,3)).*eps_v(:,1);
Z = [ones(n,1), Z]; %add intercept

norminv_tau = norminv(tau); %F_eps^{-1}(tau)
theta_true = [1+0.5*norminv_tau, 1+norminv_tau, 1+0.25*norminv_tau, 1+0.15*norminv_tau]';

end

