%Generates data from Chen & Lee paper
%and calculates Lagrangian estimator of IVQR GMM 

%PARAMETERS ------------------------------------------
n = 1000;
tau = 0.5;
B = 10; %number of blocks
Lambda = 20; %Lagrange multiplier
tol = 0.00001;
Q_sqr = eye(4); %Chen & Lee DGP has 4 end vars and 4 instruments
theta_ini = 1+0.5*randn(4,1);
% -----------------------------------------------------

[Y,W,Z,theta_true] = Generate_ChenLee(n,tau);
theta_estimate = Optimize_across_blocks_Lagrange(Y,W,Z,Q_sqr,tau,theta_ini,Lambda,B,tol);