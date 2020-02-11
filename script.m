%Generates data from Chen & Lee paper
%and calculates Lagrangian estimator of IVQR GMM 

%PARAMETERS ------------------------------------------
r_seed = 4;
n = 100;
tau = 0.5;
B = 5; %number of blocks
Lambda = 0.1; %Lagrange multiplier
psi = 1;
maxiter = 300;
tol = -1;
Q_sqr = eye(4); %Chen & Lee DGP has 4 end vars and 4 instruments
moment_norm = 'L1';
% INITIAL POINT FOR DGMM ESTIMATOR:
theta_ini_DGMM = zeros(4,1);
%theta_ini_DGMM =  [1;1;1;1]+0.5*randn(4,1); 
%theta_ini_DGMM =  theta_estimate; 
theta_lb = -20;
theta_ub = 20;
eps = 0.0001;
descent_type = 'cyclical';

% FULL SAMPLE ESTIMATION PARAMS:
calculate_full_sample_estimator = true;
theta_ini_full_sample = zeros(4,1);
% -----------------------------------------------------

% GENERATE DATA:
[Y,W,Z,theta_true] = Generate_ChenLee(n,tau,r_seed);

% FULL SAMPLE ESTIMATOR:
if calculate_full_sample_estimator
    disp(' ================== STARTING FULL SAMPLE ESTIMATION ==================');
    psi_full_sample = 1;
    Lambda_full_sample = 0;
    B_full_sample = 1;
    maxiter_full_sample = 1;
    [theta_full_sample, l1_opt_sample_moments_full, exit_status] = Optimize_across_blocks_Lagrange(Y,W,Z,Q_sqr,tau,theta_ini_full_sample,Lambda_full_sample,psi_full_sample,B_full_sample,maxiter_full_sample,tol,theta_lb,theta_ub,eps,moment_norm,'cyclical');
end    

tic;
% DGMM ESTIMATOR (STARTING POINT FOR LAGRANGE):
disp(' ================== STARTING DGMM ESTIMATION ==================');
psi_DGMM = 0;
Lambda_DGMM = 0;
descent_type_DGMM = 'cyclical';
maxiter_DGMM = B;
[theta_DGMM, l1_opt_sample_moments_DGMM, exit_status] = Optimize_across_blocks_Lagrange(Y,W,Z,Q_sqr,tau,theta_ini_DGMM,Lambda_DGMM,psi_DGMM,B,maxiter_DGMM,tol,theta_lb,theta_ub,eps,moment_norm,descent_type_DGMM);


% LAGRANGE BLOCK DESCENT ESTIMATOR:
disp(' ================== STARTING LAGRANGE DESCENT ESTIMATION ==================');
[theta_Lagrange, l1_opt_sample_moments_Lagrange, exit_status] = Optimize_across_blocks_Lagrange(Y,W,Z,Q_sqr,tau,theta_DGMM,Lambda,psi,B,maxiter,tol,theta_lb,theta_ub,eps,moment_norm,descent_type);

time = toc;
disp('=======================================================');
if calculate_full_sample_estimator
    disp('Full sample estimator = ');
    disp(theta_full_sample');
    disp(['L1 norm of sample moments at optimum = ' num2str(l1_opt_sample_moments_full)]);
end
disp('---------------------------------------------------------');
disp('Starting point for DGMM estimation = ');
disp(theta_ini_DGMM');
disp('Starting point for Lagrange estimation (DGMM estimate) = ');
disp(theta_DGMM');
disp(['L1 norm of sample moments at optimum = ' num2str(l1_opt_sample_moments_DGMM)]);
disp('---------------------------------------------------------');
disp('Lagrange estimate = ');
disp(theta_Lagrange');
disp(['L1 norm of sample moments at optimum = ' num2str(l1_opt_sample_moments_Lagrange)]);
disp(['Time elapsed = ' num2str(toc)]);