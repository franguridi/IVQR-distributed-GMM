function sim_out = Perform_simulations(sim_params)

n_sim = sim_params.n_simulations;
n = sim_params.n;
B = sim_params.B;
lambda_ini = sim_params.lambda_ini;
lambda_step = sim_params.lambda_step;
n_lambdas = sim_params.n_lambdas;
psi = sim_params.psi;
tau = sim_params.tau;
r_seed_start = sim_params.r_seed_start;
%calculate_full_sample_estimator = sim_params.calculate_full_sample_estimator;
theta_ini_DGMM = sim_params.theta_ini_DGMM;
tol = sim_params.obj_step_tolerance;
maxiter = sim_params.max_iterations;
theta_lb = sim_params.theta_lb;
theta_ub = sim_params.theta_ub;
eps = sim_params.eps;
moment_norm = sim_params.moment_norm;
descent_type = sim_params.descent_type;

Q_sqr = eye(4);

%NON-PARALLEL:
%{
sim_out.LR.bias = 0;
sim_out.LR.rmse = 0;
sim_out.LR.l1_sample_moments = 0;
sim_out.LR.comp_time = 0;
sim_out.DGMM.bias = 0;
sim_out.DGMM.rmse = 0;
sim_out.DGMM.l1_sample_moments = 0;
sim_out.DGMM.comp_time = 0;
%}

deviations_DGMM = zeros(4,n_sim);
comptime_DGMM = zeros(1,n_sim);
l1norm_DGMM = zeros(1,n_sim);
deviations_LR = zeros(4,n_sim);
comptime_LR = zeros(1,n_sim);
l1norm_LR = zeros(1,n_sim);

parfor sim_idx = 1:n_sim
    
    disp(['SIMULATION #' num2str(sim_idx)]);
    r_seed = r_seed_start + sim_idx - 1;
    rng(r_seed); %this is for random block descent
    
    % GENERATE DATA:
    [Y,W,Z,theta_true] = Generate_ChenLee(n,tau,r_seed);
    
    tic;
    % DGMM ESTIMATOR (STARTING POINT FOR LAGRANGE):
    disp(' ================== STARTING DGMM ESTIMATION ==================');
    psi_DGMM = 0;
    Lambda_DGMM = 0;
    descent_type_DGMM = 'cyclical';
    maxiter_DGMM = B;
    [theta_DGMM, l1_opt_sample_moments_DGMM, exit_status] = Optimize_across_blocks_Lagrange(Y,W,Z,Q_sqr,tau,theta_ini_DGMM,Lambda_DGMM,psi_DGMM,B,maxiter_DGMM,tol,theta_lb,theta_ub,eps,moment_norm,descent_type_DGMM,sim_idx);
    
    comptime_DGMM(:,sim_idx) = toc/n_sim;
    deviations_DGMM(:,sim_idx) = theta_DGMM-theta_true;
    l1norm_DGMM(:,sim_idx) = l1_opt_sample_moments_DGMM;
    
    %NON-PARALLEL:
    %sim_out.DGMM.comp_time = sim_out.DGMM.comp_time + toc/n_sim;
    %sim_out.DGMM.bias = sim_out.DGMM.bias + (theta_DGMM-theta_true)/n_sim;
    %sim_out.DGMM.rmse = sim_out.DGMM.rmse + ((theta_DGMM-theta_true).^2)/n_sim;
    %sim_out.DGMM.l1_sample_moments = sim_out.DGMM.l1_sample_moments + l1_opt_sample_moments_DGMM/n_sim;

    % LAGRANGE BLOCK DESCENT ESTIMATOR:
    disp(' ================== STARTING LAGRANGE DESCENT ESTIMATION ==================');
    [theta_LR, l1_opt_sample_moments_LR, exit_status] = Optimize_across_blocks_Lagrange(Y,W,Z,Q_sqr,tau,theta_DGMM,lambda_ini,psi,B,maxiter,tol,theta_lb,theta_ub,eps,moment_norm,descent_type,sim_idx);
    for k=1:n_lambdas-1
        Lambda = lambda_ini + k*lambda_step;
        [theta_LR, l1_opt_sample_moments_LR, exit_status] = Optimize_across_blocks_Lagrange(Y,W,Z,Q_sqr,tau,theta_LR,Lambda,psi,B,maxiter,tol,theta_lb,theta_ub,eps,moment_norm,descent_type,sim_idx);
    end
    
    comptime_LR(:,sim_idx) = toc/n_sim;
    deviations_LR(:,sim_idx) = theta_LR-theta_true;
    l1norm_LR(:,sim_idx) = l1_opt_sample_moments_LR;
    
    %NON-PARALLEL:
    %sim_out.LR.comp_time = sim_out.LR.comp_time + toc/n_sim;
    %sim_out.LR.bias = sim_out.LR.bias + (theta_LR-theta_true)/n_sim;
    %sim_out.LR.rmse = sim_out.LR.rmse + ((theta_LR-theta_true).^2)/n_sim;
    %sim_out.LR.l1_sample_moments = sim_out.LR.l1_sample_moments + l1_opt_sample_moments_LR/n_sim;
    
end %end simulation loop 


sim_out.DGMM.comp_time = mean(comptime_DGMM);
sim_out.DGMM.l1_sample_moments = mean(l1norm_DGMM);
sim_out.DGMM.bias = mean(deviations_DGMM,2);
sim_out.DGMM.rmse = mean(deviations_DGMM.^2,2);

sim_out.LR.comp_time = mean(comptime_LR);
sim_out.LR.l1_sample_moments = mean(l1norm_LR);
sim_out.LR.bias = mean(deviations_LR,2);
sim_out.LR.rmse = mean(deviations_LR.^2,2);

%NON-PARALLEL:
%sim_out.DGMM.rmse = sqrt(sim_out.DGMM.rmse);
%sim_out.LR.rmse = sqrt(sim_out.LR.rmse);

end
