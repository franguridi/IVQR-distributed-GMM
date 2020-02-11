function [theta_final,l1_opt_sample_moments,exit_status] = Optimize_across_blocks_Lagrange(Y,W,Z,Q_sqr,tau,theta_ini,Lambda,psi,B,maxiter,tol,theta_lb,theta_ub,eps,moment_norm,descent_type,sim_idx)
%Performs optimization across blocks
%using Lagrangian formulation of the IVQR problem
%
%INPUT:
%Y - nx1 vector 
%W - nxd vector of endogenous variables
%Z - nxl vector of instruments
%Q_sqr - lxd square root of weighting matrix
%tau - 1x1 quantile level
%theta_ini - dx1 initial value of theta
%Lambda - penalization parameter
%B - number of blocks
%tol - function value step tolerance

%TODO:
if rem(size(Y,1),B) ~= 0
    error('Not implemented: B is not a divisor of n');
end
m = size(Y,1)/B;

thetas = repmat(theta_ini,1,B);
es = zeros(m,B);
for k=1:B
    idx_k = (k-1)*m+1:k*m;
    es(:,k) = (Y(idx_k)-W(idx_k,:)*thetas(:,k)) <= 0;
end
obj_prev = Inf;
obj_step = Inf;
L_prev = Inf;
L_step = Inf;
b = 0;
iter = 0;
M = Determine_M(Y,W,theta_lb,theta_ub);
x_prev = NaN;

while obj_step > tol && iter < maxiter %%&& L_step > tol %TODO: tol_L and  stopping criterion in terms of minimizer?
    
    iter = iter + 1;
    
    switch descent_type
        case 'cyclical'
            b = mod(b,B)+1;
        case 'random'
            b_new = ceil(B*rand()); %random element of {1,...,B}
            if B~=1
                while b_new == b
                    b_new = ceil(B*rand());
                end
            end
            b = b_new;
        otherwise
            error('Not implemented');
    end
    
    disp([' ================ SIMULATION #' num2str(sim_idx) '===================']);
    disp([' ================ ITERATION #' num2str(iter) ' ================']);
    disp([' ================== BLOCK #' num2str(b) ' ==================']);
    disp([' =========================='           '====================']);
    opt_results = Optimize_within_block_Lagrange(Y,W,Z,Q_sqr,tau,thetas,es,x_prev,Lambda,psi,b,M,eps,moment_norm);
    L_curr = opt_results.obj_val_term_1;
    obj_curr = opt_results.obj_val;
    x_prev = opt_results.x;
    
    L_step = abs(L_curr - L_prev);
    L_prev = L_curr;
    obj_step = abs(obj_curr - obj_prev);
    obj_prev = obj_curr;
    
    % UPDATE THETAS AND E'S MATRICES WITH NEW VALUES:
    thetas(:,b) = opt_results.theta_opt;
    es(:,b) = opt_results.e;
end
theta_final = mean(thetas,2);
%TODO: change to theta_final !!!
%TODO: take final estimator to be opt_results.theta_opt for Lagrangian
%formulation
%otherwise sacrificing in terms of sample moment norm
%Calculate the L1 norm of sample moment conditions at theta_final:
l1_opt_sample_moments = sum(abs(((Y-W*theta_final<=0)-tau)'*Z*Q_sqr));

if iter == maxiter
    exit_status = ['Iteration limit reached, # iterations = ' num2str(iter)];
else
    exit_status = ['Objective value step tolerance reached, # iterations = ' num2str(iter)];
end

end
