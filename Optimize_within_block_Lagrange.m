function [theta_opt,obj_val,rtime,ncount] = Optimize_within_block_Lagrange(Y,W,Z,Q_sqr,tau,thetas,Lambda,b)
%Solves a MILP formulation of the Lagrangian problem for IV quantile
%regression
%
%INPUT:
%Y - nx1 vector 
%W - nxd vector of endogenous variables
%Z - nxl vector of instruments
%Q_sqr - lxd square root of weighting matrix
%tau - quantile level
%thetas - dxB matrix of pre-estimated thetas for each block
%Lambda - penalization parameter
%b - block index
%
%OUTPUT:
%theta_opt - the minimizer
%obj_value - objective function at the minimizer
%rtime - running time in seconds
%ncount - number of iterations

B = size(thetas,2); %number of blocks
m = size(Z,1)/B; %number of observations per block
%TODO: handle nonequal blocks
d = size(Q_sqr,2); %number of endogenous vars
dB = d*B;
ZbQ = Z((b-1)*m+1:b*m,:)*Q_sqr; %Z*sqrt(Q)
Wb = W((b-1)*m+1:b*m,:);
Yb = Y((b-1)*m+1:b*m,:);
M = 100*ones(m,1); %TODO: one M is enough
eps = 0.1;

sum_sZQ_alt = 0;
for b_alt = 1:B 
    % skip our fixed block b:
    if b_alt == b
        continue;
    end
    idx = (b_alt-1)*m+1:b_alt*m;
    s_b_alt = (Y(idx,:)-W(idx,:)*thetas(:,b_alt)<=0)-tau;
    sum_sZQ_alt = sum_sZQ_alt + s_b_alt'*Z(idx,:)*Q_sqr;
end
sum_sZQ_alt = sum_sZQ_alt';

avg_other_thetas = (sum(thetas,2)-thetas(:,b))/B;


%OPTIMIZATION PROBLEM:
%OBJ*vars -> min s.t. A*vars < RHS
%where OBJ, A, RHS are constructed below

% VARIABLE LAYOUT:
% q   theta_b   e (binary)    t
% 1x1 1xd       1xm           1x(B*d)

%INEQUALITIES LAYOUT:
% -q - (ZQ)'e < -tau*(ZQ)'iota + sum_sZQ_alt
% -q + (ZQ)'e < tau*(ZQ)'iota - sum_sZQ_alt
%  W_b*e - (M+eps)*e < Y_b
% -W_b*e + M*e       < Y_b + M
% -(1-1/B)*theta_b + t_{j,b} < avg_other_thetas
%  (1-1/B)*theta_b + t_{j,b} < -avg_other_thetas
% -t < 0
%  (here loop over blocks b'=1,...,B except for the b-th block)
%  (1/B)*theta_b + t_{j,b'} < -theta_{b'} + avg_other_thetas
% -(1/B)*theta_b + t_{j,b'} <  theta_{b'} - avg_other_thetas


OBJ = [1, zeros(1,d+m), Lambda*ones(1,dB)];

n_ineq = (3+3*B)*d+2*m;
n_vars = 1+d+m+dB;
A = zeros(n_ineq, n_vars); %linear inequality matrix
RHS = zeros(n_ineq, 1); %right hand side vector

ineq_idxs = 1:d;
A(ineq_idxs,:) = [-ones(d,1),zeros(d,d),-ZbQ',zeros(d,dB)];
tau_iZQ = tau*ZbQ'*ones(m,1); %dx1
RHS(ineq_idxs) = -tau_iZQ + sum_sZQ_alt;

ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
A(ineq_idxs ,:) = [-ones(d,1),zeros(d,d),ZbQ',zeros(d,dB)];
RHS(ineq_idxs) = tau_iZQ - sum_sZQ_alt;

ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+m;
A(ineq_idxs,:) = [zeros(m,1),Wb,-diag(M+eps),zeros(m,dB)];
RHS(ineq_idxs) = Yb;

ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+m;
A(ineq_idxs,:) = [zeros(m,1),-Wb,diag(M),zeros(m,dB)];
RHS(ineq_idxs) = -Yb+M;

ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
t_mat = zeros(d,dB);
t_mat(:,(b-1)*d+1:b*d) = eye(d);
A(ineq_idxs,:) = [zeros(d,1),-(1-1/B)*eye(d),zeros(d,m),-t_mat];
RHS(ineq_idxs) = avg_other_thetas;

ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
A(ineq_idxs,:) = [zeros(d,1),(1-1/B)*eye(d),zeros(d,m),-t_mat];
RHS(ineq_idxs) = -avg_other_thetas;

ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+dB;
A(ineq_idxs, d+m+2:d+m+1+dB) = -eye(dB); %this is -t_{j,b} < 0
% RHS(ineq_idxs) is zero

%For last set of inequalities,
%loop over blocks not equal to b;
%to each such block there correspond 2*d inequalities
for b_alt = 1:B

    if b_alt == b
        continue;
    end
    
    t_mat_alt = zeros(d,dB);
    t_mat_alt(:,(b_alt-1)*d+1:b_alt*d) = eye(d);
    ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
    A(ineq_idxs,:) = [zeros(d,1),1/B*eye(d),zeros(d,m),-t_mat_alt];
    RHS(ineq_idxs) = -thetas(:,b_alt) + avg_other_thetas;
    
    ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
    A(ineq_idxs,:) = [zeros(d,1),-1/B*eye(d),zeros(d,m),-t_mat_alt];
    RHS(ineq_idxs) = thetas(:,b_alt) - avg_other_thetas;

end


%OPTIMIZATION SET UP:
model.A = sparse(A);
model.obj= OBJ;
model.rhs = RHS;
model.sense = '<'; %only {=,<,>} allowed
model.modelsense = 'min';
% 'B' : int code 66
% 'C' : int code 67
model.vtype = char([67*ones(1,d+1) 66*ones(1,m) 67*ones(1,dB)]); 

params.outputflag = 0; 

try
      
    result = gurobi(model, params);
    theta_opt = result.x(2:1+d);
    obj_val = result.objval;
    %gap = (obj_v-result.objbound);
    rtime = result.runtime;
    ncount = result.nodecount;
    %fprintf('Optimization returned status: %s\n', result.status);
    disp(result);
 
catch gurobiError
    fprintf(['Gurobi error ocurred:\n' gurobiError.message]);
end

end