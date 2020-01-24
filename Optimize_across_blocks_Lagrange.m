function theta_final = Optimize_across_blocks_Lagrange(Y,W,Z,Q_sqr,tau,theta_ini,Lambda,B,tol)
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

thetas = repmat(theta_ini,1,B);
obj_prev = Inf;
obj_step = Inf;
b = 1;
while obj_step > tol
    b_new = ceil(B*rand()); %random element of {1,...,B}
    while b_new == b
        b_new = ceil(B*rand());
    end
    b = b_new;
    [theta, obj_curr] = Optimize_within_block_Lagrange(Y,W,Z,Q_sqr,tau,thetas,Lambda,b);
    obj_step = abs(obj_curr - obj_prev);
    obj_prev = obj_curr;
    thetas(:,b) = theta;
end
theta_final = mean(thetas,2);
end
