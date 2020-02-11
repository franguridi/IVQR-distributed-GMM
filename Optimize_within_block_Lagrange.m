function out = Optimize_within_block_Lagrange(Y,W,Z,Q_sqr,tau,thetas,es,x_prev,Lambda,psi,b,M,eps,moment_norm)
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
M = M*ones(m,1);

sum_sZQ_alt = 0;
for b_alt = 1:B 
    % skip our fixed block b:
    if b_alt == b
        continue;
    end
    idx = (b_alt-1)*m+1:b_alt*m;
    %s_b_alt = ((Y(idx)-W(idx,:)*thetas(:,b_alt))<=0)-tau;
    %alternative way of computing s_b_alt (when previous e's are given):
    s_b_alt = es(:,b_alt)-tau;
    sum_sZQ_alt = sum_sZQ_alt + s_b_alt'*Z(idx,:)*Q_sqr;
end
sum_sZQ_alt = sum_sZQ_alt';
psi_times_sum_sZQ_alt = psi*sum_sZQ_alt;

avg_other_thetas = (sum(thetas,2)-thetas(:,b))/B;


%OPTIMIZATION PROBLEM:
%OBJ*vars -> min s.t. A*vars < RHS
%where OBJ, A, RHS are constructed below
%depending on moment_norm input arg

switch moment_norm
    case 'sup'
        
        error('NOT IMPLEMENTED');
        
        %TODO: case with separate variable for theta_avg

        % VARIABLE LAYOUT:
        % theta_avg q   theta_b   e (binary)    t      
        % 1xd       1x1 1xd       1xm           1x(B*d)

        %INEQUALITIES LAYOUT:
        % -q - (ZQ)'e < -tau*(ZQ)'iota + sum_sZQ_alt        (d ineq)
        % -q + (ZQ)'e < tau*(ZQ)'iota - sum_sZQ_alt         (d ineq)
        %  W_b*theta_b - (M+eps)*e < Y_b                    (m ineq)
        % -W_b*theta_b + M*e       < -Y_b + M               (m ineq)
        % -(1-1/B)*theta_b - t_{j,b} < -avg_other_thetas    (d ineq)
        %  (1-1/B)*theta_b - t_{j,b} < +avg_other_thetas    (d ineq)
        % -t < 0                                            (d*B ineq)
        %  (here loop over blocks b'=1,...,B except for the b-th block)
        %  (1/B)*theta_b - t_{j,b'} <  theta_{b'} - avg_other_thetas
        %  (d*(B-1) ineq)
        % -(1/B)*theta_b - t_{j,b'} < -theta_{b'} + avg_other_thetas
        %  (d*(B-1) ineq)


        %objective function = q + Lambda*sum(t)
        OBJ = [1, zeros(1,d+m), Lambda*ones(1,dB)];

        n_ineq = (2+3*B)*d+2*m;
        n_vars = 1+d+m+dB;
        A = zeros(n_ineq, n_vars); %linear inequality matrix
        RHS = zeros(n_ineq, 1); %right hand side vector

        ineq_idxs = 1:d;
        A(ineq_idxs,:) = [-ones(d,1),zeros(d,d),-ZbQ',zeros(d,dB)];
        tau_iZQ = tau*ZbQ'*ones(m,1); %dx1
        RHS(ineq_idxs) = -tau_iZQ + psi_times_sum_sZQ_alt;

        ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
        A(ineq_idxs ,:) = [-ones(d,1),zeros(d,d),ZbQ',zeros(d,dB)];
        RHS(ineq_idxs) = tau_iZQ - psi_times_sum_sZQ_alt;

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
        RHS(ineq_idxs) = -avg_other_thetas;

        ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
        A(ineq_idxs,:) = [zeros(d,1),(1-1/B)*eye(d),zeros(d,m),-t_mat];
        RHS(ineq_idxs) = avg_other_thetas;

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
            RHS(ineq_idxs) = thetas(:,b_alt) - avg_other_thetas;

            ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
            A(ineq_idxs,:) = [zeros(d,1),-1/B*eye(d),zeros(d,m),-t_mat_alt];
            RHS(ineq_idxs) = -thetas(:,b_alt) + avg_other_thetas;

        end


        %OPTIMIZATION SET UP:
        model.A = sparse(A);
        model.obj= OBJ;
        model.rhs = RHS;
        model.sense = '<'; %only {=,<,>} allowed
        model.modelsense = 'min';
        model.start = x_prev;
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
            disp(num2str(theta_opt));
            disp(['L = ' num2str(result.x(1))]);
            obj_val_term_1 = result.x(1);
            if ~strcmp(result.status,'OPTIMAL')
                error('non optimal');
            end

        catch gurobiError
            fprintf(['Gurobi error ocurred:\n' gurobiError.message]);
        end
    
        
    %===================================================================================
    %===================================================================================
    
    case 'L1'
        
    % INEQUALITIES WITH THETA_AVG AS A SEPARATE VARIABLE ===============================
    %
    
    % VARIABLE LAYOUT:
    % theta_avg   q     theta_b   e (binary)    t
    % 1xd         1xd   1xd       1xm           1x(B*d)

    %INEQUALITIES LAYOUT:
    % -q (vector) - (ZQ)'e < -tau*(ZQ)'iota + sum_sZQ_alt
    % -q (vector) + (ZQ)'e < tau*(ZQ)'iota - sum_sZQ_alt
    %  W_b*theta_b - (M+eps)*e < Y_b
    % -W_b*theta_b + M*e       < -Y_b + M
    % -(1-1/B)*theta_b - t_{j,b} < -avg_other_thetas
    %  (1-1/B)*theta_b - t_{j,b} < +avg_other_thetas
    % -t < 0
    %  (here loop over blocks b'=1,...,B except for the b-th block)
    %  (1/B)*theta_b - t_{j,b'} <  theta_{b'} - avg_other_thetas
    % -(1/B)*theta_b - t_{j,b'} < -theta_{b'} + avg_other_thetas


    %objective function = sum(q) + Lambda*m*sum(t) (note normalization)
    OBJ = [zeros(1,d), ones(1,d), zeros(1,d+m), Lambda*m*ones(1,dB)];

    n_ineq = (3+3*B)*d+2*m;
    n_vars = d+d+d+m+dB;
    A = zeros(n_ineq, n_vars); %linear inequality matrix
    RHS = zeros(n_ineq, 1); %right hand side vector
    
    % -q (vector) - (ZQ)'e < -tau*(ZQ)'iota + sum_sZQ_alt
    ineq_idxs = 1:d;
    A(ineq_idxs,:) = [zeros(d),-eye(d),zeros(d),-ZbQ',zeros(d,dB)];
    tau_iZQ = tau*ZbQ'*ones(m,1); %dx1
    RHS(ineq_idxs) = -tau_iZQ + psi_times_sum_sZQ_alt;

    % -q (vector) + (ZQ)'e < tau*(ZQ)'iota - sum_sZQ_alt
    ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
    A(ineq_idxs ,:) = [zeros(d),-eye(d),zeros(d),ZbQ',zeros(d,dB)];
    RHS(ineq_idxs) = tau_iZQ - psi_times_sum_sZQ_alt;

    %  W_b*theta_b - (M+eps)*e < Y_b
    ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+m;
    A(ineq_idxs,:) = [zeros(m,2*d),Wb,-diag(M+eps),zeros(m,dB)];
    RHS(ineq_idxs) = Yb;

    % -W_b*theta_b + M*e       < -Y_b + M
    ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+m;
    A(ineq_idxs,:) = [zeros(m,2*d),-Wb,diag(M),zeros(m,dB)];
    RHS(ineq_idxs) = -Yb+M;
    
    % -t < 0
    ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+dB;
    A(ineq_idxs, 3*d+m+1:end) = -eye(dB);
    % RHS(ineq_idxs) is zero
    
    % inequalities with theta_avg for block b:
    % ----------------------------------------
    % theta_avg - theta_b - t_b < 0
    ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
    t_mat = zeros(d,dB);
    t_mat(:,(b-1)*d+1:b*d) = eye(d);
    A(ineq_idxs,:) = [eye(d),zeros(d),-eye(d),zeros(d,m),-t_mat];
    % RHS(ineq_idxs) is zero
    
    % -theta_avg + theta_b - t_b < 0
    ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
    A(ineq_idxs,:) = [-eye(d),zeros(d),eye(d),zeros(d,m),-t_mat];
    % RHS(ineq_idxs) is zero
    
    % inequalities defining theta_avg:
    % ----------------------------------------
    % theta_avg - theta_b/B < avg_other_thetas
    ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
    A(ineq_idxs,:) = [eye(d),zeros(d),-eye(d)/B,zeros(d,m+dB)];
    RHS(ineq_idxs) = avg_other_thetas;
    
    % -theta_avg + theta_b/B < -avg_other_thetas
    ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
    A(ineq_idxs,:) = [-eye(d),zeros(d),eye(d)/B,zeros(d,m+dB)];
    RHS(ineq_idxs) = -avg_other_thetas;
    
    % inequalities with theta_avg for blocks OTHER THAN b:
    %to each such block there correspond 2*d inequalities
    %
    for b_alt = 1:B

        if b_alt == b
            continue;
        end
        
        t_mat_alt = zeros(d,dB);
        t_mat_alt(:,(b_alt-1)*d+1:b_alt*d) = eye(d);
        % theta_avg - t_{b'} < theta_{b'}
        ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
        A(ineq_idxs,:) = [eye(d),zeros(d,2*d+m),-t_mat_alt];
        RHS(ineq_idxs) = thetas(:,b_alt);

        % -theta_avg - t_{b'} < -theta_{b'}
        ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
        A(ineq_idxs,:) = [-eye(d),zeros(d,2*d+m),-t_mat_alt];
        RHS(ineq_idxs) = -thetas(:,b_alt);
        
    end
    %}
    
    %OPTIMIZATION SET UP ==================================================
    model.A = sparse(A);
    model.obj= OBJ;
    model.rhs = RHS;
    model.sense = '<'; %only {=,<,>} allowed
    model.modelsense = 'min';
    if ~isnan(x_prev)
        %e_start = (Yb - Wb*thetas(:,b)) <= 0;
        e_start = es(:,b);
        model.start = [x_prev(1:2*d);thetas(:,b);e_start;x_prev(3*d+m+1:end)];
        
        disp(['Obj function at starting point = ' num2str(model.obj*model.start)]);
        constr_resid = model.A*model.start - model.rhs;
        constr_violated =  sum(constr_resid > 0);
        %{
        if constr_violated~=0
            disp('Starting point feasible? !!! NO !!!');
            disp(['Number of constraints violated = ' num2str(constr_violated)]);
            disp(['L_1 constraint violation = ' num2str(sum(constr_resid(constr_resid>0)))]);
            disp(['L_inf constraint violation = ' num2str(max(constr_resid(constr_resid>0)))]);
            disp('Indicies of violated constraints = ');
            disp(num2str(find(constr_resid>0)));
            disp('Residuals of violated constraints = ');
            disp(constr_resid(constr_resid>0));
            disp('First d inequalities abs(LHS)');
            lhs = 0;
            for k=1:B
                idx = (k-1)*m+1:k*m;
                lhs = lhs + (es(:,k)-tau)'*Z(idx,:)*Q_sqr;
            end
            disp(abs(lhs));
            disp(['Number of viol ' num2str(sum(abs(lhs)'>model.start(d+1:2*d)+0.00001))]);
            
            disp('Q');
            disp(x_prev(d+1:2*d));
        else
            disp('Starting point feasible? Yes');
        end
        %}
        % check feasibility:
        %disp('The following contraints are violated at model.start');
        %
        %disp(find(discrepancy>0.1));
        %disp(discrepancy(discrepancy>0.1));
        
    end
    % 'B' : int code 66
    % 'C' : int code 67
    model.vtype = char([67*ones(1,3*d) 66*ones(1,m) 67*ones(1,dB)]); 

    params.outputflag = 0; 
    %}
    
    
    % INEQUALITIES WITH THETA_AVG PLUGGED IN ========================================
    %{
    
    %objective function = sum(q) + Lambda*sum(t)
    OBJ = [ones(1,d), zeros(1,d+m), Lambda*ones(1,dB)];
    
    n_ineq = (2+3*B)*d+2*m;
    n_vars = d+d+m+dB;
    A = zeros(n_ineq, n_vars); %linear inequality matrix
    RHS = zeros(n_ineq, 1); %right hand side vector
    
    % -q (vector) - (ZQ)'e < -tau*(ZQ)'iota + sum_sZQ_alt
    ineq_idxs = 1:d;
    A(ineq_idxs,:) = [-eye(d),zeros(d),-ZbQ',zeros(d,dB)];
    tau_iZQ = tau*ZbQ'*ones(m,1); %dx1
    RHS(ineq_idxs) = -tau_iZQ + psi_times_sum_sZQ_alt;

    % -q (vector) + (ZQ)'e < tau*(ZQ)'iota - sum_sZQ_alt
    ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
    A(ineq_idxs ,:) = [-eye(d),zeros(d),ZbQ',zeros(d,dB)];
    RHS(ineq_idxs) = tau_iZQ - psi_times_sum_sZQ_alt;

    %  W_b*theta_b - (M+eps)*e < Y_b
    ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+m;
    A(ineq_idxs,:) = [zeros(m,d),Wb,-diag(M+eps),zeros(m,dB)];
    RHS(ineq_idxs) = Yb;

    % -W_b*theta_b + M*e       < -Y_b + M
    ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+m;
    A(ineq_idxs,:) = [zeros(m,d),-Wb,diag(M),zeros(m,dB)];
    RHS(ineq_idxs) = -Yb+M;
    
    
    % -(1-1/B)*theta_b - t_{j,b} < -avg_other_thetas
    ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
    t_mat = zeros(d,dB);
    t_mat(:,(b-1)*d+1:b*d) = eye(d);
    A(ineq_idxs,:) = [zeros(d,d),-(1-1/B)*eye(d),zeros(d,m),-t_mat];
    RHS(ineq_idxs) = -avg_other_thetas;

    %  (1-1/B)*theta_b - t_{j,b} < +avg_other_thetas
    ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
    A(ineq_idxs,:) = [zeros(d,d),(1-1/B)*eye(d),zeros(d,m),-t_mat];
    RHS(ineq_idxs) = avg_other_thetas;

    % -t < 0
    ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+dB;
    A(ineq_idxs, 2*d+m+1:end) = -eye(dB);
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
        A(ineq_idxs,:) = [zeros(d,d),1/B*eye(d),zeros(d,m),-t_mat_alt];
        RHS(ineq_idxs) = thetas(:,b_alt) - avg_other_thetas;

        ineq_idxs = ineq_idxs(end)+1:ineq_idxs(end)+d;
        A(ineq_idxs,:) = [zeros(d,d),-1/B*eye(d),zeros(d,m),-t_mat_alt];
        RHS(ineq_idxs) = -thetas(:,b_alt) + avg_other_thetas;

    end
    
    %OPTIMIZATION SET UP ==================================================
    model.A = sparse(A);
    model.obj= OBJ;
    model.rhs = RHS;
    model.sense = '<'; %only {=,<,>} allowed
    model.modelsense = 'min';
    % 'B' : int code 66
    % 'C' : int code 67
    model.vtype = char([67*ones(1,2*d) 66*ones(1,m) 67*ones(1,dB)]); 

    params.outputflag = 0; 
    %}
    
    % ===============================================================================
    try

        result = gurobi(model, params);
        %disp(result);
        if ~strcmp(result.status,'OPTIMAL')
            error(['Block optimization returned status ' results.status ', not OPTIMAL']);
        end
        out.x = result.x;
        
        if length(result.x) == 2*d + m + dB %case: plugged in theta_avg
            out.theta_opt = result.x(d+1:2*d);
            out.e = result.x(2*d+1:2*d+m);
            out.obj_val_term_1 = sum(result.x(1:d));
        elseif length(result.x) == 3*d + m + dB %case: theta_avg as separate var
            out.theta_opt = result.x(2*d+1:3*d);
            out.e = result.x(3*d+1:3*d+m);
            out.obj_val_term_1 = sum(result.x(d+1:2*d));
        else
            error('Output vector of unhandled dimension');
        end
        out.obj_val = result.objval;
        
        %gap = (obj_v-result.objbound);
        out.rtime = result.runtime;
        out.ncount = result.nodecount;
        disp('Optimal theta = ');
        disp(out.theta_opt');
        disp(['Obj value at optimum = ' num2str(out.obj_val)]);
        disp(['Sum(q) (term 1 in obj fun) at optimum = ' num2str(out.obj_val_term_1)]);
        
        %YminWtheta = Yb-Wb*out.theta_opt;
        %disp( YminWtheta((YminWtheta<=0) ~= out.x(3*d+1:3*d+m)));
        
    catch gurobiError
        fprintf(['Gurobi error ocurred:\n' gurobiError.message]);
    end


end