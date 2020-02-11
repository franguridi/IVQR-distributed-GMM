function M = Determine_M(Y,W,theta_lb,theta_ub)

d = size(W,2);
Comb = zeros(2^d,d);
nxt_idx = 2;
for k=1:d
    idxs = nchoosek(1:d,k);
    n_idxs = nchoosek(d,k);
    for l=1:n_idxs
        Comb(nxt_idx,idxs(l,:)) = 1;
        nxt_idx = nxt_idx+1;
    end
end

Comb(Comb==0)=theta_lb;
Comb(Comb==1)=theta_ub;

M = -Inf;
for k=1:2^d
    val = max(abs(Y - W*Comb(k,:)'));
    if val > M
        M = val;
    end
end

   
end