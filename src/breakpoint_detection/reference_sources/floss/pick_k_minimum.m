function idx = pick_k_minimum(curve, k, step)
curve_len = length(curve);
idx = zeros(k, 1);
for i = 1:k
    [~, idx(i)] =  min(curve);
    exu_st = max(1, idx(i)-step);
    exu_ed = min(curve_len, idx(i)+step);
    
    curve(exu_st:exu_ed) = inf;
end