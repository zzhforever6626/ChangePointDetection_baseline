function iac = getAsymIAC(pro_len)
mpi = zeros(pro_len, 1);
for i = 0:pro_len-1
    mpi(i+1) = randi([i+1, pro_len], 1, 1);
end

nnmark = zeros(pro_len, 1);
for i = 1:pro_len
    j = mpi(i);
    nnmark(min(i, j)) = nnmark(min(i, j)) + 1;
    nnmark(max(i, j)) = nnmark(max(i, j)) - 1;
end

n_arc = 0;
iac = zeros(pro_len, 1);
for i = 1:pro_len
    n_arc = n_arc + nnmark(i);
    iac(i) = n_arc;
end