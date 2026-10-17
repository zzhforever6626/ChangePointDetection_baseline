function arc = getArcCurve(proIdx, t_len)
%%
nnmark = zeros(t_len, 1);
for i = 1:length(proIdx)
    j = proIdx(i);
    nnmark(min(i, j)) = nnmark(min(i, j)) + 1;
    nnmark(max(i, j)) = nnmark(max(i, j)) - 1;
end

n_arc = 0;
arc = zeros(t_len, 1);
for i = 1:t_len
    n_arc = n_arc + nnmark(i);
    arc(i) = n_arc;
end