% Compute the 1 directional self-similarity join of a given time series A

%
% [matrixProfile, profileIndex] = stamp1DirSelf(A, subLen)
% Output:
%     matrixProfile: matrix porfile of the self-join (vector)
%     profileIndex: matrix porfile index of the self-join (vector)
% Input:
%     A: input time series (vector)
%     subLen: interested subsequence length (scalar)
%

function [matrixProfile, profileIndex] = stamp1DirSelf(A, subLen)
%% set trivial match exclusion zone
excZone = round(subLen/2);

%% check input
if subLen > length(A)/2
    error('Error: Time series is too short relative to desired subsequence length');
end
if subLen < 4
    error('Error: Subsequence length must be at least 4');
end
if length(A) == size(A, 2)
   A = A'; 
end

%% initialization
proLen = length(A) - subLen + 1;
matrixProfile = inf(proLen, 1);
profileIndex = inf(proLen, 1);
[X, n, sumx2, sumx, meanx, sigmax2, sigmax] = ...
    fastfindNNPre(A, subLen);

%% compute the matrix profile
pickedIdx = proLen:-1:1;
for i = 1:proLen
    % compute the distance profile
    idx = pickedIdx(i);
    subsequence = A(idx:idx+subLen-1);
    distanceProfile = fastfindNN(X, subsequence, n, subLen, ...
        sumx2, sumx, meanx, sigmax2, sigmax);
    distanceProfile = real(distanceProfile);
    
    % apply exclusion zone
    excZoneStart = max(1, idx-excZone);
    distanceProfile(excZoneStart:proLen) = inf;

    % figure out and store the neareest neighbor
    updatePos = distanceProfile < matrixProfile;
    profileIndex(updatePos) = idx;
    matrixProfile(updatePos) = distanceProfile(updatePos);
end
matrixProfile = sqrt(matrixProfile);
matrixProfile = real(matrixProfile);

%% The following two functions are modified from the code provided in the following URL
%  http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
function [X, n, sumx2, sumx, meanx, sigmax2, sigmax] = fastfindNNPre(x, m)
n = length(x);
x(n+1:2*n) = 0;
X = fft(x);
cum_sumx = cumsum(x);
cum_sumx2 =  cumsum(x.^2);
sumx2 = cum_sumx2(m:n)-[0;cum_sumx2(1:n-m)];
sumx = cum_sumx(m:n)-[0;cum_sumx(1:n-m)];
meanx = sumx./m;
sigmax2 = (sumx2./m)-(meanx.^2);
sigmax = sqrt(sigmax2);

function dist = fastfindNN(X, y, n, m, sumx2, sumx, meanx, sigmax2, sigmax)
%x is the data, y is the query
y = (y-mean(y))./std(y,1);                      %Normalize the query
y = y(end:-1:1);                                %Reverse the query
y(m+1:2*n) = 0;

%The main trick of getting dot products in O(n log n) time
Y = fft(y);
Z = X.*Y;
z = ifft(Z);

%compute y stats -- O(n)
sumy = sum(y);
sumy2 = sum(y.^2);

%computing the distances -- O(n) time
dist = (sumx2 - 2*sumx.*meanx + m*(meanx.^2))./sigmax2 - 2*(z(m:n) - sumy.*meanx)./sigmax + sumy2;
%dist = 1-dist./(2*m);