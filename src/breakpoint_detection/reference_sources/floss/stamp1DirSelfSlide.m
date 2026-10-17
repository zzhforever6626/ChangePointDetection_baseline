% Compute the 1 directional self-similarity join of a given time series A
% Chin-Chia Michael Yeh 01/11/2017
%
% [matrixProfile, profileIndex] = stamp1DirSelfSlide(A, subLen)
% Output:
%     matrixProfile: matrix porfile of the self-join (vector)
%     profileIndex: matrix porfile index of the self-join (vector)
% Input:
%     A: input time series (vector)
%     subLen: interested subsequence length (scalar)
%
% Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum, Yifei Ding, Hoang Anh Dau,
% Diego Furtado Silva, Abdullah Mueen, and Eamonn Keogh, "Matrix Profile I: All Pairs Similarity
% Joins for Time Series," ICDM 2016, http://www.cs.ucr.edu/~eamonn/MatrixProfile.html
%

function [matrixProfile, profileIndex] = stamp1DirSelfSlide(A, subLen, ...
    matrixProfile, profileIndex)
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

%% update
matrixProfile(1:end-1) = matrixProfile(2:end);
profileIndex(1:end-1) = profileIndex(2:end);
matrixProfile(end) = inf;
profileIndex(end) = inf;
profileIndex = profileIndex - 1;
distanceProfile = fastfindNN(A, A(end-subLen+1:end));
excZoneStart = length(A)-subLen+1-excZone;
distanceProfile(excZoneStart:end) = inf;
updatePos = distanceProfile < matrixProfile;
profileIndex(updatePos) = length(A) - subLen + 1;
matrixProfile(updatePos) = distanceProfile(updatePos);

%% The following function is modified from the code provided in the following URL
%  http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
function dist = fastfindNN(x,y)
%x is the data, y is the query

n = length(x);
y = (y-mean(y))./std(y,1);                      %Normalize the query
m = length(y);
x(n+1:2*n) = 0;
y = y(end:-1:1);                                %Reverse the query
y(m+1:2*n) = 0;

%The main trick of getting dot products in O(n log n) time
X = fft(x);
Y = fft(y);
Z = X.*Y;
z = ifft(Z);

%compute y stats -- O(n)
sumy = sum(y);
sumy2 = sum(y.^2);

%compute x stats -- O(n)
cum_sumx = cumsum(x);
cum_sumx2 =  cumsum(x.^2);
sumx2 = cum_sumx2(m:n)-[0;cum_sumx2(1:n-m)];
sumx = cum_sumx(m:n)-[0;cum_sumx(1:n-m)];
meanx = sumx./m;
sigmax2 = (sumx2./m)-(meanx.^2);
sigmax = sqrt(sigmax2);


%computing the distances -- O(n) time
dist = (sumx2 - 2*sumx.*meanx + m*(meanx.^2))./sigmax2 - 2*(z(m:n) - sumy.*meanx)./sigmax + sumy2;
%dist = 1-dist./(2*m);
dist = sqrt(dist);
dist = real(dist);