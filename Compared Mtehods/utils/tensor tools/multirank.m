function mrank = multirank(X,tol)
% The tensor tubal rank of a 3 way tensor
%
% X     -    n1*n2*n3 tensor
% mrank -    tensor multi rank of X

X = fft(X,[],3);
[n1,n2,n3] = size(X);
s = [];
mrank = zeros(1,n3);

% i=1
s =  svd(X(:,:,1),'econ');
mrank(1) = sum(s > tol);
s=[];

% i=2,...,halfn3
halfn3 = round(n3/2);
for i = 2 : halfn3
    s = svd(X(:,:,i),'econ');
    mrank(i) = sum(s > tol);
    s=[];
end

% if n3 is even
if mod(n3,2) == 0
    i = halfn3+1;
    s = svd(X(:,:,i),'econ');
    mrank(i) = sum(s > tol);
    s=[];
end

for  i=round(n3/2)+1:n3
    mrank(i) = mrank(n3-i+2);
end


if nargin==1
   tol = max(n1,n2) * eps(max(s));
end

end
