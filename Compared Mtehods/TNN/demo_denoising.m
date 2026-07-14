
clear
clc;
%% impluse denoising or mixed-noise deniong
load  pavia_0.6noise.mat
maxP =  max(abs(X(:)));
opts.mu = 1e-4;
opts.tol = 1e-5;
opts.rho = 1.1;
opts.max_iter = 300;
opts.DEBUG = 0;

[n1,n2,n3] = size(Xn);
lambdas =  [0.5,1,1.5]/sqrt(max(n1,n2)*n3);
temp=0;
for i=1:length(lambdas)
    lambda=lambdas(i);
    tic;
    [Xhat,E,iter] = trpca_tnn(Xn,lambda,opts);
    toc;
    Xhat = max(Xhat,0);
    Xhat = min(Xhat,maxP);
    [psnrs,ssims] = MSIQA(Xhat*255, X*255);
    psnr=mean(psnrs);
    ssim=mean(ssims);   
    if temp<psnr
        temp=psnr;
    end
    Maxpsnr=temp;
    [psnr ssim Maxpsnr]
end

