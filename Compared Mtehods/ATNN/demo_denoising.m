clear all;
clc;
 
load pavia_0.6noise.mat
 
maxP = max(abs(X(:)));
[n1,n2,n3] = size(X);
 

opts.DEBUG = 0;
lambdas=[1]/sqrt(max(n1,n2));
rks= [5,10,15 20,25,30];
temp=0;
for ii=1:length(lambdas)
    opts.lambda=  lambdas(ii);
    for jj=1:length(rks)
        opts.r = rks(jj);
        tic;
        Xhat = ATNN_RPCA(Xn,opts);
        toc;
        Xhat = max(Xhat,0);
        Xhat = min(Xhat,1);
         [psnrs,ssims] = MSIQA(Xhat*255, X*255);
        PSNR=mean(psnrs);
        SSIM=mean(ssims);
        if(PSNR>temp)
            temp=PSNR;
            bestXhat=Xhat;
        end
        fprintf("\tlambda=%.4f, rk=%.5f/(sqrt(n1*n2)),\t PSNR=%.3f,\t SSIM=%.3f, psnrMax=%.3f\n", opts.lambda, opts.r,PSNR,SSIM,temp);
    end
end
 