clear
clc
load Urban_0.1SR_0.5NR.mat
%% ATNN_TC
rks= [5 10 15 20 25 30];
temp=0;

for jj=1:length(rks)
    opts.r = rks(jj);
    tic;
    Xhat = ATNN_TC(Xmiss, sampling_mask,  opts.r);
    toc;
    Xhat = max(Xhat,0);
    Xhat = min(Xhat,1);
    [psnrs,ssims] = MSIQA(Xhat*255, X*255);
    PSNR=mean(psnrs);
    SSIM=mean(ssims);
    if(PSNR>temp)
        temp=PSNR;
        besetXhat=Xhat;
    end
    fprintf("\t r=%.5f/(sqrt(n1*n2)),\t PSNR=%.3f,\t SSIM=%.3f, psnrMax=%.3f\n", opts.r,PSNR,SSIM,temp);
end



