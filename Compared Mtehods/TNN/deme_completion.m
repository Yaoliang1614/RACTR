clear;
clc;
load Urban_0.1SR_0.5NR.mat

%% TNN_TC
[Xhat] = TNN_TC(Xmiss, sampling_mask);
Xhat = max(Xhat,0);
Xhat = min(Xhat,1);
[psnrs,ssims] = MSIQA(Xhat*255, X*255);
PSNR=mean(psnrs);
SSIM=mean(ssims);
fprintf("\t TNN-TC PSNR=%.3f,\t SSIM=%.3f \n",PSNR,SSIM);

