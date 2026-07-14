 
clc;
clear;
load pavia_0.6noise.mat

opts = [];
opts.rho = 1.1;
 
opts.directions = [1 2 3]; % consider the lowrankness and smoothness both along the spatial and spectral directions
tic
Xhat = TCTV_TRPCA(Xn, opts);
toc;
Xhat = max(Xhat,0);
Xhat = min(Xhat,1);
[psnrs,ssims] = MSIQA(Xhat*255, X*255);
PSNR=mean(psnrs);
SSIM=mean(ssims);
[PSNR, SSIM]

