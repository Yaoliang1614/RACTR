
clc;
clear;close all;

load Urban_0.1SR_0.5NR.mat
Obs=Xmiss;
Omega=logical(sampling_mask);
i=1;
opts = [];
opts.transform = 'DFT';
opts.directions = [1 2]; % for images, [1 2]; for HSI, [1 2 3]
tic
Xhat = TCTV_TC(Obs, Omega, opts);
Time(i) = toc;
%[PSNR(i), SSIM(i), FSIM(i)] = Img_QA(X,Xhat)

[psnrs,ssims] = MSIQA(Xhat*255, X*255);
PSNR=mean(psnrs);
SSIM=mean(ssims);
[PSNR, SSIM]

