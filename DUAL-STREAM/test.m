matlabpool local 50
function [maxrad] = test(x,y,onset_save_path,apex_save_path,flow_save_path,afflow_pic_path)
% img1 = double(img1);
% img2 = double(img2);
% x = double(x);
% y = double(y);

addpath('mex');

% path_xy = char(path_xy);
img1 = char(onset_save_path);
img2 = char(apex_save_path);
path_flow = char(flow_save_path);
path_afflow = char(afflow_pic_path);
% load(path_xy);


% load the two frames
im1 = im2double(imread(img1));
im2 = im2double(imread(img2));


% set optical flow parameters (see Coarse2FineTwoFrames.m for the definition of the parameters)
alpha = 0.012;
ratio = 0.75;
minWidth = 20;
nOuterFPIterations = 7;
nInnerFPIterations = 1;
nSORIterations = 30;

para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];

% this is the core part of calling the mexed dll file for computing optical flow
% it also returns the time that is needed for two-frame estimation
tic;
[vx,vy,warpI2] = Coarse2FineTwoFrames(im1,im2,para);
toc




% output gif
clear volume;
volume(:,:,:,1) = im1;
volume(:,:,:,2) = im2;
if exist('output','dir')~=7
    mkdir('output');
end
frame2gif(volume,fullfile('output','_input.gif'));
volume(:,:,:,2) = warpI2;
frame2gif(volume,fullfile('output','_warp.gif'));


% visualize flow field
clear flow;
flow(:,:,1) = vx;
flow(:,:,2) = vy;
if x and y
[u, v] = affine(flow(:,:,1), flow(:,:,2), x ,y);
afflow(:,:,1) = u;
afflow(:,:,2) = v;

imafflow = flowToColor(afflow);
imwrite(imafflow,fullfile(path_afflow),'quality',100);

maxrad = maxflow(im1*255.,im2*255.,x,y,1);
end
imflow = flowToColor(flow);
imwrite(imflow,fullfile(path_flow),'quality',100);

end
