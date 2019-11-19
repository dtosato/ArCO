% *************************** Disclaimer ************************************** 
% This program is distributed in the hope that it will be useful, but 
% WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY 
% or FITNESS FOR A PARTICULAR PURPOSE. Feel free to use this code for academic 
% purposes.  Please use the citation provided below.
% 
% The test part of this code takes 0.05 seconds per image on a Intel Xeon(R) 
% CPU E5440 @2.83 GHz 8 GB RAM on Matlab 2009b. The most computationally expensive part of 
% this code is the learning phase Y_train_light.m 
% *****************************Copyright*********************************
%   Copyright 2010 Diego Tosato
%   $Revision: 4b$  $Date: 2010/02/25$
% ***********************************************************************
% Y_BuildF Feature extractor.
% input: 
%   data         = input image
%   coords       = ROI layout
%   d            = number of features
% output:
%   F           = feature multidimensional array
%**************************************************************************
function [F] = Y_BuildF(data,coords,d) 

% feature calcualtion 
[size_arg] = size(data);
if length(size_arg)<2
    error('Data stored has to be W*H*num_images\n');
end
si = size(data);
F = zeros(si(1),si(2),d); % feature multidimensional array

% to ldg
if length(si) > 2
    data_rgb    =   data;
    data        =   rgb2gray(data);
end

[x,y]     =   meshgrid((coords(1):coords(3))',(coords(2):coords(4))'); 

img               =   double(data(:,:)) ;

% spatial layout
F(:,:,1)          =   double(x)'; 
F(:,:,2)          =   double(y)';

% gabor filters
[~,gabout] = gaborfilter1(img,2,4,16,0); 
F(:,:,3)          =  gabout;
[~,gabout] = gaborfilter1(img,2,4,16,pi/3); 
F(:,:,4)          =  gabout;
[~,gabout] = gaborfilter1(img,2,4,16,pi/6); 
F(:,:,5)          =  gabout;
[~,gabout] = gaborfilter1(img,2,4,16,pi*4/3); 
F(:,:,6)          =  gabout;

% color
F(:,:,7)          =   data_rgb(:,:,1); %R
F(:,:,8)          =   data_rgb(:,:,2); %G
F(:,:,9)         =   data_rgb(:,:,3); %B

% gradient orientation
[gh,gv]            =   gradient(img);
F(:,:,10)          =   gv;
F(:,:,11)          =   gh;
F(:,:,12)          =   atan2(gv,gh);
end 
    