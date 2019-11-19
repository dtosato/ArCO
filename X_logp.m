% ***************************Disclaimer************************************** 
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
%   X_logp is the logarithmic map associated to the riemaniann metric,
%   as defined by [1], formula (15).
%
% Input:
%   X := projection point (covariance matrix) 
%   Y : = covariance matrix to project
%
% Output:
%   logXY := projected covariance matrix (y)
%******************************References**********************************
%      [1]O. Tuzel, F. Porikli, P. Meer: Pedestrian Detection via
%      Classification on Riemannian Manifolds, Pattern Analysis and Machine
%      Intelligence, IEEE Transactions on 
%**************************************************************************
function logXY = X_logp(X,Y) 
exppos = X^(1/2);
expneg = X^(-1/2);
inner = expneg*Y*expneg;
[U,S] = eig(inner);
log_inner = U*diag(log(diag(S)))*U';
logXY = exppos * log_inner * exppos;
end
