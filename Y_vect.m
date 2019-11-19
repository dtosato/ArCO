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
%**************************************************************************
%   Y_vect calculates the minimal representation of the points on Sym_d Riemaniann manifold,
%   as definied by [1], equations (19-20).
%   NB: in this case, the points are covariance matrices, so d(d+1)/2
%   independent coefficients are needed.
% Input:
%   X := projection point (covariance matrix) 
%   y : = projected covariance matrix 
%
% Output:
%   vect := calculates the minimal representation of y
%******************************References**********************************
%      [1]O. Tuzel, F. Porikli, P. Meer: Pedestrian Detection via
%      Classification on Riemannian Manifolds, Pattern Analysis and Machine
%      Intelligence, IEEE Transactions on 
%**************************************************************************         

function vect = Y_vect(X,y) 
d           =   size(X,1);
expneg      =   X^(-0.5);
inner       =   expneg * y * expneg;
diago       =   diag(diag(inner));  
inner       =   (inner - diago).*sqrt(2) + diago;
vect        =   [];

for i=0:d-1    
    vect = [vect;  inner(i+1:end,i+1)];
end