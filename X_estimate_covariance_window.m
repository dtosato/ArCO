% *************************** Disclaimer ************************************** 
% This program is distributed in the hope that it will be useful, but 
% WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY 
% or FITNESS FOR A PARTICULAR PURPOSE. Feel free to use this code for academic 
% purposes.  Plase use the citation provided below.
% 
% The test part of this code takes 0.05 seconds per image on a Intel Xeon(R) 
% CPU E5440 @2.83 GHz 8 GB RAM on Matlab 2009b. The most computationally expensive part of 
% this code is the learning phase Y_train_light.m 
% *****************************Copyright*********************************
%   Copyright 2010 Diego Tosato
%   $Revision: 4b$  $Date: 2010/02/25$
% ************************************************************************
%  X_estimate_covariance_window build the covariance matrix for a particular
% image subwindow.
%  Input:
%   pxy   := first order integral tensor
%   Qxy   := second order integral tensor
%   r     := window coordinate
%   d     := number of features
%   Cimg := 
%
%  Output:
%    Cwin := normalized covariance for the r coordinate window
%**************************************************************************
function  Cwin = X_estimate_covariance_window(pxy,Qxy,r,d, Cimg)

% normalized covariance descriptor
Cwin     =   diag(diag(Cimg).^(-1/2))*Y_CovCal_d(pxy,Qxy,r,d)*...
             diag(diag(Cimg).^(-1/2))';

% numerical problems control
if any(isnan(Cwin(:)))
    Cwin(isnan(Cwin)) = 0;
end

if any(isinf(Cwin(:)))
    Cwin(isinf(Cwin)) = 0;
end

if any(diag(Cwin) <= 0)
    D = diag(Cwin);
    D(D <= 0) = 10^(-4);
    D = D - diag(Cwin);
    Cwin = Cwin + diag(D);
    % Cwin = Cwin + eye(d).*(10^-4);
end

% is the covariance matrix symmetry?
[V,D] = eig(Cwin);
if ~isreal(diag(D))
    tu = triu(Cwin);
    di = diag(Cwin);
    tl = tu';
    Cwin = tu+tl-diag(di); 
    [V,D] = eig(Cwin);
end

% is the covariance matrix singualr?
if any(diag(D) < 10^(-4))
    diagD = diag(D);
    diagD(diagD < 10^(-4)) = 10^(-4);
    D = diag(diagD);
    Cwin = V * D * V';
    [~,D] = eig(Cwin);
end

if any(diag(D) < 0)
    error('* non positive definite covariance matrix')
end
if any(diag(D) == 0)
    error('* singilar covariance matrix')
end
end