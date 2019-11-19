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
%  Y_CovCal returns a covariance descriptors C using integral tensors
%
% Input:
%   pxy := first order tensor of integral images WxHxd
%   Qxy := second order tensor of integral images WxHxd
%   R   := 4-tuple of the region (in absolute or relative coordinate)
%   d   := number of features
%
% Output:
%   C   := covariance descriptors
%******************************References**********************************
%      [1]O. Tuzel, F. Porikli, P. Meer: Pedestrian Detection via
%      Classification on Riemannian Manifolds, Pattern Analysis and Machine
%      Intelligence, IEEE Transactions on 
%**************************************************************************
function C = Y_CovCal_d(pxy,Qxy,R,d)
% Covariance matrix  computation 
  if(R(1) == 1 && R(2) == 1) 
    S    = R(3)*R(4); % normalization term
    to_transp = reshape(pxy(R(3),R(4),:),d,1);
    C = 1/(S-1).*(reshape(Qxy(R(3),R(4),:,:),d,d) -...
        1/S.*(to_transp*to_transp'));
    
  elseif R(1) == 1
     S    = (R(3) - R(1) + 1)*(R(4) - R(2) + 1); % normalization term
      first_fact = reshape(Qxy(R(3),R(4),:,:) - Qxy(R(3),R(2)-1,:,:),d,d);
      to_transp = reshape((pxy(R(3),R(4),:) -  pxy(R(3),R(2)-1,:)),d,1);
      C = (1/(S-1)).*(first_fact-(1/S).*(to_transp*to_transp'));
  elseif R(2) == 1
      S    = (R(3) - R(1) + 1)*(R(4) - R(2) + 1);% normalization term
      first_fact = reshape(Qxy(R(3),R(4),:,:) -  Qxy(R(1)-1,R(4),:,:),d,d);
      to_transp = reshape((pxy(R(3),R(4),:) - pxy(R(1)-1,R(4),:)),d,1);
      C = (1/(S-1)).*(first_fact-(1/S).*(to_transp*to_transp'));
  else
      S    = (R(3) - R(1) + 1)*(R(4) - R(2) + 1);% normalization term
      first_fact = reshape(Qxy(R(3),R(4),:,:)    +    Qxy(R(1)-1,R(2)-1,:,:)   -  Qxy(R(3),R(2)-1,:,:)  -  Qxy(R(1)-1,R(4),:,:),d,d);
      to_transp = reshape((pxy(R(3),R(4),:)    +    pxy(R(1)-1,R(2)-1,:)     -  pxy(R(1)-1,R(4),:)    -  pxy(R(3),R(2)-1,:)),d,1);
      C = (1/(S-1)).*(first_fact-(1/S).*(to_transp*to_transp'));
  end

  