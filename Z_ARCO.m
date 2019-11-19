% *************************** Disclaimer ************************************** 
% This program is distributed in the hope that it will be useful, but 
% WITHOUT ANY WARRANTY; without even the implied warranty of  MERCHANTABILITY 
% or FITNESS FOR A PARTICULAR PURPOSE. Feel free to use this code for academic 
% purposes.  Please use the citation provided below.
% 
% The test part of this code takes 0.05 seconds per image on a Intel Xeon(R) 
% CPU E5440 @2.83 GHz 8 GB RAM on Matlab 2009b. The most computationally expensive part of 
% this code is the learning phase Y_train_light.m 
% *****************************Citation****************************************
% Diego Tosato, Michela Farenzena, Mauro Spera, Marco Cristani, Vittorio Murino 
% “Multi-class Classification on Riemannian Manifolds for Video Surveillance,”
% ECCV, 2010.  
% *****************************Performance*************************************
% This code performs patch-based head pose detection given the a 50x50 image. 
% Should you have any problems please email me at diego.tosato@univr.it
% **********************Demo code and images provides*************************
% There is one complete script for learning and testing ARCO:
% 
% 1) Z_ARCO.m: this is the main script. It is able to learn and test a 
% multi-class Logitboost classifier on Riemannian Manifold.
%
% 2) The variable 'experiment' contains a path where all the 
% pre-computed parts of this framework are stored.
% Only the classification results are not computed in order to
% show you some qualitative results of this framework.
%
% 3) If you want to test this framework on the complete test set, just change
% the variable 'test_dir' from './QML4PoseHeads/test_demo' to
% './QML4PoseHeads/test'.
%
% 4) If you want to see the statistics of this framework on the complete
% test set without run testing, these are in [experiment
% '/full_test_results'].
% 
% This code is provided with a pre-computed training set and its learned 
% classifier in order to directly test the classifier.
 
% ********************Some Important References*****************************
%   [1] Fast spatial pattern discovery integrating boosting with 
%   constellations of contextual descriptors; Amores, J. and Sebe, 
%   N. and Radeva, P.; IEEE COMPUTER SOCIETY CONFERENCE ON COMPUTER 
%   VISION AND PATTERN RECOGNITION; 2005;
%   [2]  A local basis representation for estimating human pose from
%   cluttered images;Agarwal, A. and Triggs, B.; Lecture Notes in Computer
%   Science 2006
%   [3]  Head Pose Classification in Crowded Scenes; Orozco, J. and 
%   Gong, S. and Xiang, T; BMWC 2009;   
% *****************************Copyright***********************************
%   Copyright 2010 Diego Tosato
%   $Revision: 4b$  $Date: 2010/02/25$
%**************************************************************************

%% General settings
clc
clear
close all
addpath('./my_utils');
addpath('./my_utils/MultiClassLogitBoost/');
J               = 5; % number of classes
n_part          = 1; % number of obj sub-part
patch_step      = 8; % patch dimesion
patch_d         = 16; % patch side size
local_num       = 25; % number of patches
n_row           = 50; %image size
n_col           = 50;
ROI             = [1,1,n_row,n_col]; % image ROI
d               = 12; % number of fature
template        = zeros(d,d,n_part,J); % \in M  
min_leaf        = 150; % min. number of leaf for a regression tree
thr_leran       = .01; % learning rate
c_rate          = [.99 .99 .99 .99 .99]; % target accuracy
Id              = eye(d);
train_dir       = './QML4PoseHeads/train'; % train dir
test_dir        = './QML4PoseHeads/test_demo'; % test dir
% integral tensors (for fast covariance computatiuon) dir
trainingset_dir = ['./QML_HEAD_2010_03_03' '_r' num2str(n_row) ...
    '_c' num2str(n_col) '_d' num2str(d) '_RGB_GABOR_GRAD'];
% results dir
experiment = ['test/' 'test2010-2-28/' 'QML4PoseHeads'...
    '_r' num2str(n_row) '_c' num2str(n_col) '_d' num2str(d) '_prt' ...
    num2str(n_part) '_ln' num2str(local_num) '_ps' num2str(patch_step)...
    '_pd' num2str(patch_d) '_rtree' num2str(min_leaf)  '_RGB_GABOR_GRAD_GRID'];

disp(experiment)

%% integral tensors calculation 
[N_train]           = X_build_set(train_dir,trainingset_dir,n_row,n_col,d,ROI);
fg_dir_categories   = dir(strcat(trainingset_dir,'/Data_*'));

%% covariance descriptor calculation
C                =   zeros(d,d,max(N_train),local_num,n_part,J);
num              =   N_train;
%% covariances calculation
if(exist([experiment '/ts_covariance.mat'],'file') > 0)
    disp('skip data calculation...');
    load([experiment '/ts_covariance']);    
else
    disp('covariances calculation...')
    mkdir(experiment);
    for j = 1:J
        name_class =   dir(strcat(trainingset_dir,'/',fg_dir_categories(j).name,'/Data*'));
        for i=1:num(j)
            name =   [trainingset_dir '/' fg_dir_categories(j).name ...
                '/' name_class(i).name];
            load(name);
            Coord    =   [1,1,n_row,n_col];
            % image covariance matrix (computed for normalization purposes)
            DataC    =   Y_CovCal_d(Data.pxy,Data.Qxy,ROI,d);
            % covariance check
            if any(diag(DataC) <= 0)
                D = diag(DataC);
                D(D <= 0) = 10^(-4);
                D = D - diag(DataC);
                DataC = DataC + diag(D);
            end
            % patch layout
            range_row  =   n_row - patch_d/2;
            range_col  =   n_col - patch_d/2;
            [e_row e_col] = meshgrid(patch_d/2:patch_step:range_row,patch_d/2:patch_step:range_col);%30
            e_row_selection = e_row(:);
            e_col_selection = e_col(:);
            for p = n_part;
                C_local = zeros(d,d,local_num);
                for t = 1:local_num  % for each subwindows
                    % coords calculation
                    r(1)   =  e_row_selection(t) - patch_d/2+1;
                    r(2)   =  e_col_selection(t) - patch_d/2+1;
                    r(3)   =  e_row_selection(t) + patch_d/2;
                    r(4)   =  e_col_selection(t) + patch_d/2;
                    % covarince calculation
                    C_tmp = X_estimate_covariance_window(Data.pxy,Data.Qxy,r,d, DataC);
                    % storing covariances
                    C(:,:,i,t,p,j) = C_tmp;
                end
            end
        end
    end
    % save data
    name              =   [experiment '/' 'ts_covariance'];
    save(name,'C','num');
end

%% multi-class classifier calculation
if(exist([experiment '/model.mat'],'file') > 0)
    disp('skip model calculation...');
    load([experiment '/model']);
else
    disp('model calculation...')
    N      = sum(num); % total number of examples
    C_vect = cell(J,n_part,local_num);
    patch  = zeros(d*(d+1)/2,N); % dimension of a projected patch
    % mapping on Sim_d on T_Id
    for j = 1:J
        for p = 1:n_part
            for t = 1:local_num
                i = 1;
                for jj = 1:J
                    for ii  = 1:num(jj)
                        % covarince vectorization
                        patch(:,i) = Y_vect(Id, X_logp(Id,C(:,:,ii,t,p,jj)));
                        i = i + 1;
                    end
                end
                C_vect{j,p,t}  = patch';
            end
        end
    end
    % classifier structure
    F       = cell(local_num,n_part);
    % patch classifier learning
    % diary([experiment '/log'])
    for p = 1:n_part
        for t = 1:local_num
            disp(['-| Strong learner: ' ' patch ' num2str(t)])
            F{t,p}  = [F{t,p},{Y_Train_light({C_vect{:,p,t}},num,c_rate,min_leaf,thr_leran)}];%modelterms
        end
    end
    % diary off
    name              =   [experiment '/' 'model'];
    save(name,'F');
end


%% image testing
if(exist([experiment '/classification.mat'],'file') > 0)
    load([experiment '/classification']);
    disp('skip testing...');
else
    disp('testing...');
    CM                  =   zeros(J,n_part,J); % confusion matrix                
    fg_dir_categories   =   dir(strcat(test_dir,'/Test_*'));
    num_test            =   zeros(J,1);
    for j = 1:J % for each class
        disp(['-> classification class: ' num2str(j) '/' num2str(J)])
        % collect data
        name_class =   [dir(strcat(test_dir,'/',fg_dir_categories(j).name,'/*.jpg'));
            dir(strcat(test_dir,'/',fg_dir_categories(j).name,'/*.bmp'));
            dir(strcat(test_dir,'/',fg_dir_categories(j).name,'/*.png'));];
        % number of testing examples per class
        num_test(j)                =   size(name_class,1);
        for i=1:num_test(j)
            %tic
            disp(['image ' num2str(i) '/' num2str(num_test(j))])
            %load image
            name =   [test_dir '/' fg_dir_categories(j).name ...
                '/' name_class(i).name];
            img             =   imread(name);
            si = size(img);
            %to ldg
            if length(si) > 2
                img_rgb    =   img;
                img        =   rgb2gray(img);
            end
            img            =   img(ROI(1):ROI(3),ROI(2):ROI(4));
            img_rgb        =   img_rgb(ROI(1):ROI(3),ROI(2):ROI(4),:);
            % image features calculation
            img_feat       =   Y_BuildF(img_rgb,ROI,d);
             % integral integrals building
            Data.pxy             =   zeros(size(img_feat)); % first order tensor of integral images
            Data.Qxy             =   zeros(size(img_feat,1),size(img_feat,2),size(img_feat,3),size(img_feat,3)); % second order tensor of integral images
            for ii = 1:d
                Data.pxy(:,:,ii) =  cumsum(cumsum(img_feat(:,:,ii),2));
                for jj = ii:d
                    Data.Qxy(:,:,ii,jj)    =  cumsum(cumsum(img_feat(:,:,ii).*img_feat(:,:,jj)),2);
                    Data.Qxy(:,:,jj,ii)    =  Data.Qxy(:,:,ii,jj);
                end
            end
            Coord = ROI;
            % image covariance matrix (computed for normalization purposes)
            DataC    =   Y_CovCal_d(Data.pxy,Data.Qxy,Coord,d);
            % covariance control
            if any(diag(DataC) <= 0)
                D = diag(DataC);
                D(D <= 0) = 10^(-4);
                D = D - diag(DataC);
                DataC = DataC + diag(D);
            end
            % patch layout
            range_row  =   n_row - patch_d/2;
            range_col  =   n_col - patch_d/2;
            [e_row e_col] = meshgrid(patch_d/2:patch_step:range_row,patch_d/2:patch_step:range_col);%30
            e_row_selection = e_row(:);
            e_col_selection = e_col(:);
            for p = n_part;
                % result
                count = zeros(1,J);
                for t = 1:local_num  % for each patch
                    % coords calculation
                    r(1)   =  e_row_selection(t) - patch_d/2+1;
                    r(2)   =  e_col_selection(t) - patch_d/2+1;
                    r(3)   =  e_row_selection(t) + patch_d/2;
                    r(4)   =  e_col_selection(t) + patch_d/2;
                    % covarince calculation
                    test = X_estimate_covariance_window(Data.pxy,Data.Qxy,r,d, DataC);
                    % vectorization on T_Id
                    test = Y_vect(Id, X_logp(Id,test));
                    % classification
                    [y_out,y_confidence] = Y_Test_light(F{t,p},test,1,J);
                    % save the classification result
                    count(y_out) = count(y_out) +1;
                end
                % confusion matrix compiling
                [~,idx] = max(count);
                CM(j,p,idx(1)) = CM(j,p,idx(1)) + 1;
                % display results
                if idx == 1
                    imagesc(img_rgb);title('back');axis image;drawnow;
                elseif idx == 2
                    imagesc(img_rgb);title('frontal');axis image;drawnow;
                elseif idx == 3
                    imagesc(img_rgb);title('left');axis image;drawnow;
                elseif idx == 4
                    imagesc(img_rgb);title('right');axis image;drawnow;
                else
                    imagesc(img_rgb);title('bg');axis image;drawnow;
                end    
            end
            %toc
        end
    end
    % save results
    name              =   [experiment '/' 'classification'];
    save(name,'CM');
end


% results visualization
for z = 1:n_part
   CM_tmp = squeeze(CM(:,z,:));
   disp('-CONFUSION MATRIX')
   disp(CM_tmp)
   c_sum = sum(CM_tmp,2);
   for j = 1:J
       CM_tmp(j,:) = CM_tmp(j,:)/c_sum(j);
   end
   disp('-CONFUSION MATRIX (%)')
   disp(CM_tmp)
   disp('- AVG ACCURACY')
   disp(mean(diag(CM_tmp)))
end