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
% X_build_set biuld and store a set of integral tensors for fast covariance
% computation
% input: 
%   data_dir    = images dir
%   train_dir   = integral images dir
%   n_row,n_col = image dimension
%   d           = number of features
%   coords      = ROI layout
% output:
%   N           = number of examples per class
%**************************************************************************
function [N] = X_build_set(data_dir,train_dir,n_row,n_col,d,coords)
fb = figure;
if(size(dir(train_dir),1)~=0 )
    disp('skip set creation...');
    load([train_dir '/N']);
else
    disp('set creation...');
    data_dir_categories       =   dir(strcat(data_dir,'/Data_*'));
    J                         =   size(data_dir_categories,1); % number of classes
    N                         =   zeros(J,1); % number of examples per class
    for j = 1:J % for all classes
        NN   =   1;%example counter
        fg   =   [dir([data_dir '/' data_dir_categories(j).name '/*.png']);...
            dir([data_dir '/' data_dir_categories(j).name '/*.bmp']);...
            dir([data_dir '/' data_dir_categories(j).name '/*.jpg']);...
            dir([data_dir '/' data_dir_categories(j).name '/*.pgm']);];
        n_img               =   size(fg,1); % number of images
        mkdir([train_dir '/' data_dir_categories(j).name]);
        to_show             =  zeros(n_img,coords(3),coords(4));
        e                   =   1;
        for nn   =   1:n_img % for all images in the j-th class
            disp(['-> pos image ' num2str(nn) '/' num2str(n_img)]);
            img            =   imread([data_dir '/' data_dir_categories(j).name '/' fg(nn).name]);
            si = size(img);
            %to ldg
            if length(si) > 2
                img_rgb    =   img;
                img        =   rgb2gray(img);
            end
            img            =   img(coords(1):coords(3),coords(2):coords(4));
            img_rgb        =   img_rgb(coords(1):coords(3),coords(2):coords(4),:);
            to_show(e,:,:) =   img; % visualization
            % features extraction
            img_feat        =   Y_BuildF(img_rgb,coords,d);
            % integral integrals building
            pxy             =   zeros(size(img_feat)); % first order tensor of integral images
            Qxy             =   zeros(size(img_feat,1),size(img_feat,2),size(img_feat,3),size(img_feat,3)); % second order tensor of integral images
            for ii = 1:d
                pxy(:,:,ii) =  cumsum(cumsum(img_feat(:,:,ii),2));
                fprintf('*')
                for jj = ii:d
                    Qxy(:,:,ii,jj)    =  cumsum(cumsum(img_feat(:,:,ii).*img_feat(:,:,jj)),2);
                    Qxy(:,:,jj,ii)    =  Qxy(:,:,ii,jj);
                end
            end
            % structure data
            Data.name       =   fg(nn).name;
            Data.class      =   j;
            Data.img        =   img;
            Data.Coord      =   [1 1 n_row n_col]; % matrix coords
            Data.pxy        =   pxy;
            Data.Qxy        =   Qxy;
            imagesc(img);
            drawnow
            fprintf('\n');
            % save data
            name            =   [train_dir '/' data_dir_categories(j).name '/' 'Data_' Data.name(1:end-4)];
            save(name,'Data');
            NN   =   NN+1;
            e   =   e+1;
        end
        name            =   [train_dir '/' data_dir_categories(j).name '/' 'summary'];
        save(name,'class_pose','class_pose_var');
        N(j) = NN-1;
    end
    name              =   [train_dir '/' 'N'];
    save(name,'N');
end
close(fb);
end