%%%%%%%VERSION 2
%%ANOTHER DESCRIBTION OF GABOR FILTER

%The Gabor filter is basically a Gaussian (with variances sx and sy along x and y-axes respectively)
%modulated by a complex sinusoid (with centre frequencies U and V along x and y-axes respectively) 
%described by the following equation
%%
%                            -1     x' ^     y'  ^             
%%% G(x,y,theta,f) =  exp ([----{(----) 2+(----) 2}])*cos(2*pi*f*x');
%                             2    sx'       sy'
%%% x' = x*cos(theta)+y*sin(theta);
%%% y' = y*cos(theta)-x*sin(theta);

%% Describtion :

%% I : Input image
%% Sx & Sy : Variances along x and y-axes respectively
%% f : The frequency of the sinusoidal function
%% theta : The orientation of Gabor filter

%% G : The output filter as described above
%% gabout : The output filtered image

% from 'gaborfilter1' with different f(Frequency) and theta(Angle). 
% for example
% 
% f:0,2,4,8,16,32 
% theta = 0,pi/3,pi/6,pi/2,3pi/4
% 
% then for any input image like(eg. stereo.jpg)
% 
% you have 6x5 = 30 filtered images.
% 
% You can choose your desired angles or frequencies.
% 
% You can put nominaly Sx & Sy = 2,4 or some one else.
% 
% For instance I tested above example on ('cameraman.tif')(in MATLAB pictures)
% 
% I = imread('cameraman.tif'); 
% [G,gabout] = gaborfilter1(I,2,4,16,pi/3); 
% figure,imshow(uint8(gabout));

%%  Author : Ahmad poursaberi  e-mail : a.poursaberi@ece.ut.ac.ir
%%          Faulty of Engineering, Electrical&Computer Department,Tehran
%%          University,Iran,June 2004

function [G,gabout] = gaborfilter1(I,Sx,Sy,f,theta)

if isa(I,'double')~=1 
    I = double(I);
end

for x = -fix(Sx):fix(Sx)
    for y = -fix(Sy):fix(Sy)
        xPrime = x * cos(theta) + y * sin(theta);
        yPrime = y * cos(theta) - x * sin(theta);
        G(fix(Sx)+x+1,fix(Sy)+y+1) = exp(-.5*((xPrime/Sx)^2+(yPrime/Sy)^2))*cos(2*pi*f*xPrime);
    end
end

Imgabout = conv2(I,double(imag(G)),'same');
Regabout = conv2(I,double(real(G)),'same');

gabout = sqrt(Imgabout.*Imgabout + Regabout.*Regabout);