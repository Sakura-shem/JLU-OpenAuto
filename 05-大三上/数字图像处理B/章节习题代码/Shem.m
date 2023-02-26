% Matlab R2022b
% charmean

% 第一章：基本操作
% 打开一张图片
    % 绝对路径读取 到 Image图像数组
image0 = rgb2gray(imread("C:\Users\yangc\Desktop\Test0.jpg"));
image0_0 = imread("C:\Users\yangc\Desktop\Test0_0.jpg");
    % 图像添加 椒盐噪声
image0_1 = imnoise(image0, "salt & pepper", 0.2);
imwrite(image0_1, "C:\Users\yangc\Desktop\Test0_1.jpg", "quality", 100);
    % 彩色图
image0_2 = imread("C:\Users\yangc\Desktop\Test0.jpg");
% 返回图片的行数和列数
[Row, Col] = size(image);
% 显示图像数组的附加信息，有无分号无影响
whos Image0;
% 展示图像 (low, high)--->小于等于黑，大于等于白
imshow(image0);
% 交互式的显示图片的亮度值和坐标和显示光标初始位置和当前位置的欧几里得距离【左键长按】
% pixval;
% 使得 imshow 可以同时展示两幅图，【正常第二次会替代前一次】
figure, imshow(image0);
% 输出 / 保存图像, 适用于 JPEG 图像, 0 ~ 100 间的整数 --> 越小压缩越严重
imwrite(image0, "C:\Users\yangc\Desktop\Test1_0.jpg", "quality", 10);
% 将照片详细信息存入imageInfo中
imageInfo = imfinfo("C:\Users\yangc\Desktop\Test0.jpg");
% 打印信息
fprintf("Image Width is %d, Height is %d, FileSize is %d", imageInfo.Width, imageInfo.Width, imageInfo.FileSize)
% 数组 一维--->向量 ， 二维--->矩阵
v = [1, 3, 5, 7, 9];
% 向量转置
w = v.';
% 取数据, 步长为 2，重新赋值
v = v(2 : 2 :end);



% 第二章：空域增强：点操作

%   图像平移
se = translate(strel(1), [50 140]);%将一个平面结构化元素分别向下和向右移动30个位置
image2_0 = imdilate(image0,se);%利用膨胀函数平移图像
imwrite(image2_0, "C:\Users\yangc\Desktop\Test2_0.jpg", "quality", 100);

%   图片放缩 1.5倍
image2_1 = imresize(image0, 1.5);
imwrite(image2_1, "C:\Users\yangc\Desktop\Test2_1.jpg", "quality", 100);

%   图片旋转 30度，封装的函数在脚本后面
image2_2 = rotate(image0, 30);
imwrite(image2_2, "C:\Users\yangc\Desktop\Test2_2.jpg", "quality", 100);

% 亮度变换 --- 灰度映射
% gamma指定曲线的形状 -- 默认 1
%   负片/明暗翻转
image2_3 = imadjust(image0, [0, 1], [1, 0], 1);
imwrite(image2_3, "C:\Users\yangc\Desktop\Test2_3.jpg", "quality", 100);
%   灰度级别映射
image2_4 = imadjust(image0, [0.5, 0.75], [0, 1]);
imwrite(image2_4, "C:\Users\yangc\Desktop\Test2_4.jpg", "quality", 100);
%   压缩灰度级的低端并扩展灰度级的高端
image2_5 = imadjust(image0, [], [], 2);
imwrite(image2_5, "C:\Users\yangc\Desktop\Test2_5.jpg", "quality", 100);
%   使用对数变换减小动态范围
image2_6 = im2uint8(mat2gray(log(1 + double(image0))));
imwrite(image2_6, "C:\Users\yangc\Desktop\Test2_6.jpg", "quality", 100);
%   对比度拉伸增强
image2_7 = intrans(image0, "stretch", mean2(im2double(image0)), 0.9);
imwrite(image2_7, "C:\Users\yangc\Desktop\Test2_7.jpg", "quality", 100);
%   直方图变换，
% 	直方图显示，b --- 默认256
imhist(image0);
%	直方图均衡化
image2_8 = histeq(image0);
imwrite(image2_8, "C:\Users\yangc\Desktop\Test2_8.jpg", "quality", 100);


% 第三章：空域增强：模板操作
% 线性滤波 
% params：输入图像，滤波掩膜，滤波类型（corr / conv），边界选型（p，replicate，symmetric，circular），大小选型（full，same）
% fspecial params："average", "disk", "gaussian", "laplacian", "log", "motion"
% 均值滤波器
w_a = fspecial('average');
image3_0 = imfilter(image0_1, w_a, "replicate");
imwrite(image3_0, "C:\Users\yangc\Desktop\Test3_0.jpg", "quality", 100);
% 高斯滤波器
w_g = fspecial('gaussian');
image3_1 = imfilter(image0_1, w_g, "replicate");
imwrite(image3_1, "C:\Users\yangc\Desktop\Test3_1.jpg", "quality", 100);
% 非线性滤波
% 中值滤波器
image3_2 = medfilt2(image0_1, "symmetric");
imwrite(image3_2, "C:\Users\yangc\Desktop\Test3_2.jpg", "quality", 100);

% 第四章：频域图像增强
% 高斯低通滤波器
PQ = paddedsize(size(image0_1));
[U, V] = dftuv(PQ(1), PQ(2));
D0 = 0.05 * PQ(2);
F = fft2(image0_1, PQ(1), PQ(2));
H = exp(-(U.^2 + V.^2) / (2 * (D0 ^ 2)));
image4_0 = dftfilt(image0_1, H);
imwrite(image4_0, "C:\Users\yangc\Desktop\Test4_0.jpg", "quality", 100);
% 高斯高通滤波器
PQ = paddedsize(size(image0_1));
D0 = 0.05 * PQ(1);
H = hpfilter("gaussian", PQ(1), PQ(2), D0);
image4_1 = dftfilt(image0_1, H);
imwrite(image4_1, "C:\Users\yangc\Desktop\Test4_1.jpg", "quality", 100);
% 高频增强滤波
PQ = paddedsize(size(image0_1));
D0 = 0.05 * PQ(1);
HBW = hpfilter("btw", PQ(1), PQ(2), D0, 2);
H = 0.5 + 2*HBW;
image4_2 = dftfilt(image0_1, H);
imwrite(image4_2, "C:\Users\yangc\Desktop\Test4_2.jpg", "quality", 100);

% 第五章：图像消噪和恢复 && 第六章：图像矫正和修补
% 添加噪声：imnoise, imnoise2---噪声本身
r = imnoise2("gaussian", 100000, 1, 0, 1);
% 直方图展示噪声
p = hist(r, 50);
% 调和均值滤波器
% Q 为正值
image5_0 = spfilt(image0_1, "chmean", 3, 3, 1.5);
imwrite(image5_0, "C:\Users\yangc\Desktop\Test5_0.jpg", "quality", 100);
% Q 为负值
image5_1 = spfilt(image0_1, "chmean", 3, 3, -1.5);
imwrite(image5_1, "C:\Users\yangc\Desktop\Test5_1.jpg", "quality", 100);
% 自适应中值滤波
image5_2 = adpmedian(image0_1, 7);
imwrite(image5_2, "C:\Users\yangc\Desktop\Test5_2.jpg", "quality", 100);
% 逆滤波
PSF = fspecial("gaussian", 7, 10);
image5_3= deconvreg(image0_1, PSF, 400);
imwrite(image5_3, "C:\Users\yangc\Desktop\Test5_3.jpg", "quality", 100);
% 仿射变换, 默认插值---"bilinear"
s = 0.8;
theta = pi / 6;
T = [s * cos(theta), s * sin(theta), 0
    -s * sin(theta), s * cos(theta), 0
    0, 0, 1];
tform = maketform("affine", T);
image6_0 = imtransform(image0, tform);
imwrite(image6_0, "C:\Users\yangc\Desktop\Test6_0.jpg", "quality", 100);
%  仿射变换，插值 --- "nearest"
image6_1 = imtransform(image0, tform, "nearest");
imwrite(image6_1, "C:\Users\yangc\Desktop\Test6_1.jpg", "quality", 100);

% 第六章：彩色图像处理
% Matlab 只能显示图像的 RGB 版本
% RGB 转 HSI
image1_0 = rgb2hsi(image0_2);
imwrite(image1_0, "C:\Users\yangc\Desktop\Test1_0.jpg", "quality", 100);
% 彩色图像处理思路
% 将三个通道抽离出来，之后对每个通道处理，最后叠加
% 以直方图规定化为例展示彩色图像处理
% 彩色图像的直方图规定化
R = image0_2(:,:,1);%获取原图像R通道
G = image0_2(:,:,2);%获取原图像G通道
B = image0_2(:,:,3);%获取原图像B通道
Rmatch = image0_0(:,:,1);%获取匹配图像R通道
Gmatch = image0_0(:,:,2);%获取匹配图像G通道
Bmatch = image0_0(:,:,3);%获取匹配图像B通道
Rmatch_hist = imhist(Rmatch);%获取匹配图像R通道直方图
Gmatch_hist = imhist(Gmatch);%获取匹配图像G通道直方图
Bmatch_hist = imhist(Bmatch);%获取匹配图像B通道直方图
Rout=histeq(R, Rmatch_hist);%R通道直方图匹配
Gout=histeq(G, Gmatch_hist);%G通道直方图匹配
Bout=histeq(B, Bmatch_hist);%B通道直方图匹配
image1_1(:,:,1) = Rout;
image1_1(:,:,2) = Gout;
image1_1(:,:,3) = Bout;
imwrite(image1_1, "C:\Users\yangc\Desktop\Test1_1.jpg", "quality", 100);




% 函数定义区
function [newimage]=rotate(img,degree)
%获取图片信息 注意三通道获取完 即定义三个变量
[m,n,dep]=size(img);
%计算出旋转之后，形成一个大矩形的长宽 可以看效果图
rm=round(m*abs(cosd(degree))+n*abs(sind(degree)));
rn=round(m*abs(sind(degree))+n*abs(cosd(degree)));
%定义一个新矩阵，三通道的，存储新图片的信息
newimage=zeros(rm,rn,dep);
%坐标变换 分三步 
m1=[1,0,0;0,1,0;-0.5*rm,-0.5*rn,1];
m2=[cosd(degree),sind(degree),0;-sind(degree),cosd(degree),0;0,0,1];
m3=[1,0,0;0,1,0;0.5*m,0.5*n,1];
%利用循环，对每一个像素点进行变换
for i=1:rm
    for j=1:rn
        tem=[i j 1];
        tem=tem*m1*m2*m3;
        x=tem(1,1);
        y=tem(1,2);
        x=round(x);
        y=round(y);
        if(x>0&&x<=m)&&(y>0&&y<=n)
        newimage(i,j,:)=img(x,y,:);
        end
    end
end
end
function image = changeclass(class, varargin)
%CHANGECLASS changes the storage class of an image.
%  I2 = CHANGECLASS(CLASS, I);
%  RGB2 = CHANGECLASS(CLASS, RGB);
%  BW2 = CHANGECLASS(CLASS, BW);
%  X2 = CHANGECLASS(CLASS, X, 'indexed');

%  Copyright 1993-2002 The MathWorks, Inc.  Used with permission.
%  $Revision: 1.2 $  $Date: 2003/02/19 22:09:58 $

switch class
case 'uint8'
   image = im2uint8(varargin{:});
case 'uint16'
   image = im2uint16(varargin{:});
case 'double'
   image = im2double(varargin{:});
otherwise
   error('Unsupported IPT data class.');
end
end

function g = intrans(f, varargin)
%INTRANS Performs intensity (gray-level) transformations.
% Verify the correct number of inputs.
narginchk(2, 4)
% Store the class of the input for use later.
classin = class(f);
% If the input is of class double, and it is outside the range
% [0, 1], and the specified transformation is not 'log', convert the
% input to the range [0, 1].
if strcmp(class(f), 'double') && max(f(:)) > 1 && ...
      ~strcmp(varargin{1}, 'log')
   f = mat2gray(f);
else % Convert to double, regardless of class(f).
   f = im2double(f);
end

% Determine the type of transformation specified.
method = varargin{1};

% Perform the intensity transformation specified.    
switch method
case 'neg' 
   g = imcomplement(f); 

case 'log'
   if length(varargin) == 1  
      c = 1;
   elseif length(varargin) == 2  
      c = varargin{2}; 
   elseif length(varargin) == 3 
      c = varargin{2}; 
      classin = varargin{3};
   else 
      error('Incorrect number of inputs for the log option.')
   end
   g = c*(log(1 + double(f)));

case 'gamma'
   if length(varargin) < 2
      error('Not enough inputs for the gamma option.')
   end
   gam = varargin{2}; 
   g = imadjust(f, [ ], [ ], gam);
   
case 'stretch'
   if length(varargin) == 1
      % Use defaults.
      m = mean2(f);  
      E = 4.0;           
   elseif length(varargin) == 3
      m = varargin{2};  
      E = varargin{3};
   else
       error('Incorrect number of inputs for the stretch option.')
   end
   g = 1./(1 + (m./(f + eps)).^E);
otherwise
   error('Unknown enhancement method.')
end
% Convert to the class of the input image.
g = changeclass(classin, g);
end

function PQ = paddedsize(AB,CD,~ )  
%PADDEDSIZE Computes padded sizes useful for FFT-based filtering.  
%   Detailed explanation goes here  
if nargin == 1  
    PQ = 2*AB;  
elseif nargin ==2 && ~ischar(CD)  
    PQ = QB +CD -1;  
    PQ = 2*ceil(PQ/2);  
elseif nargin == 2  
    m = max(AB);%maximum dimension  

    %Find power-of-2 at least twice m.  
    P = 2^nextpow(2*m);  
    PQ = [P,P];  
elseif nargin == 3  
    m = max([AB CD]);%maximum dimension  
    P = 2^nextpow(2*m);  
    PQ = [P,P];  
else   
    error('Wrong number of inputs');  
end  
end

function [ U,V ] = dftuv( M, N )
%DFTUV 实现频域滤波器的网格函数
%   Detailed explanation goes here
u = 0:(M - 1);
v = 0:(N - 1);
idx = find(u > M/2); %找大于M/2的数据
u(idx) = u(idx) - M; %将大于M/2的数据减去M
idy = find(v > N/2);
v(idy) = v(idy) - N;
[V, U] = meshgrid(v, u);      
end

function g = dftfilt(f,H)
% 此函数可接受输入图像和一个滤波函数，可处理所有的
% 滤波细节并输出经滤波和剪切后的图像
F=fft2(f,size(H,1),size(H,2));
g=real(ifft2(H.*F));
g=g(1:size(f,1),1:size(f,2));
end

function H = hpfilter(type, M, N, D0, n)
if nargin == 4
    n = 1;
end
Hlp = lpfilter(type, M, N, D0, n);
H = 1 - Hlp;
end

function H=lpfilter(type,M,N,D0,n)
[U,V]=dftuv(M,N);
D=hypot(U,V);
switch type
case'ideal'
H=single(D<=D0);
case'btw'
if nargin==4
n=1;
end
H=1./(1+(D./D0).^(2*n));
case'gaussian'
H=exp(-(D.^2)./(2*(D0^2)));
otherwise
error('Unknown filter type.')
end
end

function R=imnoise2(type,M,N,a,b)%type是函数类型，M*N是噪声数组的大小，a,b为两个参数
%设置默认值
if nargin==1%如果函数的输入参数为1，则默认a=0;b=1;M=1;N=1
    a=0;b=1;
    M=1;N=1;
elseif nargin==3%如果函数的输入参数为3，则默认a=0;b=1
    a=0;b=1;
end
%开始运行程序
switch lower(type)
    case 'gaussian'%如果是高斯类型，执行下面方程
        R=a+b*randn(M,N);
    case 'salt & pepper'%如果是焦盐类型，当输入参数小于等于3，a=0.05,b=0.05
        if nargin<=3
            a=0.05;b=0.05;
        end
        %检验Pa+Pb是否大于1
        if(a+b)>1
            error('The sum Pa+Pb must be not exceed >1')
        end
        R(1:M,1:N)=0.5;
        X=rand(M,N);%(0,1)范围内产生一个M*N大小的均匀随机数组
        c=X<=a;%寻找X中小于等于a的数，并赋值为0
        R(c)=0;
        u=a+b;
        c=X>a & X<=u;%寻找X中大于a并小于等于u的数，并赋值为1
        R(c)=1;
    case 'lognormal'%对数正态类型，当输入参数小于等于3，a=1,b=0.25,执行下面方程
        if nargin<=3
            a=1;b=0.25;
        end
        R=a*exp(b*randn(M,N));
    case 'rayleigh'%瑞利类型，执行下面方程
        R=a+(-b*log(1-rand(M,N))).^0.5;
    case 'exponential'%指数类型，执行下面方程
        if nargin<=3%如果输入参数小于等于3，a=1
            a=1;
        end
        if a<=0%如果a=0,错误类型
            error('Parameter a must be positive for exponential type.')
        end
        k=-1/a;
        R=k*log(1-rand(M,N));
    case 'erlang'%厄兰类型，如果输入参数小于等于3，a=2，b=5
        if nargin<=3
            a=2;b=5;
        end
      if(b~=round(b)||b<=0)%如果b=0,错误类型
          error('Param b must a positive integer for Erlang.')
      end 
      k=-1/a;
      R=zeros(M,N);
      for j=1:b
          R=R+k*log(1-rand(M,N));
      end
    otherwise%如果不是以上类型，输出未知分配类型
        error('Unknown distribution type.')
end
end
function f = charmean(g, m, n, q)
inclass = class(g);
g = im2double(g);
f = imfilter(g.^(q + 1), ones(m, n), "replicate");
f = f./(imfilter(g.^q, ones(m, n), "replicate") + eps);
f = changeclass(inclass, f);
end

function f = spfilt(g,type,m,n,parameter)
%spfilt执行线性和非线性的空间滤波器，g为原图像，type为滤波器类型，M*N为滤波器模板大小
%处理输入参数
if nargin==2
    m=3;n=3;Q=1.5;d=2;
elseif nargin==5
    Q=parameter;d=parameter;
elseif nargin==4
    Q=1.5;d=2;
else
    error('Wrong number of inputs.');
end
%开始执行滤波
switch type
    case 'amean'%算数平均滤波
        w=fspecial('average',[m n]);
        f=imfilter(g,w,'replicate');
    case 'gmean'%几何平均滤波
        f=gmean(g,m,n);
    case 'hmean'%调和平均滤波
        f=harmean(g,m,n);
    case 'chmean'%反调和平均滤波，Q的默认值是1.5
        f = charmean(g,m,n,Q);
    case 'median'%中值滤波
        f = medfilt2(g,[m n],'symmetric');
    case 'max'%最大值滤波
        f = ordfilt2(g,m*n,ones(m,n),'symmetric');
    case 'min'%最小值滤波
        f = ordfilt2(g,1,ones(m,n),'symmetric');
    case 'midpoint'%中值滤波
        f = ordfilt2(g,1,ones(m,n),'symmetric');
        f = ordfilt2(g,m*n,ones(m,n),'symmetric');
        f = imlincomb(0.5,f1,0.5,f2);
    case 'atrimmed'%顺序平均值滤波，d必须是非负的数，d的默认值是2
        if(d<0)||(d/2~=round(d/2))
            error('d must be a nonnegative,even integer.')
        end
        f=alphatrim(g,m,n,d);
    otherwise
        error('Unkown filter type.')
end
end

function f = adpmedian(g,Smax)
 
if (Smax<=1)||(Smax/2==round(Smax/2))||(Smax~=round(Smax))
    error('Smax must be an odd integer >1.')
end
f=g;
f(:)=0;
alreadyProcessed=false(size(g));
 
for k=3:2:Smax
    zmin=ordfilt2(g,1,ones(k,k),'symmetric');
    zmax=ordfilt2(g,k*k,ones(k,k),'symmetric');
    zmed=medfilt2(g,[k k],'symmetric');
    
    processUsingLevelB=(zmed>zmin)&(zmax>zmed)&...
        ~alreadyProcessed;
    zB=(g>zmin)&(zmax>g);
    outputZxy=processUsingLevelB & zB;
    outputZmed=processUsingLevelB&~zB;
    f(outputZxy)=g(outputZxy);
    f(outputZmed)=zmed(outputZmed);
    
    alreadyProcessed=alreadyProcessed | processUsingLevelB;
    if all(alreadyProcessed(:))
        break;
    end
end
f(~alreadyProcessed)=zmed(~alreadyProcessed); 
end
function hsi = rgb2hsi(rgb) 
% RGB2HSI Converts an RGB image to HSI. 
% Extract the individual component immages. 
rgb = im2double(rgb); 
r = rgb(:, :, 1); 
g = rgb(:, :, 2); 
b = rgb(:, :, 3); 
% Implement the conversion equations. 
num = 0.5*((r - g) + (r - b)); 
den = sqrt((r - g).^2 + (r - b).*(g - b)); 
theta = acos(num./(den + eps)); 
H = theta; 
H(b > g) = 2*pi - H(b > g); 
H = H/(2*pi); 
num = min(min(r, g), b); 
den = r + g + b; 
den(den == 0) = eps; 
S = 1 - 3.* num./den; 
H(S == 0) = 0; 
I = (r + g + b)/3; 
hsi = cat(3, H, S, I); 
end

