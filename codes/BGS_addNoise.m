clear;clc;close all;
%% 主要将BGS切分成为固定长度的段，并添加随机噪声
BGS1 = load('BGS5point_noNoise.mat');
BGS1 = BGS1.BGS2;

%% BGS切分
num = 71;
BGS2 = zeros(num,540,size(BGS1,3)*(floor(size(BGS1,2)/500)-2));
k = 1;
for m = 1:size(BGS1,3)
    for n = 1:floor(size(BGS1,2)/500)-2
        BGS2(:,:,k) = BGS1(:,500*(n-1)+1+200:500*(n-1)+540+200,m);       % BGS分割
        BGS2(:,:,k) = BGS2(:,:,k)+randn(num,540)/1000*(rand(1)*4.5+0.5); % BGS添加噪声，噪声均方差为0.5/1000到5/1000
        BGS2(:,:,k) = BGS2(:,:,k)/max(max(BGS2(:,:,k)));                 % BGS归一化
        k = k+1;
    end
end

%% BFS切分
BFS1 = load('BFS5point.mat');
BFS1 = BFS1.BFS2;

%% 是否对BFS进行高斯滤波
gausFilter = fspecial('gaussian',[1 5],0.936953125646388);  % 0.5 m高斯卷积核
figure
plot(BFS1(2,:))
% BFS1 = conv2(BFS1,ones(1,5)/5,'same');
BFS1 = imfilter(BFS1,gausFilter,'circular');                % 高斯卷积
hold on
plot(BFS1(2,:))

BFS2 = zeros(size(BFS1,1)*(floor(size(BFS1,2)/500)-2),540);
k = 1;
for m = 1:size(BFS1,1)
    for n = 1:floor(size(BFS1,2)/500)-2
        BFS2(k,:) = BFS1(m,500*(n-1)+1+200:500*(n-1)+540+200);
        k = k+1;
    end
end

[X1,Y1] = meshgrid((0.1:0.1:54),10.78:0.002:10.92);

z = ones(length(BFS2(1,:)),1)*2;

figure
set(gcf,'Units','centimeter','Position',[5 5 8.5 6]);
surf(X1,Y1,BGS2(:,:,end),'EdgeColor','interp','FaceColor','interp')
view(0,90)
colormap(jet);
hold on
plot3(X1,BFS2(end,:)/10^9,z,'k')
axis tight
xlabel(('Fiber length (m)'),'FontSize',8,'FontWeight','bold');
ylabel(('Frequency (GHz)'),'FontSize',8,'FontWeight','bold');
set(gca,'FontName','Cambria','FontSize',8,'FontWeight','bold');
set(gca,'looseInset',[0 0 0.01 0.01])
print('1','-dpng','-r600');

%% 保存数据
save('BGS5point2.mat','BGS2')
save('BFS5point2.mat','BFS2')
