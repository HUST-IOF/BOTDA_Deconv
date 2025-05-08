clear;clc;close all;

%% 仿真参数设置
deltaZ = 0.01;                                               % 计算空间精度
deltaT = deltaZ/2/10^8;                                      % 对应的时间精度
pulseWidth = 40*10^-9;                                       % 脉冲宽度

sectionLength = randi([50,500],800000,1);                    % 每段光纤长度，从0.5 m到5 m
fiberLength = cumsum(sectionLength);                         % 光纤总长度
BFS = (10.89-10.81)*rand(length(sectionLength),1)+10.81;     % BFS随机范围，从10.81 GHz到10.89 GHz
BFS = BFS*10^9;
SW = (35-25)*rand(length(sectionLength),1)+25;               % 谱宽随机范围，从25 MHz到35 MHz
SW = SW*10^6;
Intensity = (1-0.8)*rand(length(sectionLength),1)+0.8;       % 归一化强度随机范围，从0.8到1

BFS1(1:fiberLength(1)) = BFS(1);
SW1(1:fiberLength(1)) = SW(1);
Intensity1(1:fiberLength(1)) = Intensity(1);
for i = 2:length(sectionLength)
    BFS1(fiberLength(i-1)+1:fiberLength(i)) = BFS(i);
    SW1(fiberLength(i-1)+1:fiberLength(i)) = SW(i);
    Intensity1(fiberLength(i-1)+1:fiberLength(i)) = Intensity(i);
end

%% 脉冲扫频输出信号
sweepFreq = 10.78*10^9:2*10^6:10.92*10^9;                    % 频率扫描范围，从10.78 GHz到10.92 GHz，步长2 MHz
% x = 4000000;                                                 % 每一整段光纤长度，看电脑内存配置
x = 400000;                                                 % 每一整段光纤长度，看电脑内存配置
num = 10;                                                    % 光纤段数量
BGS2 = zeros(71,x/10,num);
BFS2 = zeros(num,x/10);
for i = 1:num
    BGS1 = BGSfunction(deltaT,pulseWidth,x,BFS1((i-1)*x+1:i*x),SW1((i-1)*x+1:i*x),Intensity1((i-1)*x+1:i*x),sweepFreq);
    BGS1 = BGS1/max(BGS1(:));                                % 对整段BGS进行归一化
    BGS2(:,:,i) = BGS1(:,251:10:end-150);                    % 让BGS长度和BFS长度一一对应               
    BFS2(i,:) = BFS1((i-1)*x+1:10:i*x);
end

[X1,Y1] = meshgrid((0.1:0.1:0.1*size(BGS2,2)),sweepFreq/10^9);
z = ones(size(BGS2,2),1)*2;

figure
surf(X1,Y1,BGS2(:,:,end),'EdgeColor','interp','FaceColor','interp')
view(0,90)
colormap(jet);
hold on
plot3(X1,BFS2(end,:)/10^9,z,'k')
axis tight
xlabel('Fiber length (m)')
ylabel('Frequency (GHz)')

%% 保存数据
save('BGS5point_noNoise.mat','BGS2')
save('BFS5point.mat','BFS2')

bgs1 = BGS2(:,50);
bgs2 = BGS2(:,100);
bgs3 = BGS2(:,250);
x = 10.78:0.002:10.92;
figure
set(gcf,'Units','centimeter','Position',[5 5 8.5 6]);
plot(x,bgs1,'Linewidth', 2)
hold on
plot(x,bgs2,'Linewidth', 2)
hold on
plot(x,bgs3,'Linewidth', 2)
xlabel(('Frequency (GHz)'),'FontSize',8,'FontWeight','bold');
ylabel(('Brillouin Gain Intensity'),'FontSize',8,'FontWeight','bold');
set(gca,'XLim',[10.78 10.92]);%X轴的数据显示范围
set(gca,'YLim',[0 1]);%X轴的数据显示范围
set(gca,'YTick',[0:0.5:1]);%设置要显示坐标刻度
legend('10m','20m','40m','Color', 'none','location','South'); % legend 会自动根据画图顺序分配图形
