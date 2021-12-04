clear all; close all; clc

% Comparing to signals to find the highest correlation and eliminate the delay
load('Vicon.mat');
load('Mediapipe.mat');
Vicon = Vicon_08;
Mediapipe = Mediapipe_08;

%% VICON
dist_vic = zeros(size(Vicon,1),6); %distance between shoulder-elbow and wrist-elbow
dist_vic(:,1) = table2array(Vicon(:,'Xshoulder')) - table2array(Vicon(:,'Xelbow'));
dist_vic(:,2) = table2array(Vicon(:,'Yshoulder')) - table2array(Vicon(:,'Yelbow'));
dist_vic(:,3) = table2array(Vicon(:,'Zshoulder')) - table2array(Vicon(:,'Zelbow'));
dist_vic(:,4) = table2array(Vicon(:,'Xwrist')) - table2array(Vicon(:,'Xelbow'));
dist_vic(:,5) = table2array(Vicon(:,'Ywrist')) - table2array(Vicon(:,'Yelbow'));
dist_vic(:,6) = table2array(Vicon(:,'Zwrist')) - table2array(Vicon(:,'Zelbow'));

angles_vicon = acos((dist_vic(:,1).*dist_vic(:,4)+dist_vic(:,2).*dist_vic(:,5)+dist_vic(:,3).*dist_vic(:,6))./...
    (sqrt(dist_vic(:,1).^2+dist_vic(:,2).^2+dist_vic(:,3).^2).*sqrt(dist_vic(:,4).^2+dist_vic(:,5).^2+dist_vic(:,6).^2)));

%% Mediapipe

mp_wrist = table2array(Mediapipe(1:3:end,:));
mp_elbow = table2array(Mediapipe(2:3:end,:));
mp_shoulder = table2array(Mediapipe(3:3:end,:));

%filter depth
% mp_wrist(:,3) = hampel(mp_wrist(:,3),20,0.5);
% mp_elbow(:,3) = hampel(mp_elbow(:,3),20,0.5);
% mp_shoulder(:,3) = hampel(mp_shoulder(:,3),20,0.5);

dist_mp = zeros(size(mp_elbow,1),6); %distance between shoulder-elbow and wrist-elbow
dist_mp(:,1:3) = mp_shoulder - mp_elbow;
dist_mp(:,4:6) = mp_wrist - mp_elbow;

angles_mp = acos((dist_mp(:,1).*dist_mp(:,4)+dist_mp(:,2).*dist_mp(:,5)+dist_mp(:,3).*dist_mp(:,6))./...
    (sqrt(dist_mp(:,1).^2+dist_mp(:,2).^2+dist_mp(:,3).^2).*sqrt(dist_mp(:,4).^2+dist_mp(:,5).^2+dist_mp(:,6).^2)));

% angles_mp = hampel(angles_mp,10,0.5);
%% Time vectors
t_vic = linspace(0,size(angles_mp,1),size(angles_vicon,1));
t_mp = linspace(0,size(angles_mp,1),size(angles_mp,1));

% Plot both angles
figure
hold on
plot(t_vic,angles_vicon)
plot(t_mp,angles_mp)
title('BEFORE');
legend('Vicon','Mediapipe');
%% Resample

% Find max freq
freq_max = max(max(1./diff(t_vic)),max(1./diff(t_mp)));
new_freq = ceil(freq_max);

% Upsample to same freq
[s1,t1]=resampleT(angles_vicon,new_freq,t_vic);
[s2,t2]=resampleT(angles_mp,new_freq,t_mp);
s1=(s1-mean(s1))/std(s1);
s2=(s2-mean(s2))/std(s2);

figure
hold on
plot(t1,s1)
plot(t2,s2)
legend('Vicon','Mediapipe')

%% Cross-correlation to find delay
figure
[del,lags]=xcorr(s1,s2);
plot(lags,del)
[~,ind]=max(del);
delay=lags(ind)/new_freq;
sample_delay = lags(ind);

%% Remove delay

t_mp=t_mp+delay;
freq_mp = mean(1./diff(t_mp));
figure
hold on
plot(t_vic,angles_vicon)
plot(t_mp,angles_mp)
ylim([1,3.5])
title('AFTER');
legend('Vicon','Mediapipe');

%% Animation
% L1 = 1;
% sample_delay = new_freq*delay;
% for i = 1:length(s1)
%     x1 = -L1*sin(s1(i));
%     y1 = L1*cos(s1(i)); 
%     x2 = -L1*sin(s2(sample_delay+i));
%     y2 = L1*cos(s2(sample_delay+i)); 
%     
%     figure(7);    
%     plot([0,0],[0,-L1],'-k','Linewidth',2);
%     plot([0,x1],[-L1,-L1+y1],'-k','Linewidth',2);
%     hold on
%     plot([0,0],[0,-L1],'-r','Linewidth',2);
%     plot([0,x2],[-L1,-L1+y2],'-r','Linewidth',2);
%     hold off
%     xlim([-2,2])
%     ylim([-2,2])
%  %pause(0.1)
%     drawnow
% end
%% Coeff Corr 
% s1 = vicon, s2 = mp
if(sample_delay<=0)
    corr = corrcoef(s1(1:length(s2((1+abs(lags(ind))):end))),s2((abs(sample_delay)+1):end));
    
else
    corr = corrcoef(s1((1+abs(sample_delay)):end),s2(1:length(s1((1+abs(lags(ind))):end))));
end