clear %all
close all
clc

%%

load('signals.mat')

signal1=signals.signal1;
signal2=signals.signal2;
time1=signals.time1;
time2=signals.time2;
clear signals

%% find max freq

freq_max = max(max(1./diff(time1)),max(1./diff(time2)));
new_freq = ceil(freq_max);

%% Upsample to same freq

[s1,t1]=resampleT(signal1,new_freq,time1);
[s2,t2]=resampleT(signal2,new_freq,time2);
s1=(s1-mean(s1))/std(s1);
s2=(s2-mean(s2))/std(s2);

figure
hold on
plot(t1,s1)
plot(t2,s2)

%% cross-correlation to find delay
figure
[del,lags]=xcorr(s1,s2);
plot(lags,del)
[~,ind]=max(del);
delay=lags(ind)/new_freq;

%% remove delay

time2=time2+delay;
figure
hold on
plot(time1,signal1)
plot(time2,signal2)