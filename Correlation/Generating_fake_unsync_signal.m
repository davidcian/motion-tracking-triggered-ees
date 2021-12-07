close all
clear %all
clc

amp=linspace(0,180,1000);
x=linspace(pi()/2,1.5*pi(),1000);

t=linspace(0,60,1000);

signal=cos(x).*sin(amp);

figure
plot(t,signal)

%%

thres=0.25;
signal1=zeros(1,length(signal));
signal2=zeros(1,length(signal));
time1=t;
time2=t;
for k=1:length(signal)
    if rand()<thres
        signal1(k)=signal(k)+(rand()-0.5);
    else
        signal2(k)=signal(k)+(rand()-0.5);
    end
end
id1=signal1==0;
signal1(id1)=[];
time1(id1)=[];
signal2(id1==0)=[];
time2(id1==0)=[];

del=10;

time2=[0:0.1:9.9,time2+del];
signal2=[zeros(1,length(0:0.1:9.9)),signal2];

figure
subplot(211)
plot(time1,signal1)
subplot(212)
plot(time2,signal2)

signals.signal1=signal1;
signals.signal2=signal2;
signals.time1=time1;
signals.time2=time2;