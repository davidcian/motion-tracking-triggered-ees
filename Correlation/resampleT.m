function [signal2,time2] = resampleT(signal,n_freq,time,tmax)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%%

if nargin < 4
    tmax = time(end);
end

time2=0:tmax*n_freq;
time2=time2/n_freq;

try
    [a,b]=unique([time,time2]);
catch
    [a,b]=unique([time',time2]);
end

new_time = b>length(time);

signal2=zeros(1,length(a));
signal2(new_time)=NaN;
signal2(new_time==0)=signal;
F = fillmissing(signal2,'linear','SamplePoints',a);
% F = fillmissing2(signal2,a);

[~,pos]=intersect(a,time2);

signal2=F(pos);

end

