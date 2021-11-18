FrameLength = 10;
Fs = 2222;
movrmsWin = dsp.MovingRMS(20);
movrmsWin_overlap = dsp.MovingRMS(20,15);
movrmsExp = dsp.MovingRMS('Method','Exponential weighting',...
    'ForgettingFactor',0.995);
scope  = timescope('SampleRate',[Fs,Fs,Fs/(20-15),Fs],...
    'TimeSpanOverrunAction','Scroll',...
    'TimeSpanSource','Property',...
    'TimeSpan',100,...
    'ShowGrid',true,...
    'YLimits',[-1.0 5.5]);
title = 'Moving RMS';
scope.Title = title;
scope.ChannelNames = {'Original Signal',...
    'Sliding window of 20 samples with default overlap',...
    'Sliding window of 20 samples with an overlap of 15 samples',...
    'Exponential weighting with forgetting factor of 0.995'};

count = 1;
Vect = [1/8 1/2 1 2 3 4];
for index = 1:length(Vect)
    V = Vect(index);
    for i = 1:1600
        x = V + 0.1 * randn(FrameLength,1);
        y1 = movrmsWin(x);
        y2 = movrmsWin_overlap(x);
        y3 = movrmsExp(x);
        scope(x,y1,y2,y3);
    end
end