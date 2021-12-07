clear all
close all
%% Correction of .mat files (ok)

% for i=1:7
%     fname = sprintf("config%d.mat", i);
%     sname = sprintf("config%d", i);
%     save(fname, '-struct', sname);
% end

%% Loading of EMG

% Set indices of the different sensors
EMG_idx = [1, 2, 3, 5, 12, 19, 26, 33];
unused_EMG_idx = 4;
ACC_idx = [6, 7, 8, 13, 14, 15, 20, 21, 22, 27, 28, 29, 34, 35, 36];
GYR_idx = [9, 10, 11, 16, 17, 18, 23, 24, 25, 30, 31, 32, 37, 38, 39];
trigger_idx = 40;

nb_EMG = length(EMG_idx);

% Assign channel name to the muscle it is recording 
Channels_EMG = ["Deltoid Anterior", "Deltoid Middle", ...
    "Deltoid Posterior", "Fingers extensor", "Biceps", "Triceps", ...
    "Finger flexor", "Bracchioradial"];

% Load baseline and keep only EMG
baseline = load("Run_number_438_baseline_Rep_1.5.mat");
baseline_EMG = struct('Channels', ...
    mat2cell(baseline.Channels(1, EMG_idx), 1, nb_EMG), ...
    'Data', mat2cell(baseline.Data(1, EMG_idx), 1, nb_EMG), ...
    'Time', mat2cell(baseline.Time(1, EMG_idx), 1, nb_EMG), ...
    'Fs', mat2cell(baseline.Fs(1, EMG_idx), 1, nb_EMG));

% Load MVC and keep only EMG
MVC = load("Run_number_439_MVC_Rep_1.6.mat");
MVC_EMG = struct('Channels', Channels_EMG, ...
    'Data', mat2cell(MVC.Data(1, EMG_idx), 1, nb_EMG), ...
    'Time', mat2cell(MVC.Time(1, EMG_idx), 1, nb_EMG), ...
    'Fs', mat2cell(MVC.Fs(1, EMG_idx), 1, nb_EMG));

% Init configs struct
configs = struct('Channels', {}, 'Data', {}, 'Time', {}, 'Fs', {});
configs_EMG = struct('Channels', {}, 'Data', {}, 'Time', {}, 'Fs', {});

% Load each config and keep only EMG
for i=1:7
    fname = sprintf("config%d.mat", i);
    cfg = load(fname);
    configs(i) = cfg;
    configs_EMG(i) = struct('Channels', Channels_EMG, ...
        'Data', mat2cell(cfg.Data(1, EMG_idx), 1, nb_EMG), ...
        'Time', mat2cell(cfg.Time(1, EMG_idx), 1, nb_EMG), ...
        'Fs', mat2cell(cfg.Fs(1, EMG_idx), 1, nb_EMG));
end

%% New preprocessing pipeline - Rectification

% Remove mean to each signal (mean) and rectify
for i=1:nb_EMG
    % MVC
    MVC_EMG.DataRect{1,i} = abs(MVC_EMG.Data{1,i} - mean(MVC_EMG.Data{1,i}));
    
    % Configurations
    for j=1:length(configs_EMG)
        configs_EMG(j).DataRect{1,i} = abs(configs_EMG(j).Data{1,i} - ...
            mean (configs_EMG(j).Data{1,i}));
    end
end

%% New preprocessing pipeline - Smoothing
% Set parameters
fc = 100;
order = 4;
muscle_MVC = zeros(1, length(baseline_EMG.Data));

for i=1:nb_EMG
    % Prepare butterworth and window size
    % 300ms window for moving rms
    win = round(300e-3 * MVC_EMG.Fs(i));
    % Set butterworth filter
    Wn = fc / (MVC_EMG.Fs(i)/2);
    [b, a] = butter(order, Wn);

    % MVC
    MVC_EMG.butter{1,i} = filtfilt(b1, a1, MVC_EMG.DataRect{1,i});
    MVC_EMG.movRMS{1,i} = fastrms(MVC_EMG.butter{1,i}, win);

    muscle_MVC(i) = max(MVC_EMG.movRMS{1,i});
    
    % Configurations
    for j=1:length(configs_EMG)
        configs_EMG(j).butter{1,i} = ...
            filtfilt(b, a, configs_EMG(j).DataRect{1,i});
        configs_EMG(j).movRMS{1,i} = ...
            fastrms(configs_EMG(j).butter{1,i}, win);
    end
end

% for i=1:nb_EMG
%     figure; hold on
%     plot(MVC_EMG.Time{1,i}, MVC_EMG.DataRect{1,i})
%     plot(MVC_EMG.Time{1,i}, MVC_EMG.butter{1,i})
%     plot(MVC_EMG.Time{1,i}, MVC_EMG.movRMS{1,i})
% end

%% New preprocessing pipeline - Normalization

for i=1:length(configs_EMG)
    for j=1:nb_EMG
        configs_EMG(i).movRMS{1,j} = ...
            fastrms(configs_EMG(i).butter{1,j}, win) / muscle_MVC(j);
    end
end

%% Apply preprocessing to the whole dataset

% Set Butterworth filter parameters
fc = 20;
order = 4;

for i=1:nb_EMG
    % 300ms window for moving rms
    win = round(300e-3 * baseline_EMG.Fs(i));
    % Set butterworth filter
    Wn = fc / (baseline_EMG.Fs(i)/2);
    [b, a] = butter(order, Wn);

    % Baseline preprocessing
    %baseline_EMG.DataBis{1,i} = filtfilt(b, a, abs(baseline_EMG.Data{1,i} - mean(baseline_EMG.Data{1,i})));
    baseline_EMG.DataBis{1,i} = filtfilt(b, a, baseline_EMG.Data{1,i});
    baseline_EMG.Envelope{1,i} = fastrms(baseline_EMG.DataBis{1,i}, win);

    % MVC preprocessing
    MVC_EMG.DataBis{1,i} = filter(b, a, MVC_EMG.Data{1,i} - mean(MVC_EMG.Data{1,i}));
    MVC_EMG.Envelope{1,i} = fastrms(MVC_EMG.DataBis{1,i}, win);

    % Configs preprocessing
    for j=1:length(configs_EMG)
        configs_EMG(j).DataBis{1,i} = filter(b, a, configs_EMG(j).Data{1,i} - mean (configs_EMG(j).Data{1,i}));
        configs_EMG(j).Envelope{1,i} = fastrms(configs_EMG(j).DataBis{1,i}, win);
    end
end


% for i=1:nb_EMG
%     figure; hold on
%     plot(baseline_EMG.Time{1,i}, abs(baseline_EMG.Data{1,i}))
%     plot(baseline_EMG.Time{1,i}, baseline_EMG.Envelope{1,i})
% end

% Baseline
% figure; hold on
% plot(baseline_EMG.Time{1,2}, abs(baseline_EMG.Data{1,2}))
% plot(baseline_EMG.Time{1,2}, abs(baseline_EMG.DataBis{1,2}))
% plot(baseline_EMG.Time{1,2}, baseline_EMG.Envelope{1,2})

% MVC
% for i=1:nb_EMG
%     figure
%     subplot(3,1,1)
%     plot(MVC_EMG.Time{1,i}, abs(MVC_EMG.Data{1,i} - mean(MVC_EMG.Data{1,i})))
%     subplot(3,1,2)
%     plot(MVC_EMG.Time{1,i}, abs(MVC_EMG.DataBis{1,i}))
%     subplot(3,1,3)
%     plot(MVC_EMG.Time{1,i}, MVC_EMG.Envelope{1,i})
%     title(MVC_EMG.Channels(i))
% end

%% Baseline computation and correction

% Plot baseline for each EMG channel (i.e. each muscle)
% for i=1:nb_EMG
%     subplot(8, 1, i)
%     plot(baseline_EMG.Time{1,i}, baseline_EMG.Data{1,i})
% end

% Variables init
muscle_baseline = zeros(1, length(baseline_EMG.Data));
muscle_MVC = zeros(1, length(baseline_EMG.Data));

% Baseline correction
for i=1:nb_EMG
    % Baseline computation
    muscle_baseline(i) = max(baseline_EMG.Envelope{1,i});

    % MVC correction 
    MVC_EMG.Envelope{1,i} = MVC_EMG.Envelope{1,i} - muscle_baseline(i);
    
    % Configs correction
    for j=1:length(configs_EMG)
        configs_EMG(j).Envelope{1,i} = (configs_EMG(j).Envelope{1,i} ...
            - muscle_baseline(i)) / muscle_MVC(i);
        % Set neg values to 0
        configs_EMG(j).Envelope{1,i}(configs_EMG(j).Envelope{1,i} < 0) = 0;
    end
end

for i=1:nb_EMG
    figure; hold on
    plot(MVC_EMG.Time{1,i}, abs(MVC_EMG.Data{1,i}))
    plot(MVC_EMG.Time{1,i}, MVC_EMG.Envelope{1,i})
end

for i=1:nb_EMG
    figure; hold on
    plot(configs_EMG(1).Time{1,i}, abs(configs_EMG(1).Data{1,i}))
    plot(configs_EMG(1).Time{1,i}, configs_EMG(1).Envelope{1,i})
end

% MVC computation
% muscle_MVC(i) = max(MVC_EMG.Envelope{1,i});
%% Visualization

% Plot baseline for each EMG channel (i.e. each muscle)
% figure
% for i=1:nb_EMG
%     subplot(nb_EMG, 1, i)
%     plot(baseline_EMG.Time{1,i}, abs(baseline_EMG.Data{1,i}))
%     hold on;
%     plot(baseline_EMG.Time{1,i}, abs(baseline_EMG.Envelope{1,i}))
% end

% Plot MVC for each EMG channel (i.e. each muscle)
% figure
% for i=1:nb_EMG
%     subplot(nb_EMG, 1, i)
%     plot(MVC_EMG.Time{1,i}, abs(MVC_EMG.Data{1,i}))
%     hold on;
%     plot(MVC_EMG.Time{1,i}, MVC_EMG.Envelope{1,i})
% end
for i=1:nb_EMG
    figure
    subplot(3,1,1)
    plot(MVC_EMG.Time{1,i}, abs(MVC_EMG.Data{1,i}))
    subplot(3,1,2)
    plot(MVC_EMG.Time{1,i}, abs(MVC_EMG.DataBis{1,i}))
    subplot(3,1,3)
    plot(MVC_EMG.Time{1,i}, MVC_EMG.Envelope{1,i})
    title(MVC_EMG.Channels(i))
end

% Plot EMG of config1
figure
for i=1:nb_EMG
    subplot(nb_EMG+1, 1, i)
    plot(configs_EMG(2).Time{1,i}, abs(configs_EMG(2).Data{1,i}))
    hold on;
    plot(configs_EMG(2).Time{1,i}, configs_EMG(2).Envelope{1,i})
end
subplot(nb_EMG+1, 1, 9)
plot(configs(2).Time{1,40}, configs(2).Data{1,40})

%% Statistics

activation = struct('Channels', {}, 'Data', {});
activity_threshold = 0.05;

% Extract stat of each config
for i=1:length(configs)
    % Compute transition times of trigger
    d = diff(configs(i).Data{1,end});
    rise_times = configs(i).Time{1,end}(circshift(d==1, 1));
    fall_times = configs(i).Time{1,end}(d==-1);
    
    perc = zeros(nb_EMG, length(rise_times));
    for j=1:length(rise_times)
        for k=1:nb_EMG
            [m1, t1_idx] = ...
                min(abs(configs_EMG(i).Time{1,k} - rise_times(j)));
            [m2, t2_idx] = ...
                min(abs(configs_EMG(i).Time{1,k} - fall_times(j)));
            
%             useful_signal = configs_EMG(i).Data{1,k}(t1_idx:t2_idx);
%             perc(k, j) = 100 * mean(useful_signal(useful_signal>activity_threshold));
            
            perc(k, j) = 100 * max(configs_EMG(i).Envelope{1,k}(t1_idx:t2_idx));
        end
    end
    rowDist = ones(1, nb_EMG);
    activation(i) = struct('Channels', configs_EMG(i).Channels, ...
        'Data', perc);
end

% Example of visualization (one of each config)
% for i=1:length(configs_EMG)
%     figure
%     bar(categorical(activation(i).Channels), activation(i).Data(:,4))
% end

%% Visualization of stat

config_nb = 2;

for i=1:length(activation(config_nb).Data(1,:))
    figure;
    bar(categorical(activation(config_nb).Channels), ...
        activation(config_nb).Data(:,i))
    ylim([0 100])
    ylabel('Percentage of activation')
    title('Percentage of activation of each muscle for configuration')
end

for i=1:nb_EMG
    figure; hold on;
    plot(configs_EMG(config_nb).Time{1,i}, ...
        abs(configs_EMG(config_nb).Data{1,i}))
    plot(configs_EMG(config_nb).Time{1,i}, ...
        configs_EMG(config_nb).Envelope{1,i})
    title(configs_EMG(config_nb).Channels(i))
end

% Older trials
% movAvg = dsp.MovingAverage(666);
% movRMS = dsp.MovingRMS(666);
% emg_avg = movAvg(abs(configs_EMG(1).Data{1,1}));
% emg_rms = movRMS(abs(configs_EMG(1).Data{1,1}));
% figure(1); plot(configs_EMG(1).Time{1,1}, abs(configs_EMG(1).Data{1,1}))
% figure(2); plot(configs_EMG(1).Time{1,1}, emg_avg)
% figure(3); plot(configs_EMG(1).Time{1,1}, emg_rms)
% figure(4); hold on
% plot(configs_EMG(1).Time{1,1}, abs(configs_EMG(1).Data{1,1}))
% plot(configs_EMG(1).Time{1,1}, emg_avg)
% plot(configs_EMG(1).Time{1,1}, emg_rms)

% envelope(abs(configs_EMG(1).Data{1,1}),666,'rms') ?

% Plan : 
% - apply same preprocessing to each recording
% - compute baseline
% - remove baseline 
% - compute MVC (mean amplitude of the highest signal portion with
% e.g. 500 ms duration)
% - scale data from configs by VMC
% - create epochs using the trigger channel
% - quantify the amplitude of the signal of each contraction and build
% stats for each configuration