clear all
close all

%% Data loading

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

%% Rectification

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

%% Smoothing

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
    MVC_EMG.butter{1,i} = filtfilt(b, a, MVC_EMG.DataRect{1,i});
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

%% Normalization

for i=1:length(configs_EMG)
    for j=1:nb_EMG
        configs_EMG(i).DataNorm{1,j} = ...
            fastrms(configs_EMG(i).butter{1,j}, win) / muscle_MVC(j);
    end
end

%% Statistics

activation = struct('Channels', {}, 'Data', {});
activation2 = struct('Channels', {}, 'Data', {});
activity_threshold = 0.1;

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
            
            useful_signal = configs_EMG(i).DataNorm{1,k}(t1_idx:t2_idx);
            perc2(k, j) = 100 * mean(useful_signal(useful_signal>activity_threshold));
            
            perc(k, j) = 100 * max(configs_EMG(i).DataNorm{1,k}(t1_idx:t2_idx));
        end
    end
    
    activation(i) = struct('Channels', configs_EMG(i).Channels, ...
        'Data', perc);
    activation2(i) = struct('Channels', configs_EMG(i).Channels, ...
        'Data', perc2);
end

%% Visualization

% for i=1:nb_EMG
%     figure; hold on
%     plot(MVC_EMG.Time{1,i}, MVC_EMG.DataRect{1,i})
%     plot(MVC_EMG.Time{1,i}, MVC_EMG.butter{1,i})
%     plot(MVC_EMG.Time{1,i}, MVC_EMG.movRMS{1,i})
% end
config_nb = 7;

% Plot percentages of activation
for i=1:length(activation(config_nb).Data(1,:))
    figure;
    subplot(2,1,1)
    bar(categorical(activation(config_nb).Channels), ...
        activation(config_nb).Data(:,i))
    ylim([0 100])
    ylabel('Percentage of activation')
    title('Percentage of activation - STIM ', num2str(i))
    subplot(2,1,2)
    bar(categorical(activation2(config_nb).Channels), ...
        activation2(config_nb).Data(:,i))
    ylim([0 100])
    ylabel('Percentage of activation')
    title('Percentage of activation - STIM ', num2str(i))
end

% Verif
% for i=1:nb_EMG
%     figure; hold on
%     plot(configs_EMG(config_nb).Time{1,i}, configs_EMG(config_nb).DataRect{1,i})
%     plot(configs_EMG(config_nb).Time{1,i}, configs_EMG(config_nb).movRMS{1,i})
%     title(configs_EMG(config_nb).Channels(i))
% end