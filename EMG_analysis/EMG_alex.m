clear all
close all

%% Data loading

% Dataset names
fname_sys1 = "data/EMG healthy participant/System_EMG_1.mat";
fname_sys2 = "data/EMG healthy participant/System_EMG_2.mat";

fname_alex = "data/Alex.mat";

% Set indices of the different sensors
EMG_idx_sys1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 23, 30, 37, 44, 51, 62];
EMG_idx_sys2 = [1, 8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 91, 92];

nb_EMG_sys1 = length(EMG_idx_sys1) - 1;
nb_EMG_sys2 = length(EMG_idx_sys2) - 2;

% Assign channel name to the muscle it is recording
Channels_sys1 = ["The", "FPL", "APL", "FD2", "Brach", "FCR", "ECR", ...
    "PT", "FD34", "Brach", "FCU", "ECU", "ED", "BBlg", "LatDor", ...
    "IMU_sync"];

Channels_sys2 = ["BBsh", "TBlg", "Tblat", "DELant", "DELmed", ...
    "DELpost", "PMcl", "PMst", "Infra", "Supra", "Trhigh", "Trmed", ...
    "IMU_sync", "TRIG_sync"];

% Load sys1
cfg = load(fname_sys1);
cfgs_EMG_sys1 = struct('Channels', Channels_sys1, ...
    'Data', mat2cell(cfg.Data(1, EMG_idx_sys1), 1, length(EMG_idx_sys1)), ...
    'Time', mat2cell(cfg.Time(1, EMG_idx_sys1), 1, length(EMG_idx_sys1)), ...
    'Fs', mat2cell(cfg.Fs(1, EMG_idx_sys1), 1, length(EMG_idx_sys1)));

% Load sys2
cfg = load(fname_sys2);
cfgs_EMG_sys2 = struct('Channels', Channels_sys2, ...
    'Data', mat2cell(cfg.Data(1, EMG_idx_sys2), 1, length(EMG_idx_sys2)), ...
    'Time', mat2cell(cfg.Time(1, EMG_idx_sys2), 1, length(EMG_idx_sys2)), ...
    'Fs', mat2cell(cfg.Fs(1, EMG_idx_sys2), 1, length(EMG_idx_sys2)));

% Load alex data
load(fname_alex); % var name : ALExkinematics
alex = table2array(ALExkinematics);

%% Synchro sys1-sys2

idx = cfgs_EMG_sys2.Time{1, end}(cfgs_EMG_sys2.Data{1, end} > 4);
delay_t = idx(1);

for i=1:nb_EMG_sys2
    delay_s = round(delay_t * cfgs_EMG_sys2.Fs(i));
    cfgs_EMG_sys2.Time{1, i} = cfgs_EMG_sys2.Time{1, i} - delay_t;
    cfgs_EMG_sys2.Time{1, i} = cfgs_EMG_sys2.Time{1, i}(delay_s:end);
    cfgs_EMG_sys2.Data{1, i} = cfgs_EMG_sys2.Data{1, i}(delay_s:end);
end

%% Rectification

% Remove mean to each signal (mean) and rectify
% Sys1
for i=1:nb_EMG_sys1
    cfgs_EMG_sys1.DataRect{1,i} = abs(cfgs_EMG_sys1.Data{1,i} - ...
        mean (cfgs_EMG_sys1.Data{1,i}));
end

% Sys2
for i=1:nb_EMG_sys2
    cfgs_EMG_sys2.DataRect{1,i} = abs(cfgs_EMG_sys2.Data{1,i} - ...
        mean (cfgs_EMG_sys2.Data{1,i}));
end


%% Smoothing

% Set parameters
fc = 100;
order = 6;

% Sys1
for i=1:nb_EMG_sys1
    % Prepare butterworth and window size
    % 300ms window for moving rms
    win = round(300e-3 * cfgs_EMG_sys1.Fs(i));
    % Set butterworth filter
    Wn = fc / (cfgs_EMG_sys1.Fs(i)/2);
    [b, a] = butter(order, Wn);

    cfgs_EMG_sys1.butter{1,i} = ...
        filtfilt(b, a, cfgs_EMG_sys1.DataRect{1,i});
    cfgs_EMG_sys1.movRMS{1,i} = ...
        fastrms(cfgs_EMG_sys1.butter{1,i}, win);
end

% Sys2
for i=1:nb_EMG_sys2
    % Prepare butterworth and window size
    % 300ms window for moving rms
    win = round(300e-3 * cfgs_EMG_sys2.Fs(i));
    % Set butterworth filter
    Wn = fc / (cfgs_EMG_sys2.Fs(i)/2);
    [b, a] = butter(order, Wn);

    cfgs_EMG_sys2.butter{1,i} = ...
        filtfilt(b, a, cfgs_EMG_sys2.DataRect{1,i});
    cfgs_EMG_sys2.movRMS{1,i} = ...
        fastrms(cfgs_EMG_sys2.butter{1,i}, win);
end

%% Envelope visualization 

% for i=1:nb_EMG_sys1
%     figure; hold on;
%     plot(cfgs_EMG_sys1.Time{1,i}, cfgs_EMG_sys1.DataRect{1,i})
%     plot(cfgs_EMG_sys1.Time{1,i}, cfgs_EMG_sys1.movRMS{1,i})
% end
% 
% for i=1:nb_EMG_sys2
%     figure; hold on;
%     plot(cfgs_EMG_sys2.Time{1,i}, cfgs_EMG_sys2.DataRect{1,i})
%     plot(cfgs_EMG_sys2.Time{1,i}, cfgs_EMG_sys2.movRMS{1,i})
% end

%% Normalization

max_sys1 = zeros([1, nb_EMG_sys1]);
for i=1:nb_EMG_sys1
    max_sys1(i) = max(cfgs_EMG_sys1.movRMS{1,i});
    cfgs_EMG_sys1.MVC(i) = max_sys1(i);
    cfgs_EMG_sys1.DataNorm{1,i} = cfgs_EMG_sys1.movRMS{1,i} / max_sys1(i);
end

max_sys2 = zeros([1, nb_EMG_sys2]);
for i=1:nb_EMG_sys2
    max_sys2(i) = max(cfgs_EMG_sys2.movRMS{1,i});
    cfgs_EMG_sys2.MVC(i) = max_sys2(i);
    cfgs_EMG_sys2.DataNorm{1,i} = cfgs_EMG_sys2.movRMS{1,i} / max_sys2(i);
end

%% Normalized visualization

% for i=1:nb_EMG_sys1
%     figure; hold on;
%     plot(cfgs_EMG_sys1.Time{1,i}, cfgs_EMG_sys1.DataNorm{1,i})
% end
% 
% for i=1:nb_EMG_sys2
%     figure; hold on;
%     plot(cfgs_EMG_sys2.Time{1,i}, cfgs_EMG_sys2.DataNorm{1,i})
% end


%% Upsampling

% Find max freq
freq_max = max([max(1./diff(ALExkinematics.timer)), cfgs_EMG_sys1.Fs(8), ...
    cfgs_EMG_sys1.Fs(9)]);
new_freq = ceil(freq_max);

for i=1:nb_EMG_sys1
    [cfgs_EMG_sys1.DataNormUp{1,i}, cfgs_EMG_sys1.TimeUp{1,i}] = ... 
        resampleT(cfgs_EMG_sys1.DataNorm{1,i}, new_freq, cfgs_EMG_sys1.Time{1,i});
end

for i=1:nb_EMG_sys2
    [cfgs_EMG_sys2.DataNormUp{1,i}, cfgs_EMG_sys2.TimeUp{1,i}] = ... 
        resampleT(cfgs_EMG_sys2.DataNorm{1,i}, new_freq, cfgs_EMG_sys2.Time{1,i});
end

alex(:,1) = alex(:,1) - alex(1,1);
[x, t] = resampleT(alex(:,1), new_freq, alex(:,1));
alex_up = zeros(length(x), 9);
for i=2:9
    [alex_up(:,i), ~] = resampleT(alex(:,i), new_freq, alex(:,1));
end

%% Write in files

shortest_len = length(alex_up);
csvwrite('data/alex.csv', alex_up(:, 2:end));

sys1 = zeros(shortest_len, nb_EMG_sys1);
for i=1:nb_EMG_sys1
    if i == 9
        continue
    end
    sys1(:, i) = cfgs_EMG_sys1.DataNormUp{1,i}(1:shortest_len);
end
csvwrite('data/sys1.csv', sys1);

sys2 = zeros(shortest_len, nb_EMG_sys2);
for i=1:nb_EMG_sys2
    if i == 8
        continue
    end
    sys2(:, i) = cfgs_EMG_sys2.DataNormUp{1,i}(1:shortest_len);
end
csvwrite('data/sys2.csv', sys2);

