clear all
close all


%% Data loading

% Dataset names
fname_day1(1) = "data/Day 1/config1freq10.mat";
fname_day1(2) = "data/Day 1/config1freq50.mat";
fname_day1(3) = "data/Day 1/config2freq50.mat";
fname_day1(4) = "data/Day 1/config3freq50.mat";
fname_day1(5) = "data/Day 1/config4freq10.mat";
fname_day1(6) = "data/Day 1/config5freq10.mat";

fname_day2(1) = "data/Day 2/config6freq10.mat";
fname_day2(2) = "data/Day 2/config7freq10.mat";
fname_day2(3) = "data/Day 2/config8freq10.mat";

fname_day3(1) = "data/Day 3/config9freq70.mat";
fname_day3(2) = "data/Day 3/config10freq10.mat";
fname_day3(3) = "data/Day 3/config11freq10.mat";
fname_day3(4) = "data/Day 3/config12freq10.mat";

fname_sys1 = "data/EMG healthy participant/System_EMG_1.mat";
fname_sys2 = "data/EMG healthy participant/System_EMG_2.mat";

% Set indices of the different sensors
EMG_idx_day1 = [1, 2, 3, 4, 5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82];
EMG_idx_day2 = [1, 8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85, 92, 99, 106];
EMG_idx_day3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
EMG_idx_sys1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 23, 30, 37, 44, 51, 62];
EMG_idx_sys2 = [1, 8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 91, 92];

nb_EMG_day1 = length(EMG_idx_day1) - 1;
nb_EMG_day2 = length(EMG_idx_day2) - 1;
nb_EMG_day3 = length(EMG_idx_day3) - 2; % EMG for stim artifacts not taken into account
nb_EMG_sys1 = length(EMG_idx_sys1) - 1;
nb_EMG_sys2 = length(EMG_idx_sys2) - 2;

% Assign channel name to the muscle it is recording 
Channels_day1 = ["FCR", "APL", "the", "FD2", "FD34", "FCU", "ED", "BR", ...
    "BB", "TB", "DELant", "DELmed", "DELpost", "PM", "Trap", "TRIG"];

Channels_day2 = ["FCR", "FCU", "ED", "FD34", "Brach", "BB", "TB", ...
    "DELant", "DELmed", "DELpost", "BR", "PM", "Supra", "Infra", "Rh" "TRIG"];

Channels_day3 = ["ECR", "BR", "FD34", "ED", "DELant", "DELmed", ...
    "DELpost", "BB", "TB", "PM", "LevScap", "FCU", "STIM_ARTEFACTS", "TRIG"];

Channels_sys1 = ["The", "FPL", "APL", "FD2", "Brach", "FCR", "ECR", ...
    "PT", "FD34", "Brach", "FCU", "ECU", "ED", "BBlg", "LatDor", "IMU_sync"];

Channels_sys2 = ["BBsh", "TBlg", "Tblat", "DELant", "DELmed", ...
    "DELpost", "PMcl", "PMst", "Infra", "Supra", "Trhigh", "Trmed", "IMU_sync", "TRIG_sync"];

% Init configs struct
cfgs_EMG_day1 = struct('Channels', {}, 'Data', {}, 'Time', {}, 'Fs', {});
cfgs_EMG_day2 = struct('Channels', {}, 'Data', {}, 'Time', {}, 'Fs', {});
cfgs_EMG_day3 = struct('Channels', {}, 'Data', {}, 'Time', {}, 'Fs', {});

% Load day1
for i=1:length(fname_day1)
    cfg = load(fname_day1(i));
    cfgs_EMG_day1(i) = struct('Channels', Channels_day1, ...
        'Data', mat2cell(cfg.Data(1, EMG_idx_day1), 1, length(EMG_idx_day1)), ...
        'Time', mat2cell(cfg.Time(1, EMG_idx_day1), 1, length(EMG_idx_day1)), ...
        'Fs', mat2cell(cfg.Fs(1, EMG_idx_day1), 1, length(EMG_idx_day1)));
end

% Load day2
for i=1:length(fname_day2)
    cfg = load(fname_day2(i));
    cfgs_EMG_day2(i) = struct('Channels', Channels_day2, ...
        'Data', mat2cell(cfg.Data(1, EMG_idx_day2), 1, length(EMG_idx_day2)), ...
        'Time', mat2cell(cfg.Time(1, EMG_idx_day2), 1, length(EMG_idx_day2)), ...
        'Fs', mat2cell(cfg.Fs(1, EMG_idx_day2), 1, length(EMG_idx_day2)));
end

% Load day3
for i=1:length(fname_day3)
    cfg = load(fname_day3(i));
    cfgs_EMG_day3(i) = struct('Channels', Channels_day3, ...
        'Data', mat2cell(cfg.Data(1, EMG_idx_day3), 1, length(EMG_idx_day3)), ...
        'Time', mat2cell(cfg.Time(1, EMG_idx_day3), 1, length(EMG_idx_day3)), ...
        'Fs', mat2cell(cfg.Fs(1, EMG_idx_day3), 1, length(EMG_idx_day3)));
end

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

%% Synchro

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
% Day1
for i=1:nb_EMG_day1
    for j=1:length(cfgs_EMG_day1)
        cfgs_EMG_day1(j).DataRect{1,i} = abs(cfgs_EMG_day1(j).Data{1,i} - ...
            mean(cfgs_EMG_day1(j).Data{1,i}));
    end
end

% Day2
for i=1:nb_EMG_day2
    for j=1:length(cfgs_EMG_day2)
        cfgs_EMG_day2(j).DataRect{1,i} = abs(cfgs_EMG_day2(j).Data{1,i} - ...
            mean(cfgs_EMG_day2(j).Data{1,i}));
    end
end

% Day3
for i=1:nb_EMG_day3
    for j=1:length(cfgs_EMG_day3)
        cfgs_EMG_day3(j).DataRect{1,i} = abs(cfgs_EMG_day3(j).Data{1,i} - ...
            mean(cfgs_EMG_day3(j).Data{1,i}));
    end
end

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

% Day1
for i=1:nb_EMG_day1
    for j=1:length(cfgs_EMG_day1)
        % Prepare butterworth and window size
        % 300ms window for moving rms
        win = round(300e-3 * cfgs_EMG_day1(j).Fs(i));
        % Set butterworth filter
        Wn = fc / (cfgs_EMG_day1(j).Fs(i)/2);
        [b, a] = butter(order, Wn);

        cfgs_EMG_day1(j).butter{1,i} = ...
            filtfilt(b, a, cfgs_EMG_day1(j).DataRect{1,i});
        cfgs_EMG_day1(j).movRMS{1,i} = ...
            fastrms(cfgs_EMG_day1(j).butter{1,i}, win);
    end
end

% Day2
for i=1:nb_EMG_day2
    for j=1:length(cfgs_EMG_day2)
        % Prepare butterworth and window size
        % 300ms window for moving rms
        win = round(300e-3 * cfgs_EMG_day2(j).Fs(i));
        % Set butterworth filter
        Wn = fc / (cfgs_EMG_day2(j).Fs(i)/2);
        [b, a] = butter(order, Wn);

        cfgs_EMG_day2(j).butter{1,i} = ...
            filtfilt(b, a, cfgs_EMG_day2(j).DataRect{1,i});
        cfgs_EMG_day2(j).movRMS{1,i} = ...
            fastrms(cfgs_EMG_day2(j).butter{1,i}, win);
    end
end

% Day3
for i=1:nb_EMG_day3
    for j=1:length(cfgs_EMG_day3)
        % Prepare butterworth and window size
        % 300ms window for moving rms
        win = round(300e-3 * cfgs_EMG_day3(j).Fs(i));
        % Set butterworth filter
        Wn = fc / (cfgs_EMG_day3(j).Fs(i)/2);
        [b, a] = butter(order, Wn);

        cfgs_EMG_day3(j).butter{1,i} = ...
            filtfilt(b, a, cfgs_EMG_day3(j).DataRect{1,i});
        cfgs_EMG_day3(j).movRMS{1,i} = ...
            fastrms(cfgs_EMG_day3(j).butter{1,i}, win);
    end
end

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

%% 

for i=1:nb_EMG_sys1
    figure; hold on;
    plot(cfgs_EMG_sys1.Time{1,i}, cfgs_EMG_sys1.DataRect{1,i})
    plot(cfgs_EMG_sys1.Time{1,i}, cfgs_EMG_sys1.movRMS{1,i})
end

for i=1:nb_EMG_sys2
    figure; hold on;
    plot(cfgs_EMG_sys2.Time{1,i}, cfgs_EMG_sys2.DataRect{1,i})
    plot(cfgs_EMG_sys2.Time{1,i}, cfgs_EMG_sys2.movRMS{1,i})
end

%%

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

%%

for i=1:nb_EMG_sys1
    figure; hold on;
    plot(cfgs_EMG_sys1.Time{1,i}, cfgs_EMG_sys1.DataNorm{1,i})
end

for i=1:nb_EMG_sys2
    figure; hold on;
    plot(cfgs_EMG_sys2.Time{1,i}, cfgs_EMG_sys2.DataNorm{1,i})
end


%%

% Find max freq
% freq_max = max([max(1./diff()), max(1./diff()), max(1./diff())]);
% new_freq = ceil(freq_max);

% Upsample to same freq
% [,]=resampleT(,new_freq,);
% [,]=resampleT(,new_freq,);
% [,]=resampleT(,new_freq,);
