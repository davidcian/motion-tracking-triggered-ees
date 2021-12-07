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

% Set indices of the different sensors
EMG_idx_day1 = [1, 2, 3, 4, 5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75, 82];
EMG_idx_day2 = [1, 8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85, 92, 99, 106];
EMG_idx_day3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];

nb_EMG_day1 = length(EMG_idx_day1) - 1;
nb_EMG_day2 = length(EMG_idx_day2) - 1;
nb_EMG_day3 = length(EMG_idx_day3) - 2; % EMG for stim artifacts not taken into account

% Assign channel name to the muscle it is recording 
Channels_day1 = ["FCR", "APL", "the", "FD2", "FD34", "FCU", "ED", "BR", ...
    "BB", "TB", "DELant", "DELmed", "DELpost", "PM", "Trap", "TRIG"];

Channels_day2 = ["FCR", "FCU", "ED", "FD34", "Brach", "BB", "TB", ...
    "DELant", "DELmed", "DELpost", "BR", "PM", "Supra", "Infra", "Rh" "TRIG"];

Channels_day3 = ["ECR", "BR", "FD34", "ED", "DELant", "DELmed", ...
    "DELpost", "BB", "TB", "PM", "LevScap", "FCU", "STIM_ARTEFACTS", "TRIG"];

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

%% Envelope visualization 


%% Normalization


%% Normalized visualization


%% Write activations in csv files

