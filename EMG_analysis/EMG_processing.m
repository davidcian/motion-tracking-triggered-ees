clear all
close all

%% Data loading

% Set indices of the different sensors
EMG_idx_day1 = [1, 2, 3, 4, 5, 12, 19, 26, 33, 40, 47, 54, 61, 68, 75];
EMG_idx_day2 = [1, 8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85, 92, 99];
EMG_idx_day3 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13];
EMG_idx_sys1 = [1, 8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85];
EMG_idx_sys2 = [1, 8, 15, 22, 29, 36, 43, 50, 57];
% ACC_idx = [6, 7, 8, 13, 14, 15, 20, 21, 22, 27, 28, 29, 34, 35, 36];
% GYR_idx = [9, 10, 11, 16, 17, 18, 23, 24, 25, 30, 31, 32, 37, 38, 39];

nb_EMG = length(EMG_idx);

% Assign channel name to the muscle it is recording 
Channels_day1 = ["FCR", "APL", "the", "FD2", "FD34", "FCU", "ED", "BR", ...
    "BB", "TB", "DELant", "DELmed", "DELpost", "PM", "Trap"];

Channels_day2 = ["FCR", "FCU", "ED", "FD34", "Brach", "BB", "TB", ...
    "DELant", "DELmed", "DELpost", "BR", "PM", "Supra", "Infra", "Rh"];

Channels_day3 = ["ECR", "BR", "FD34", "ED", "DELant", "DELmed", ...
    "DELpost", "BB", "TB", "PM", "LevScap", "FCU", "STIM_ARTEFACTS"];

Channels_sys1 = ["The", "FPL", "APL", "FD2", "Brach", "FCR", "ECR", ...
    "PT", "FD34", "Brach", "FCU", "ECU", "ED", "BBlg", "LatDor"];

Channels_sys2 = ["BBsh", "TBlg", "Tblat", "DELant", "DELmed", ...
    "DELpost", "PMcl", "PMst", "Infra", "Supra", "Trhigh", "Trmed"];

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

%% 