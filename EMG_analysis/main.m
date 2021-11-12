%% Correction of .mat files (ok)

% for i=1:7
%     fname = sprintf("config%d.mat", i);
%     sname = sprintf("config%d", i);
%     save(fname, '-struct', sname);
% end

%% Loading of EMG

EMG_idx = [1, 2, 3, 5, 12, 19, 26, 33];
nb_EMG = length(EMG_idx);
unused_EMG_idx = 4;
ACC_idx = [6, 7, 8, 13, 14, 15, 20, 21, 22, 27, 28, 29, 34, 35, 36];
GYR_idx = [9, 10, 11, 16, 17, 18, 23, 24, 25, 30, 31, 32, 37, 38, 39];
trigger_idx = 40;

% Load baseline and pick EMG
baseline = load("Run_number_438_baseline_Rep_1.5.mat");
baseline_EMG = struct('Channels', ...
    mat2cell(baseline.Channels(1, EMG_idx), 1, nb_EMG), ...
    'Data', mat2cell(baseline.Data(1, EMG_idx), 1, nb_EMG), ...
    'Time', mat2cell(baseline.Time(1, EMG_idx), 1, nb_EMG), ...
    'Fs', mat2cell(baseline.Fs(1, EMG_idx), 1, nb_EMG));
% Load MVC and pick MVC
MVC = load("Run_number_439_MVC_Rep_1.6.mat");
MVC_EMG = struct('Channels', ...
    mat2cell(MVC.Channels(1, EMG_idx), 1, nb_EMG), ...
    'Data', mat2cell(MVC.Data(1, EMG_idx), 1, nb_EMG), ...
    'Time', mat2cell(MVC.Time(1, EMG_idx), 1, nb_EMG), ...
    'Fs', mat2cell(MVC.Fs(1, EMG_idx), 1, nb_EMG));

% Init config struct
configs = struct('Channels', {}, 'Data', {}, 'Time', {}, 'Fs', {});
configs_EMG = struct('Channels', {}, 'Data', {}, 'Time', {}, 'Fs', {});

for i=1:7
    fname = sprintf("config%d.mat", i);
    cfg = load(fname);
    configs(i) = cfg;
    configs_EMG(i) = struct('Channels', ...
        mat2cell(cfg.Channels(1, EMG_idx), 1, nb_EMG), ...
        'Data', mat2cell(cfg.Data(1, EMG_idx), 1, nb_EMG), ...
        'Time', mat2cell(cfg.Time(1, EMG_idx), 1, nb_EMG), ...
        'Fs', mat2cell(cfg.Fs(1, EMG_idx), 1, nb_EMG));
end

%% Normalization

% Collect baseline for each EMG (and acc, gyr blabla)
muscle_baseline = zeros(1, length(baseline_EMG.Data));
for i=1:nb_EMG
    muscle_baseline(i) = mean(baseline_EMG.Data{1,i});
end

% Plot VMC for each EMG channel (i.e. each muscle)
for i=1:nb_EMG
    subplot(8, 1, i)
    plot(MVC_EMG.Time{1,i}, MVC_EMG.Data{1,i})
end

% Compute max values for each muscle

