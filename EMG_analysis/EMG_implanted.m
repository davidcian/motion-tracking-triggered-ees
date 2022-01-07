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

cfg_idx = 1;

for i=1:nb_EMG_day1
    figure; 
    subplot(2,1,1);
    hold on;
    plot(cfgs_EMG_day1(cfg_idx).Time{1,i}, cfgs_EMG_day1(cfg_idx).DataRect{1,i})
    plot(cfgs_EMG_day1(cfg_idx).Time{1,i}, cfgs_EMG_day1(cfg_idx).movRMS{1,i})
    title(cfgs_EMG_day1(cfg_idx).Channels(i))

    subplot(2,1,2)
    plot(cfgs_EMG_day1(cfg_idx).Time{1,end}, cfgs_EMG_day1(cfg_idx).Data{1,end})

    figure;
    subplot(2,1,1);
    hold on;
    plot(cfgs_EMG_day1(cfg_idx+1).Time{1,i}, cfgs_EMG_day1(cfg_idx+1).DataRect{1,i})
    plot(cfgs_EMG_day1(cfg_idx+1).Time{1,i}, cfgs_EMG_day1(cfg_idx+1).movRMS{1,i})
    title(cfgs_EMG_day1(cfg_idx+1).Channels(i))

    subplot(2,1,2)
    plot(cfgs_EMG_day1(cfg_idx+1).Time{1,end}, cfgs_EMG_day1(cfg_idx+1).Data{1,end})
end

% for i=1:nb_EMG_day2
%     figure;
%     subplot(2,1,1);
%     hold on;
%     plot(cfgs_EMG_day2(cfg_idx).Time{1,i}, cfgs_EMG_day2(cfg_idx).DataRect{1,i})
%     plot(cfgs_EMG_day2(cfg_idx).Time{1,i}, cfgs_EMG_day2(cfg_idx).movRMS{1,i})
%     title(cfgs_EMG_day2(cfg_idx).Channels(i))
% 
%     subplot(2,1,2)
%     plot(cfgs_EMG_day2(cfg_idx).Time{1,end}, cfgs_EMG_day2(cfg_idx).Data{1,end})
% end

% for i=1:nb_EMG_day3
%     figure; 
%     subplot(2,1,1);
%     hold on;
%     plot(cfgs_EMG_day3(cfg_idx).Time{1,i}, cfgs_EMG_day3(cfg_idx).DataRect{1,i})
%     plot(cfgs_EMG_day3(cfg_idx).Time{1,i}, cfgs_EMG_day3(cfg_idx).movRMS{1,i})
%     title(cfgs_EMG_day3(cfg_idx).Channels(i))
% 
%     subplot(2,1,2)
%     plot(cfgs_EMG_day3(cfg_idx).Time{1,end}, cfgs_EMG_day3(cfg_idx).Data{1,end})
% end

%% Normalization

% Day1
max_day1 = zeros(1, nb_EMG_day1);

for j=1:length(cfgs_EMG_day1)
    trig_idx_activated = cfgs_EMG_day1(j).Data{1,end} < 0;
    t_activated = cfgs_EMG_day1(j).Time{1,end}(trig_idx_activated);
    for i=1:nb_EMG_day1
        mus_idx_activated = round(t_activated * cfgs_EMG_day1(j).Fs(i));
        M = max(cfgs_EMG_day1(j).movRMS{1,i}(mus_idx_activated));
        max_day1(i) = max(max_day1(i), M);

%         cfgs_EMG_day1(j).DataNorm{1,i} = cfgs_EMG_day1(j).movRMS{1,i} / M;
    end
end

for j=1:length(cfgs_EMG_day1)
    for i=1:nb_EMG_day1
        cfgs_EMG_day1(j).DataNorm{1,i} = ...
            cfgs_EMG_day1(j).movRMS{1,i} / max_day1(i);
    end
end

% Day2
max_day2 = zeros(1, nb_EMG_day2);

for j=1:length(cfgs_EMG_day2)
    trig_idx_activated = cfgs_EMG_day2(j).Data{1,end} < 0;
    t_activated = cfgs_EMG_day2(j).Time{1,end}(trig_idx_activated);
    for i=1:nb_EMG_day2
        mus_idx_activated = round(t_activated * cfgs_EMG_day2(j).Fs(i));
        M = max(cfgs_EMG_day2(j).movRMS{1,i}(mus_idx_activated));
        max_day2(i) = max([max_day2(i), M]);

%         cfgs_EMG_day2(j).DataNorm{1,i} = cfgs_EMG_day2(j).movRMS{1,i} / M;
    end
end

for j=1:length(cfgs_EMG_day2)
    for i=1:nb_EMG_day2
        cfgs_EMG_day2(j).DataNorm{1,i} = ...
            cfgs_EMG_day2(j).movRMS{1,i} / max_day2(i);
    end
end

% Day3
max_day3 = zeros(1, nb_EMG_day3);

for j=1:length(cfgs_EMG_day3)
    trig_idx_activated = cfgs_EMG_day3(j).Data{1,end} < 0;
    t_activated = cfgs_EMG_day3(j).Time{1,end}(trig_idx_activated);
    for i=1:nb_EMG_day3
        mus_idx_activated = round(t_activated * cfgs_EMG_day3(j).Fs(i));
        M = max(cfgs_EMG_day3(j).movRMS{1,i}(mus_idx_activated));
        max_day3(i) = max([max_day3(i), M]);

%         cfgs_EMG_day3(j).DataNorm{1,i} = cfgs_EMG_day3(j).movRMS{1,i} / M;
    end
end

for j=1:length(cfgs_EMG_day3)
    for i=1:nb_EMG_day3
        cfgs_EMG_day3(j).DataNorm{1,i} = ...
            cfgs_EMG_day3(j).movRMS{1,i} / max_day3(i);
    end
end


%% Normalized visualization

cfg_idx = 3;

for i=1:nb_EMG_day1
    figure;
    subplot(2,1,1);
    hold on;
    plot(cfgs_EMG_day1(cfg_idx).Time{1,i}, cfgs_EMG_day1(cfg_idx).DataNorm{1,i})
    title(cfgs_EMG_day1(cfg_idx).Channels(i))

    subplot(2,1,2)
    plot(cfgs_EMG_day1(cfg_idx).Time{1,end}, cfgs_EMG_day1(cfg_idx).Data{1,end})
end

%% Statistics

activation_day1 = struct('Channels', {}, 'Data', {}, 'I', {});
activation_day2 = struct('Channels', {}, 'Data', {}, 'I', {});
activation_day3 = struct('Channels', {}, 'Data', {}, 'I', {});

% Extract stat of each config - Day 1
for i=1:length(cfgs_EMG_day1)
    % Compute transition times of trigger
    [~, r_times, ~] = falltime(cfgs_EMG_day1(i).Data{1,end}, ...
                               cfgs_EMG_day1(i).Time{1,end});
    [~, ~, f_times] = risetime(cfgs_EMG_day1(i).Data{1,end}, ...
                               cfgs_EMG_day1(i).Time{1,end});
    
    perc = zeros(nb_EMG_day1, length(r_times));

    for j=1:length(r_times)
        for k=1:nb_EMG_day1
            t1_idx = round(r_times(j) * cfgs_EMG_day1(i).Fs(k));
            t2_idx = round(f_times(j) * cfgs_EMG_day1(i).Fs(k));
            
            useful_signal = cfgs_EMG_day1(i).DataNorm{1,k}(t1_idx:t2_idx);
            perc(k, j) = 100 * max(useful_signal);
        end
    end
    
    activation_day1(i) = struct('Channels', Channels_day1, 'Data', perc, ...
        'I', linspace(1, length(r_times), length(r_times)));
    % Note : I starts from value X (=1) and increases by 1 at each step
end

% Extract stat of each config - Day 2
for i=2:length(cfgs_EMG_day2) %start at 2 because stim 1 beurk
    % Compute transition times of trigger
    [~, r_times, ~] = ...
        falltime(cfgs_EMG_day2(i).Data{1,end}, cfgs_EMG_day2(i).Time{1,end});
    [~, ~, f_times] = ...
        risetime(cfgs_EMG_day2(i).Data{1,end}, cfgs_EMG_day2(i).Time{1,end});
    
    perc = zeros(nb_EMG_day2, length(r_times));

    for j=1:length(r_times)
        for k=1:nb_EMG_day2
            t1_idx = round(r_times(j) * cfgs_EMG_day2(i).Fs(k));
            t2_idx = round(f_times(j) * cfgs_EMG_day2(i).Fs(k));
            
            useful_signal = cfgs_EMG_day2(i).DataNorm{1,k}(t1_idx:t2_idx);
            perc(k, j) = 100 * max(useful_signal);
        end
    end
    
    activation_day2(i) = struct('Channels', Channels_day2, 'Data', perc, ...
        'I', linspace(1, length(r_times), length(r_times)));
    % Note : I starts from value X (=1) and increases by 1 at each step
end

% Extract stat of each config - Day 3
for i=1:length(cfgs_EMG_day3)
    % Compute transition times of trigger
    [~, r_times, ~] = ...
        falltime(cfgs_EMG_day3(i).Data{1,end}, cfgs_EMG_day3(i).Time{1,end});
    [~, ~, f_times] = ...
        risetime(cfgs_EMG_day3(i).Data{1,end}, cfgs_EMG_day3(i).Time{1,end});
    
    perc = zeros(nb_EMG_day3, length(r_times));

    for j=1:length(r_times)
        for k=1:nb_EMG_day3
            t1_idx = round(r_times(j) * cfgs_EMG_day3(i).Fs(k));
            t2_idx = round(f_times(j) * cfgs_EMG_day3(i).Fs(k));
            
            useful_signal = cfgs_EMG_day3(i).DataNorm{1,k}(t1_idx:t2_idx);
            perc(k, j) = 100 * max(useful_signal);
        end
    end
    
    activation_day3(i) = struct('Channels', Channels_day3, 'Data', perc, ...
        'I', linspace(1, length(r_times), length(r_times)));
    % Note : I starts from value X (=1) and increases by 1 at each step
end

%% Stat visualization
config_idx = 4;

% Plot percentages of activation
% for i=1:length(activation_day1(config_idx).Data(1,:))
%     figure;
%     bar(categorical(activation_day1(config_idx).Channels(1:nb_EMG_day1)), ...
%         activation_day1(config_idx).Data(:,i))
%     ylim([0 100])
%     ylabel('Percentage of activation')
%     title('Percentage of activation - STIM ', num2str(i))
% end

for i=1:length(activation_day3(config_idx).Data(1,:))
    figure;
    bar(categorical(activation_day3(config_idx).Channels(1:nb_EMG_day3)), ...
        activation_day3(config_idx).Data(:,i))
    ylim([0 100])
    ylabel('Percentage of activation')
    title('Percentage of activation - STIM ', num2str(i))
end

% for i=1:nb_EMG_day1
%     figure;
%     subplot(2,1,1);
%     hold on;
%     plot(cfgs_EMG_day1(cfg_idx).Time{1,i}, cfgs_EMG_day1(cfg_idx).DataNorm{1,i})
%     title(cfgs_EMG_day1(cfg_idx).Channels(i))
% 
%     subplot(2,1,2)
%     plot(cfgs_EMG_day1(cfg_idx).Time{1,end}, cfgs_EMG_day1(cfg_idx).Data{1,end})
% end

% figure;
figure;
subplot(2,1,1);
hold on;
box on;
plot(cfgs_EMG_day3(config_idx).Time{1,8}, cfgs_EMG_day3(config_idx).Data{1,8})
% title(cfgs_EMG_day3(config_idx).Channels(8) + " - raw", 'FontSize', 15)
title("Biceps - raw", 'FontSize', 15)
xlabel('Time [s]', 'FontSize', 15)
ylabel('Muscle activity [V]', 'FontSize', 15)
subplot(2,1,2)
box on;
plot(cfgs_EMG_day3(config_idx).Time{1,8}, cfgs_EMG_day3(config_idx).DataNorm{1,8} - 0.13)
% title(cfgs_EMG_day3(config_idx).Channels(8) + " - preprocessed and normalized", 'FontSize', 15)
title("Biceps - preprocessed and normalized", 'FontSize', 15)
xlabel('Time [s]', 'FontSize', 15)
ylabel('Muscle activity [V]', 'FontSize', 15)

%% Write activations in csv files