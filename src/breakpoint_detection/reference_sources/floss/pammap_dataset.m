%%
close all
clear
clc

%%
data = importdata('.\subject3.dat');

label = data(:, 2);

% data = data(:, 33);
data = data(:, 34);
% data = data(:, 35);
data = data + randn(size(data))*0.001;

sub_len = 65;
win_len = 2000;

%%
data_len = length(data);
exc_zone = round(sub_len / 2);
n_win = data_len - win_len + 1;
pro_len = win_len - sub_len + 1;
iac = getAsymIAC(pro_len);
mid_idx = round(win_len / 2);

%%
crc_final = inf(data_len, 1);
update_time = zeros(n_win, 1);
for i = 1:n_win
    timer = tic();
    if i == 1
        [mat_pro, pro_idx] = stamp1DirSelf(data(i:i+win_len-1), sub_len);
    else
        [mat_pro, pro_idx] = stamp1DirSelfSlide(data(i:i+win_len-1), ...
            sub_len, mat_pro, pro_idx);
    end
    arc = getArcCurve(pro_idx(1:end-exc_zone-1), win_len-sub_len+1);
    crc = arc./iac;
    crc = min(1, crc);

    crc_final(mid_idx + i) = crc(mid_idx);
    update_time(i) = toc(timer);
    fprintf('iter %d/%d, %0.5f\n', i, n_win, update_time(i))
end

%%
dict_id = [10, 11, 12, 13, 16, 22, 23];
dict = {
    'walking very slow';
    'normal walking';
    'Nordic walking';
    'running';
    'cycling';
    'rope jumping';
    'playing soccer';
    };
dictTxt = {
    'ws';
    'nm';
    'nd';
    'ru';
    'cy';
    'rj';
    'so';
    };
pos_st = cell(length(dict_id), 1);
pos_ed = cell(length(dict_id), 1);
for i = 1:length(dict_id)
    label_1class = label == dict_id(i);
    pos_st{i} = find((label_1class(1:end-1) - label_1class(2:end)) == -1) + 1;
    pos_ed{i} = find((label_1class(1:end-1) - label_1class(2:end)) == 1);
end

%%
k = 3;
idx = pick_k_minimum(crc_final, k, 5*sub_len);

data_plot = data - min(data);
data_plot = data_plot / max(data_plot);

colors = jet(length(pos_st));

figure;
plot(1:length(data_plot), data_plot + 1.7, 'b');
hold on;
rectangle('position', [1, 1.1, length(data_plot), 0.5], ...
    'edgecolor', 'k', 'facecolor', 'k');
for i = 1:length(pos_st)
    for j = 1:length(pos_st{i})
        rectangle('position', ...
            [pos_st{i}(j), 1.1, pos_ed{i}(j)-pos_st{i}(j), 0.5], ...
            'edgecolor', colors(i,:), 'facecolor', colors(i,:));
        text(pos_st{i}(j)+(pos_ed{i}(j)-pos_st{i}(j))*0.5-4000, ...
            1.35,dictTxt{i}, 'color', 'k')
    end
end
plot(crc_final, 'r');
for i = 1:k
    line([idx(i), idx(i)], [-0.1, 2.8], 'color', [0.5,0.5,0.5])
end
hold off;
ylim([-0.1, 2.8]);
xlim([0, length(data_plot)]);
