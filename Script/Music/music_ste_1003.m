clear
close all
clc
 
% Load all data
load 'data/music_kernels.mat'
load 'data/music_triplets.mat'
load 'data/music_labels.mat'
addpath(genpath('drtoolbox'));
% Relabel triplets in [1...N]
no_triplets = size(triplets, 1);
[included, ~, triplets] = unique(triplets(:));
triplets = reshape(triplets, [no_triplets 3]);
names = names(included);
% remove artists without triplets
label_matrix = label_matrix(included,:);
% Label artists according to superclasses
super_classes = {'rock', 'metal', 'pop', 'dance', 'hiphop', 'jazz', 'country', 'reggae'};
super_class_list = [5 1 1 1 1 1 1 1 5 3 3 3 3 2 1 1 4 4 4 1 4 7 7 4 1 2 2 1 5 6 6 6 8 1 3 4 4 1 3 1 2 2 1 7 7 1 1 1 1 1 4 5 2 3 6 2 2 5];
new_label_matrix = zeros(size(label_matrix, 1), length(super_classes));
for k=1:length(super_classes)
    new_label_matrix(:,k) = any(label_matrix(:,super_class_list == k) == 1, 2);
end
label_matrix = new_label_matrix;
labels = (length(super_classes) + 1) .* ones(size(label_matrix, 1), 1);
for i=1:size(label_matrix, 1)
    if any(label_matrix(i,:) == 1)
        labels(i) = find(label_matrix(i,:) == 1, 1, 'first');
    end
end
super_classes{end + 1} = 'other';
 
% Initialize some variables for experiments
no_folds = 50;
no_dims = 9;
N = length(included);
num_triplets = size(triplets, 1);
no_train = floor(0.8*num_triplets);
no_test = num_triplets-no_train;
%techniques = {'ste_kernel', 'ste_batch', 'ste_sgd', 'ste_svrg', 'ste_svrg_bb', 'ste_svrg_sbb_0', 'ste_svrg_sbb_epsilon'};

techniques = {'ste_kernel'};

delta = 0.05;
alpha = no_dims-1;
svrg_iter  = 1000;
no_repeat = 1;
eta = 0.0001;
eta_par = 0.01;
scheduling = 1;
error_type = 1;

Predict_X = zeros(length(techniques), no_folds, N, no_dims);
train_errors = zeros(length(techniques), no_folds, 1+svrg_iter);
test_errors = zeros(length(techniques), no_folds, 1+svrg_iter);
run_time = zeros(length(techniques), no_folds, svrg_iter);
eta_seq = zeros(2, no_folds, svrg_iter);
 
% Loop over folds
for t = 1:no_folds
    rng(t);
    % Split triplets into training and test data
    train_ind = randsample(num_triplets, no_train);
    test_ind = setdiff(1:num_triplets, train_ind);
    train_triplets = triplets(train_ind, :);
    test_triplets = triplets(test_ind, :);
    train_triplets_stoch = train_triplets - 1;
    test_triplets_stoch = test_triplets - 1;
    train_triplets_batch = train_triplets;
    test_triplets_batch = test_triplets;
    frq_iter = no_repeat*no_train;
    sgd_iter = (no_repeat+2)*svrg_iter;
    batch_iter = sgd_iter;
    
    X_int = rand(N, no_dims);
    % mappedX = ste_kernel(X_int, train_triplets_batch, test_triplets_batch, no_dims, eta, no_repeat, 30, 10, 0);
    % [V, L] = eig(mappedX*mappedX');
    % [L, ind] = sort(diag(L), 'descend');
    % L = diag(L);
    % V = V(ind, :);
    % X_int = bsxfun(@times, sqrt(diag(L(1:no_dims, 1:no_dims)))', V(:,1:no_dims));
    sum_X = sum(X_int .^ 2, 2);
    D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X_int * X_int')));
 
    no_train_viol = sum(D(sub2ind([N N], train_triplets(:, 1), train_triplets(:, 2))) > ...
            D(sub2ind([N N], train_triplets(:, 1), train_triplets(:, 3))));
    no_test_viol = sum(D(sub2ind([N N], test_triplets(:, 1), test_triplets(:, 2))) > ...
            D(sub2ind([N N], test_triplets(:, 1), test_triplets(:, 3))));
    train_errors(:, t, 1) = no_train_viol/no_train;
    test_errors(:, t, 1) = no_test_viol/no_test;
 
    % Loop over techniques
    for k = 1:length(techniques)
        % Perform kernel learning only once
        switch techniques{k}
            case 'ste_sgd'
                tic
                [mappedX, train_errors(k, t, 2:end), test_errors(k, t, 2:end), run_time(k, t, :)] = ste_sgd_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, N, no_dims, ...
                    no_train, no_test, 5*eta, eta_par, scheduling, sgd_iter, svrg_iter, error_type);
                toc
            case 'ste_svrg'
                tic
                [mappedX, train_errors(k, t, 2:end), test_errors(k, t, 2:end), run_time(k, t, :)] = ste_svrg_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, N, no_dims, ...
                    no_train, no_test, 15*eta, frq_iter, svrg_iter, error_type);
                toc
            case 'ste_batch'
                tic
                [mappedX, train_errors(k, t, 2:end), test_errors(k, t, 2:end), run_time(k, t, :)]= ste_x_epoch_time(X_int, train_triplets_batch, test_triplets_batch, no_dims, 10*eta, no_repeat, ...
                    batch_iter, svrg_iter);
                toc
            case 'ste_kernel'
                tic
                [mappedX, train_errors(k, t, 2:end), test_errors(k, t, 2:end), run_time(k, t, :), function_value(1, t, :)]= ste_kernel(X_int, train_triplets_batch, test_triplets_batch, no_dims, 75*eta, no_repeat, ...
                    batch_iter, svrg_iter);
                toc
            case 'ste_svrg_bb'
                tic
                [mappedX, train_errors(k, t, 2:end), test_errors(k, t, 2:end), run_time(k, t, :), eta_seq(1, t, :)] = ste_svrg_bb_epsilon_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, N, no_dims, ...
                    no_train, no_test, 15*eta, 0, 0, frq_iter, svrg_iter, error_type);
                toc
            case 'ste_svrg_sbb_0'
                tic
                [mappedX, train_errors(k, t, 2:end), test_errors(k, t, 2:end), run_time(k, t, :), eta_seq(2, t, :)] = ste_svrg_bb_epsilon_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, N, no_dims, ...
                    no_train, no_test, 15*eta, 0, 2, frq_iter, svrg_iter, error_type);
                toc
            case 'ste_svrg_sbb_epsilon'
                tic
                [mappedX, train_errors(k, t, 2:end), test_errors(k, t, 2:end), run_time(k, t, :), eta_seq(3, t, :)] = ste_svrg_bb_epsilon_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, N, no_dims, ...
                    no_train, no_test, 15*eta, 2e-2, 2, frq_iter, svrg_iter, error_type);
                toc
        end
        Predict_X(k, t, :, :) = mappedX;
    end
end
 
%lineColor = linspecer(length(techniques));
lineColor = linspecer(6);
x = 1:3:3*(svrg_iter+1);
figure(1);
t = quantile(squeeze(test_errors(1, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(6,:), 'LineStyle', '-' , 'Marker', '+', 'LineWidth', 0.5);hold on
t = quantile(squeeze(test_errors(2, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(5,:), 'LineStyle', '--', 'Marker', 'o', 'LineWidth', 0.5);hold on
t = quantile(squeeze(test_errors(3, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(4,:), 'LineStyle', ':' , 'Marker', '*', 'LineWidth', 0.5);hold on
t = quantile(squeeze(test_errors(4, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(3,:), 'LineStyle', '-.', 'Marker', 'x', 'LineWidth', 0.5);hold on
%t = quantile(squeeze(test_errors(5, :, 1:end)),[0.25,0.5,0.75]);
%y = t(2, :);
%error_L=t(2, :)-t(1, :);
%error_H=t(3, :)-t(2, :);
%errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(3,:), 'LineStyle', '-' , 'Marker', '^', 'LineWidth', 0.5);hold on
t = quantile(squeeze(test_errors(5, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(2,:), 'LineStyle', '--', 'Marker', 'v', 'LineWidth', 0.5);hold on
t = quantile(squeeze(test_errors(6, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(1,:), 'LineStyle', ':' , 'Marker', 's', 'LineWidth', 0.5);
%legend('ProjGD', 'FGD', 'SFGD', 'SVRG', 'SVRG-BB', 'SVRG-SBB_0', 'SVRG-SBB_{0.02}');
legend('ProjGD', 'FGD', 'SFGD', 'SVRG', 'SVRG-SBB_0', 'SVRG-SBB_{0.02}');
xlabel('Epoch Number');
ylabel('Test Error');
ylim([0.15 0.55]);
xlim([0 3*svrg_iter+10]);
 
figure(2);
t = quantile(squeeze(train_errors(1, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(7,:), 'LineStyle', '-' , 'Marker', '+', 'LineWidth', 0.5);hold on
t = quantile(squeeze(train_errors(2, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(6,:), 'LineStyle', '--', 'Marker', 'o', 'LineWidth', 1.5);hold on
t = quantile(squeeze(train_errors(3, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(5,:), 'LineStyle', ':' , 'Marker', '*', 'LineWidth', 1.5);hold on
t = quantile(squeeze(train_errors(4, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(4,:), 'LineStyle', '-.', 'Marker', 'x', 'LineWidth', 1.5);hold on
t = quantile(squeeze(train_errors(5, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(3,:), 'LineStyle', '-' , 'Marker', '^', 'LineWidth', 1.5);hold on
t = quantile(squeeze(train_errors(6, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(2,:), 'LineStyle', '--', 'Marker', 'v', 'LineWidth', 1.5);hold on
t = quantile(squeeze(train_errors(7, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(1,:), 'LineStyle', ':' , 'Marker', 's', 'LineWidth', 1.5);
legend('ProjGD', 'FGD', 'SFGD', 'SVRG', 'SVRG-BB', 'SVRG-SBB_0', 'SVRG-SBB_{0.02}');
xlabel('No. of epochs');
ylabel('Train Error');
ylim([0 0.55]);
xlim([0 3*svrg_iter+10]);
 
figure(3);
x = 1:3:3*(svrg_iter);
t = quantile(squeeze(eta_seq(1, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(3,:), 'LineStyle', '-' , 'Marker', '^', 'LineWidth', 1.5);hold on
t = quantile(squeeze(eta_seq(2, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(2,:), 'LineStyle', '--', 'Marker', 'v', 'LineWidth', 1.5);hold on
t = quantile(squeeze(eta_seq(3, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(1,:), 'LineStyle', ':' , 'Marker', 's', 'LineWidth', 1.5);
legend('SVRG-BB', 'SVRG-SBB_0', 'SVRG-SBB_{0.02}');
xlabel('No. of epochs');
ylabel('Step Size');
xlim([0 3*svrg_iter+10]);
set(gca, 'FontName', 'Arial','FontSize', 16);
set(findall(gcf,'type','text'), 'FontName', 'Arial', 'FontSize', 16);
 
for k = 1:length(techniques)
    tt = zeros(1, no_folds);
    [row, col] = find(squeeze(test_errors(k, :, 2:end))>=0.15);
    A = squeeze(run_time(k,:,:));
    for n = 1:length(row)
        tt(row(n)) = tt(row(n)) + A(row(n), col(n));
    end
    disp('Min:');
    disp(min(tt));
    disp('Max:');
    disp(max(tt));
    disp('Mean:');
    disp(mean(tt));
    disp('Median:');
    disp(median(tt));
    disp('Standard Deviation');
    disp(std(tt));
end
