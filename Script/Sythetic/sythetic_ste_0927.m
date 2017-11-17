close all;
clear;
clc

N = 100;
no_dims = 10;
means = zeros(1, no_dims);
std_dev = 1/(2*no_dims).*eye(no_dims);
%rng default  % For reproducibility
Ground_X = mvnrnd(means, std_dev, N);
sum_X = sum(Ground_X.^2, 2);
Ground_D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (Ground_X*Ground_X')));
% generate the triple-wise comparison based on the Euclidean distance
triplets = [ ];
j = repmat(1:N-1, 1, N-1);
k = repmat(1:N-1, N-1, 1);
k = k(:)';
jk = [j(j<k); k(j<k)]';
len_jk = size(jk, 1);
for n = 1:N
	temp = jk+(jk>=n);
	triplets = [triplets; [repmat(n, len_jk, 1), temp]];
end
temp = Ground_D(triplets(:, 1)+(triplets(:, 2)-1)*N)-Ground_D(triplets(:, 1)+(triplets(:, 3)-1)*N);
triplets = triplets(temp ~= 0, :);
temp = temp(temp ~= 0);
temp = find(temp>0);
triplets(temp, [2; 3]) = triplets(temp, [3; 2]);
num_triplets = size(triplets, 1);
perm = randperm(num_triplets);
triplets = triplets(perm, :);
%techniques = {'ste_x_sgd', 'ste_x_svrg', 'ste_x_batch', 'ste_k_batch', 'ste_x_sgd', 'ste_x_svrg', 'ste_x_batch', 'ste_k_batch', ...
%	'ckl_x_sgd', 'ckl_x_svrg', 'ckl_x_batch', 'ckl_k_batch', 'tste_x_sgd', 'tste_x_svrg', 'tste_x_batch'};
%techniques = {'ste_kernel', 'ste_rank', 'ste_kernel_bb', 'ste_batch', 'ste_sgd', 'ste_svrg', 'ste_svrg_bb'};
techniques = {'ste_kernel', 'ste_batch', 'ste_sgd', 'ste_svrg', 'ste_svrg_bb', 'ste_svrg_sbb_0', 'ste_svrg_sbb_epsilon'};
num_train = 20000;
num_test = 5000;
num_trial = 50;
alpha = no_dims-1;
delta = 0.05;
svrg_iter = 500;
no_repeat = 1;
eta = 0.0001;
eta_par = 0.01;
scheduling = 1;
error_type = 1;
train_errors = zeros(length(techniques), num_trial, 1+svrg_iter);
test_errors = zeros(length(techniques), num_trial, 1+svrg_iter);
Predict_X = zeros(length(techniques), num_trial, N, no_dims);
run_time = zeros(length(techniques), num_trial, svrg_iter);
function_value = zeros(3, num_trial, svrg_iter);
eta_seq = zeros(3, num_trial, svrg_iter);

label = ones(N, 1);
link_val = zeros(num_train, 1);
prob = zeros(num_train, 1);
labels = ones(N, 1);

epsilon = 2.5e-2;

for t = 1:num_trial
	X_int = rand(N, no_dims);
	%X_int = zeros(N, no_dims);
	sum_X = sum(X_int .^ 2, 2);
	D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X_int * X_int')));
	train_ind = randi(num_triplets, num_train, 1);
	test_ind = true(1, num_triplets);
	test_ind(train_ind) = false;
	train_triplets = triplets(train_ind, :);
	test_triplets = triplets(test_ind, :);
	%test_triplets = triplets(test_ind(1:num_test), :);
	no_train_viol = sum(D(sub2ind([N N], train_triplets(:, 1), train_triplets(:, 2))) > ...
			D(sub2ind([N N], train_triplets(:, 1), train_triplets(:, 3))));
	no_test_viol = sum(D(sub2ind([N N], test_triplets(:, 1), test_triplets(:, 2))) > ...
			D(sub2ind([N N], test_triplets(:, 1), test_triplets(:, 3))));
	no_train = size(train_triplets, 1);
	no_test = size(test_triplets, 1);
	train_errors(:, t, 1) = no_train_viol/no_train;
	test_errors(:, t, 1) = no_test_viol/no_test;
	%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Bradley-Terry-Luce Model 
	%link_train = 1./(1+exp(Ground_D(train_triplets(:, 1)+(train_triplets(:, 2)-1)*N)-Ground_D(train_triplets(:, 1)+(train_triplets(:, 3)-1)*N)));
	%link_test = 1./(1+exp(Ground_D(test_triplets(:, 1)+(test_triplets(:, 2)-1)*N)-Ground_D(test_triplets(:, 1)+(test_triplets(:, 3)-1)*N)));
	%prob_train = rand(num_train, 1);
	%prob_test = rand(num_test, 1);
	%invs_train = find(prob_train>link_train);
	%invs_test = find(prob_test>link_test);
	%train_triplets(invs_train, [2, 3]) = train_triplets(invs_train, [3, 2]);
	%test_triplets(invs_test, [2, 3]) = test_triplets(invs_test, [3, 2]);
	%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Random Noise
	%invs_train = randi(num_train, floor(num_train*noise_ratio), 1);
	%invs_test = randi(num_test, floor(num_train*noise_ratio), 1);
	%train_triplets(invs_train, [2, 3]) = train_triplets(invs_train, [3, 2]);
	%test_triplets(invs_test, [2, 3]) = test_triplets(invs_test, [3, 2]);
	%%%%%%%%%%%%%%%%%%%%%%%%%%
	% Noise-Free
	%%%%%%%%%%%%%%%%%%%%%%%%%%
	train_triplets_stoch = train_triplets - 1;
	test_triplets_stoch = test_triplets - 1;
	train_triplets_batch = train_triplets;
	test_triplets_batch = test_triplets;
	frq_iter = no_repeat*no_train;
	sgd_iter = (no_repeat+2)*svrg_iter;
	batch_iter = sgd_iter;
	for k = 1:length(techniques)
		switch techniques{k}
			case 'ste_sgd'
				tic
				[mappedX, train_errors(k, t, 2:end), test_errors(k, t, 2:end), run_time(k, t, :)] = ste_sgd_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, N, no_dims, ...
					no_train, no_test, eta, eta_par, scheduling, sgd_iter, svrg_iter, error_type);
				toc
			case 'ste_svrg'
				tic
				[mappedX, train_errors(k, t, 2:end), test_errors(k, t, 2:end), run_time(k, t, :)] = ste_svrg_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, N, no_dims, ...
					no_train, no_test, 5*eta, frq_iter, svrg_iter, error_type);
				toc
			case 'ste_batch'
				tic
				[mappedX, train_errors(k, t, 2:end), test_errors(k, t, 2:end), run_time(k, t, :)] = ste_x_epoch_time(X_int, train_triplets_batch, test_triplets_batch, no_dims, 5*eta, no_repeat, ...
					batch_iter, svrg_iter);
				toc
			case 'ste_rank'
				tic
				[mappedX, train_errors(k, t, 2:end), test_errors(k, t, 2:end), run_time(k, t, :), function_value(2, t, :)] = ste_rank(X_int, train_triplets_batch, test_triplets_batch, no_dims, 5*eta, no_repeat, ...
					batch_iter, svrg_iter);
				toc
			case 'ste_kernel'
				tic
				[mappedX, train_errors(k, t, 2:end), test_errors(k, t, 2:end), run_time(k, t, :), function_value(1, t, :)] = ste_kernel(X_int, train_triplets_batch, test_triplets_batch, no_dims, 0.75*eta, no_repeat, ...
					batch_iter, svrg_iter, 0);
				toc
			case 'ste_kernel_bb'
				tic
				[mappedX, train_errors(k, t, 2:end), test_errors(k, t, 2:end), run_time(k, t, :), function_value(3, t, :)] = ste_kernel_svrg_bb(X_int, train_triplets_stoch, test_triplets_batch, no_dims, 5*eta, no_repeat, ...
					frq_iter, svrg_iter);
				toc
			case 'ste_svrg_bb'
				tic
				[mappedX, train_errors(k, t, 2:end), test_errors(k, t, 2:end), run_time(k, t, :), eta_seq(1, t, :)] = ste_svrg_bb_epsilon_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, N, no_dims, ...
					no_train, no_test, 5*eta, 0, 0, frq_iter, svrg_iter, error_type);
				toc				
			case 'ste_svrg_sbb_0'
				tic
				[mappedX, train_errors(k, t, 2:end), test_errors(k, t, 2:end), run_time(k, t, :), eta_seq(1, t, :)] = ste_svrg_bb_epsilon_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, N, no_dims, ...
					no_train, no_test, 5*eta, 0, 2, frq_iter, svrg_iter, error_type);
				toc
			case 'ste_svrg_sbb_epsilon'
				tic
				[mappedX, train_errors(k, t, 2:end), test_errors(k, t, 2:end), run_time(k, t, :), eta_seq(2, t, :)] = ste_svrg_bb_epsilon_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, N, no_dims, ...
					no_train, no_test, 5*eta, epsilon, 2, frq_iter, svrg_iter, error_type);
				toc
		end
		Predict_X(k, t, :, :) = mappedX;
	end
end

lineColor = linspecer(length(techniques));
%lineColor = colorscale(length(techniques), 'hue', [0.1 0.8], 'saturation' , 1, 'value', 0.5);
lineStyle = ['-', '--', ':', '-.'];
LineMarker = ['+', 'o', '*', 'x'];
x = 1:3:3*(svrg_iter+1);
figure(1);
t = quantile(squeeze(test_errors(1, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(7,:), 'LineStyle', '-' , 'Marker', '+', 'LineWidth', 1.5);hold on
t = quantile(squeeze(test_errors(2, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(6,:), 'LineStyle', '--' , 'Marker', 'o', 'LineWidth', 1.5);hold on
t = quantile(squeeze(test_errors(3, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(5,:), 'LineStyle', ':' , 'Marker', '*', 'LineWidth', 1.5);hold on
t = quantile(squeeze(test_errors(4, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(4,:), 'LineStyle', '-.' , 'Marker', 'x', 'LineWidth', 1.5);hold on
t = quantile(squeeze(test_errors(5, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(3,:), 'LineStyle', '-' , 'Marker', '^', 'LineWidth', 1.5);hold on
t = quantile(squeeze(test_errors(6, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(2,:), 'LineStyle', '--' , 'Marker', 'v', 'LineWidth', 1.5);hold on
t = quantile(squeeze(test_errors(7, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(1,:), 'LineStyle', ':' , 'Marker', 's', 'LineWidth', 1.5);hold on
legend('cvx', 'ncvx-Batch', 'ncvx-SGD', 'ncvx-SVRG', 'ncvx-SVRG-BB', 'ncvx-SVRG-SBB_0', 'ncvx-SVRG-SBB_{0.025}');
xlabel('#gradients / #constraints');
ylabel('Generalization Error');
%ylim([-0.05 0.55]);
xlim([0 3*svrg_iter+10]);
set(gca, 'FontName', 'Arial','FontSize', 16);
set(findall(gcf,'type','text'), 'FontName', 'Arial', 'FontSize', 16);

figure(2);
t = quantile(squeeze(train_errors(1, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(7,:), 'LineStyle', '-' , 'Marker', '+', 'LineWidth', 1.5);hold on
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
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(2,:), 'LineStyle', '--' , 'Marker', 'v', 'LineWidth', 1.5);hold on
t = quantile(squeeze(train_errors(7, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(1,:), 'LineStyle', ':' , 'Marker', 's', 'LineWidth', 1.5);hold on
legend('cvx', 'ncvx-Batch', 'ncvx-SGD', 'ncvx-SVRG', 'ncvx-SVRG-BB', 'ncvx-SVRG-SBB_0', 'ncvx-SVRG-SBB_{0.025}');
xlabel('#gradients / #constraints');
ylabel('Train Error');
%ylim([-0.05 0.55]);
xlim([0 3*svrg_iter+10]);
set(gca, 'FontName', 'Arial','FontSize', 16);
set(findall(gcf,'type','text'), 'FontName', 'Arial', 'FontSize', 16);

figure(3);
x = 1:3:3*(svrg_iter);
t = quantile(squeeze(eta_seq(1, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(3,:), 'LineStyle', '-' , 'Marker', '+', 'LineWidth', 1.5);hold on
t = quantile(squeeze(eta_seq(2, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(2,:), 'LineStyle', '--', 'Marker', 'o', 'LineWidth', 1.5);hold on
t = quantile(squeeze(eta_seq(3, :, 1:end)),[0.25,0.5,0.75]);
y = t(2, :);
error_L=t(2, :)-t(1, :);
error_H=t(3, :)-t(2, :);
errorbar(x(1:1:end), y(1:1:end), error_L(1:1:end), error_H(1:1:end), 'Color', lineColor(1,:), 'LineStyle', '--', 'Marker', 'o', 'LineWidth', 1.5);hold on
legend('BB', 'SBB_0', 'SBB_{0.025}');
xlabel('#gradients / #constraints');
ylabel('Step Size');
xlim([0 3*svrg_iter+10]);
set(gca, 'FontName', 'Arial','FontSize', 16);
set(findall(gcf,'type','text'), 'FontName', 'Arial', 'FontSize', 16);

for k = 1:length(techniques)
	tt = zeros(1, num_trial);
	[row, col] = find(squeeze(train_errors(k, :, 2:end))>=0.1);
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
	disp('Median:')
	disp(median(tt));
	disp('Standard Deviation');
	disp(std(tt));
end