close all;
clear;
clc;

load('sun_split.mat');
filename = 'sun_triplets_permu_train_test.mat';

% initialize the random number generator to make the results in this example repeatable.
rng('default');
% initialize the generator using a seed of 1 and specify the generator as Mersenne Twister.
% save the generator settings in a structure s.
random_seed = rng(1,'twister');

% ordinal embedding requires queries are also the database samples
% all the database samples are used for training in unsupervised setting
N = 1080;
per_class = 60;
M = length(testgnd);
no_class = length(unique(testgnd));
no_dims = no_class;
labels = 1:18;
labels = repmat(labels, 100, 1);
labels = reshape(labels, M, 1);
train_ind_n = zeros(1, M);
for c = 1:no_class
	tmp = randperm(100, per_class);
	tmp = tmp + (c-1)*100;
	train_ind_n(tmp) = 1;
end
test_ind_n = ones(1, M)-train_ind_n;
permu = randperm(M);
labels = labels(permu);
data = testdata(permu, :);
train_ind_n = train_ind_n(permu);
test_ind_n = test_ind_n(permu);
train_ind_n = find(train_ind_n == 1);
test_ind_n = find(test_ind_n == 1);
%train_ind_n = randperm(M, N);
%test_ind_n = true(1, M);
%test_ind_n(train_ind_n) = false;
train_labels = labels(train_ind_n, :);
test_labels = labels(test_ind_n, :);
%test_ind_n = find(test_ind_n == 1);

sum_X = sum(data .^ 2, 2);
DD = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (data * data')));
sig = sum(sum(DD))/(M^2);
kernel = rbf(data, sig);

class_idx = cell(no_class, 1);
for c = 1:no_class
	idx = find(train_labels == c);
	class_idx{c} = idx; 
end

no_triplets = 70000;

disp('Generating unsupervised triplets...');

triplets = zeros(max(no_triplets), 3);

for i = 1:max(no_triplets)
	class_ind = randperm(no_dims, 2);
	same_ind = randperm(per_class, 2);
	un_same_ind = randperm(per_class, 1);
	triplets(i, :) = [class_idx{class_ind(1)}(same_ind(1)), class_idx{class_ind(1)}(same_ind(2)), class_idx{class_ind(2)}(un_same_ind)];
end
perm = randperm(no_triplets);
triplets = triplets(perm, :);

noise_ratio = 0.1;
noise_id = randperm(no_triplets, no_triplets*noise_ratio);
triplets(noise_id, [2 3]) = triplets(noise_id, [3 2]);

techniques = {'ste_kernel', 'ste_batch', 'ste_sgd', 'ste_svrg', 'ste_svrg_bb', 'ste_svrg_rbb'};

no_train = 60000;
no_test = 10000;
svrg_iter  = 100;
no_repeat = 1;
frq_iter = no_repeat*no_train;
sgd_iter = (no_repeat+2)*svrg_iter;
batch_iter = sgd_iter;	
eta = 0.00005;
eta_par = 0.05;
scheduling = 1;	
error_type = 1;
lambda = 0.1;

train_errors = zeros(length(techniques), 1+svrg_iter);
test_errors = zeros(length(techniques), 1+svrg_iter);
embed_X = zeros(length(techniques), N, no_dims);	
run_time = zeros(length(techniques), svrg_iter);
coeff = zeros(length(techniques), N, no_dims);
eta_seq = zeros(2, svrg_iter);

train_ind = randperm(no_triplets, no_train);
test_ind = true(1, no_triplets);
test_ind(train_ind) = false;
train_triplets = triplets(train_ind, :);
test_triplets = triplets(test_ind, :);
train_triplets_stoch = train_triplets - 1;
test_triplets_stoch = test_triplets - 1;

X_int = rand(N, no_dims);
sum_X = sum(X_int .^ 2, 2);
D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X_int * X_int')));
if error_type == 1
	no_train_viol = sum(D(sub2ind([N N], train_triplets(:, 1), train_triplets(:, 2))) > ...
			D(sub2ind([N N], train_triplets(:, 1), train_triplets(:, 3))));
	no_test_viol = sum(D(sub2ind([N N], test_triplets(:, 1), test_triplets(:, 2))) > ...
			 D(sub2ind([N N], test_triplets(:, 1), test_triplets(:, 3))));
	train_errors(:, 1) = no_train_viol/no_train;
	test_errors(:, 1) = no_test_viol/no_test;
else
	[~, sort_ind] = sort(D, 2, 'ascend');
	train_errors(:, 1) = sum(labels(sort_ind(:, 2)) ~= labels) ./ N;
	test_errors(:, 1) = train_errors(:, 1);
end

epsilon = 1e-5;

for k = 1:length(techniques)
	switch techniques{k}
		case 'ste_sgd'
			tic
			[mappedX, train_errors(k, 2:end), test_errors(k, 2:end), run_time(k, :)] = ste_sgd_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, N, no_dims, ...
				no_train, no_test, 12.5*eta, eta_par, scheduling, sgd_iter, svrg_iter, error_type);
			toc
		case 'ste_svrg'
			tic
			[mappedX, train_errors(k, 2:end), test_errors(k, 2:end), run_time(k, :)] = ste_svrg_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, N, no_dims, ...
				no_train, no_test, 25*eta, frq_iter, svrg_iter, error_type);
			toc
		case 'ste_batch'
			tic
			[mappedX, train_errors(k, 2:end), test_errors(k, 2:end), run_time(k, :)] = ste_x_epoch_time(X_int, train_triplets, test_triplets, no_dims, 25*eta, no_repeat, ...
				batch_iter, svrg_iter);
			toc
		case 'ste_svrg_bb'
			tic
			[mappedX, train_errors(k, 2:end), test_errors(k, 2:end), run_time(k, :), eta_seq(1, :)] = ste_svrg_bb_epsilon_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, N, no_dims, ...
				no_train, no_test, 25*eta, 0, 0, frq_iter, svrg_iter, error_type);
			toc
		case 'ste_svrg_rbb'
			tic
			[mappedX, train_errors(k, 2:end), test_errors(k, 2:end), run_time(k, :), eta_seq(2, :)] = ste_svrg_bb_epsilon_mex_ifo_time(X_int, train_triplets_stoch, test_triplets_stoch, labels, N, no_dims, ...
				no_train, no_test, 25*eta, epsilon, 2, frq_iter, svrg_iter, error_type);
			toc
		case 'ste_kernel'
			tic
			[mappedX, train_errors(k, 2:end), test_errors(k, 2:end), run_time(k, :), ~] = ste_kernel(X_int, train_triplets, test_triplets, no_dims, 50*eta, no_repeat, ...
				batch_iter, svrg_iter, 0);
			toc
	end
	embed_X(k, :, :) = mappedX;
	% RLS
	sum_X = sum(mappedX .^ 2, 2);
	DD = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (mappedX * mappedX')));
	sig = sum(sum(DD))/(N^2);
	K = rbf(mappedX, sig);
	R = chol(K+lambda*eye(N));
	for p = 1:no_dims
		Y = mappedX(:, p);
		coeff(k, :, p) = (R\(R'\Y));
	end
end

position = [100-per_class:10:100];
%Precision@100 & Recall@Position
precision = zeros(length(techniques), length(position));
recall = zeros(length(techniques), length(position));
%Mean Average Precision
MAP = zeros(length(techniques), length(position));
classes = [1 2];
S = M-N;

for k = 1:length(techniques)
	%X = squeeze(embed_X(k, :, :));
	%sum_X = sum(X.^2, 2);
	%D = bsxfun(@plus, sum_X, bsxfun(@plus, sum_X', -2 * (X * X')));
	feature = kernel(test_ind_n, train_ind_n)*squeeze(coeff(k, :, :));
	sum_feature = sum(feature.^2, 2);
	D = bsxfun(@plus, sum_feature, bsxfun(@plus, sum_feature', -2 * (feature * feature')));
	[~, sort_ind] = sort(D, 2, 'ascend');
	for p = 1:length(position)
		for n = 1:S
			ssort = zeros(1, S);
			gnd  = 2*ones(1, S);
			pred = 2*ones(1, S);
			gnd(find(test_labels == test_labels(n))) = 1; 
			pred(sort_ind(n, 1:position(p))) = 1;
			for i = 1:2
				a = classes(i);
				d = find(gnd == a);% d has indices of points with class a
				for j = 1:2
					confus(i, j) = length(find(pred(d) == classes(j)));
				end
			end
			precision(k, p) = precision(k, p) + confus(1, 1)./sum(confus(:, 1));
			recall(k, p) = recall(k, p) + confus(1, 1)./sum(confus(1, :));
			pp = intersect(find(gnd == 1), find(pred == 1));
			for j = 1:length(pp)
				ssort(find(sort_ind(n, :) == pp(j))) = 1;
			end
			for j = 1:S
				if ssort(j)
					MAP(k, p) = MAP(k, p) + sum(ssort(1:j))/j;
				end
			end
		end
		recall(k, p) = recall(k, p)./S;
		MAP(k, p) = MAP(k, p)./((100-per_class)*S);
		precision(k, p) = precision(k, p)./S;
	end
end	

lineColor = linspecer(length(techniques));
plot(position, recall(1, :), 'Color', lineColor(6,:), 'LineStyle', '-' , 'Marker', '+', 'LineWidth', 2);hold on
plot(position, recall(2, :), 'Color', lineColor(5,:), 'LineStyle', '--', 'Marker', 'o', 'LineWidth', 2);hold on
plot(position, recall(3, :), 'Color', lineColor(4,:), 'LineStyle', ':' , 'Marker', '*', 'LineWidth', 2);hold on
plot(position, recall(4, :), 'Color', lineColor(3,:), 'LineStyle', '-.', 'Marker', 'x', 'LineWidth', 2);hold on
plot(position, recall(5, :), 'Color', lineColor(2,:), 'LineStyle', '-' , 'Marker', '^', 'LineWidth', 2);hold on
plot(position, recall(6, :), 'Color', lineColor(1,:), 'LineStyle', '--', 'Marker', 'v', 'LineWidth', 2);hold on
%legend('cvx', 'ncvx-Batch', 'ncvx-SGD', 'ncvx-SVRG', 'ncvx-SVRG-SBB_0', 'ncvx-SVRG-SBB_{10^{-5}}');
legend('ProjGD', 'FGD', 'SFGD', 'SVRG', 'SVRG-SBB_0', 'SVRG-SBB_{10^{-5}}');
xlabel('K');
ylabel('Recall@K');
% set(gca, 'FontName', 'Arial','FontSize', 16);
% set(findall(gcf,'type','text'), 'FontName', 'Arial', 'FontSize', 16);