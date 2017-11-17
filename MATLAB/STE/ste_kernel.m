function [X, train_error, test_error, run_time, function_value] = ste_kernel(init_X, train_triplets, test_triplets, no_dims, eta, no_repeat, batch_iter, svrg_iter, lambda)
%STE_K_EPOCH Stochastic Triplet Embedding
%
%   [X, train_error, test_error, run_time] = ste_kernel(init_X, train_triplets, test_triplets, no_dims, no_repeat, batch_iter, svrg_iter, lambda)
% 
% The function implements stochastic triplet embedding (STE) based on the 
% specified triplets, to construct an embedding with no_dims dimensions. 
% The parameter lambda specifies the amount of L2-regularization (default =
% 0).
%
% Note: This function learns the kernel K, and gets X via an SVD.
% It is modifed by the code provided by Laurens van der Maaten, 2012, Delft University of Technology

if ~exist('no_dims', 'var') || isempty(no_dims)
	no_dims = 2;
end
if ~exist('lambda', 'var') || isempty(lambda)
	lambda = 0;
end
addpath(genpath('minFunc'));

% Determine number of objects
N = max(train_triplets(:));
train_triplets(any(train_triplets == -1, 2),:) = [];
no_train = size(train_triplets, 1);
no_test = size(test_triplets, 1);

% Initialize some variables
% K = randn(N, no_dims) * .1;
K = init_X * init_X';
C = Inf;
old_C = Inf;
% convergence tolerance 
tol = 1e-5;
train_error = zeros(1, svrg_iter);
test_error = zeros(1, svrg_iter);
run_time = zeros(1, svrg_iter);
function_value = zeros(1, svrg_iter);

% Perform main learning iterations
iter = 0;
t = 1;

while iter < batch_iter
	tt = clock;
	% Perform gradient update
	[C, G] = ste_k_grad(K(:), N, train_triplets, lambda);
	K = K - (eta ./ no_train .* N) .* reshape(G, [N N]);
	% D = bsxfun(@plus, bsxfun(@plus, -2 .* K, diag(K)), diag(K)');
	% exp_D = exp(-D);
	% P = exp_D(sub2ind([N N], train_triplets(:,1), train_triplets(:,2))) ./ ...
	% 	(exp_D(sub2ind([N N], train_triplets(:,1), train_triplets(:,2))) +  ...
	% 	exp_D(sub2ind([N N], train_triplets(:,1), train_triplets(:,3))));
	% % Compute value of cost function
	% C = (-sum(log(max(P(:), realmin)))/no_train) + lambda .* trace(K);
	% Project kernel back onto the PSD cone
	[V, L] = eig(K);
	V = real(V);
	L = real(L);
	ind = find(diag(L) > 0);
	if isempty(ind)
		warning('Projection onto PSD cone failed. All eigenvalues were negative.'); break
	end
	K = V(:, ind) * L(ind, ind) * V(:, ind)';
	if any(isinf(K(:)))
		warning('Projection onto PSD cone failed. Metric contains Inf values.'); break
	end
	if any(isnan(K(:)))
		warning('Projection onto PSD cone failed. Metric contains NaN values.'); break
	end
	run_time(t) = run_time(t) + etime(clock, tt);
	
	% Print out progress
	iter = iter + 1;
	if ~rem(iter, no_repeat+2)
		D = bsxfun(@plus, bsxfun(@plus, -2 .* K, diag(K)), diag(K)');
		no_train_viol = sum(D(sub2ind([N N], train_triplets(:,1), train_triplets(:,2))) > ...
						D(sub2ind([N N], train_triplets(:,1), train_triplets(:,3))));
		no_test_viol = sum(D(sub2ind([N N], test_triplets(:,1), test_triplets(:,2))) > ...
						D(sub2ind([N N], test_triplets(:,1), test_triplets(:,3))));
		train_error(t) = no_train_viol ./ no_train;
		test_error(t) = no_test_viol ./ no_test;
		function_value(t) = C;
		t = t+1;
	end
end

% Compute low-dimensional embedding as well        
if nargout > 1
	% [X, L, ~] = svd(K);
	% X = bsxfun(@times, sqrt(diag(L(1:no_dims, 1:no_dims)))', X(:, 1:no_dims));
	X = bsxfun(@times, sqrt(diag(L(1:no_dims, 1:no_dims)))', V(:, 1:no_dims));
end  