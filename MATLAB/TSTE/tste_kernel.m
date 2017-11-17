function [X, train_error, test_error, run_time, function_value] = tste_kernel(init_X, train_triplets, test_triplets, no_dims, eta, no_repeat, batch_iter, svrg_iter, lambda, alpha, use_log)
%TSTE t-Distributed Stochastic Triplet Embedding
%
%   [X, train_error, test_error] = tste_epoch(init_X, train_triplets, test_triplets, no_dims, no_repeat, batch_iter, svrg_iter, use_log, lambda, alpha)
% 
% The function implements t-distributed stochastic triplet embedding (t-STE) 
% based on the specified triplets, to construct an embedding with no_dims 
% dimensions. The parameter lambda specifies the amount of L2-
% regularization (default = 0), whereas alpha sets the number of degrees of
% freedom of the Student-t distribution (default = 1). The variable use_log
% determines whether the sum of the log-probabilities or the sum of the
% probabilities is maximized (default = true).
%
% Note: This function directly learns the embedding X and returns the train & test error for each epoch.
% It is modifed with the code provided by Laurens van der Maaten, 2012, Delft University of Technology

if ~exist('no_dims', 'var') || isempty(no_dims)
	no_dims = 2;
end
if ~exist('lambda', 'var') || isempty(lambda)
	lambda = 0;
end
if ~exist('alpha', 'var') || isempty(alpha)
	alpha = no_dims - 1;
end
if ~exist('use_log', 'var') || isempty(use_log)
	use_log = true;
end
addpath(genpath('minFunc'));

% Determine number of objects
N = max(train_triplets(:));
train_triplets(any(train_triplets == -1, 2),:) = [];
no_train = size(train_triplets, 1);
no_test = size(test_triplets, 1);

% Initialize some variables
% X = randn(N, no_dims) .* .0001;
C = Inf;
% convergence tolerance
tol = 1e-5;
% maximum number of iterations
% max_iter = 1000;
% learning rate
% eta = 0.01;
% best error obtained so far               
% best_C = Inf;
% best embedding found so far          
% best_X = X;
K = init_X * init_X';
train_error = zeros(1, svrg_iter);
test_error = zeros(1, svrg_iter);
run_time = zeros(1, svrg_iter);

% Perform main learning iterations
iter = 0;
no_incr = 0;
t = 1;

%while iter < max_iter && no_incr < 5
while iter < batch_iter
	tt = clock;
	% Compute value of slack variables, cost function, and gradient
	old_C = C;
	% checkgrad('tste_grad', X(:), 1e-7, N, no_dims, triplets, lambda, alpha, use_log);
	%, pause
	% [C, G] = tste_grad(X(:), N, no_dims, train_triplets, lambda, alpha, use_log);
	[C, G] = tste_k_grad(K(:), N, train_triplets, lambda, alpha);
	% Maintain best solution found so far
	% if C < best_C
	% 	best_C = C;
	% 	best_X = X;
	% end
	K = K - (eta ./ no_train .* N) .* reshape(G, [N N]);
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
	% Perform gradient update        

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
	X = bsxfun(@times, sqrt(diag(L(1:no_dims, 1:no_dims)))', V(:, 1:no_dims));
end  