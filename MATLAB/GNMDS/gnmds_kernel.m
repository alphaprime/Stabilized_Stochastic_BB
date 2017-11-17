function [X, train_error, test_error, run_time, function_value, norm_value] = gnmds_kernel(init_X, train_triplets, test_triplets, no_dims, eta, no_repeat, batch_iter, svrg_iter, lambda)
%GNMDS_K Generalized Non-metric Multi-Dimensional Scaling w.r.t. K
%
%   [X, train_error, test_error, run_time, function_value] = gnmds_kernel(init_X, train_triplets, test_triplets, no_dims, no_repeat, batch_iter, svrg_iter)
%
% The function implements generalized non-metric MDS (GNMDS) based on the 
% specified triplets, to learn a kernel K and a corresponding embedding X 
% with no_dims dimensions. The parameter lambda specifies the amount of L2-
% regularization (default = 0).
%
% Note: This function learns the kernel K and gets X via an SVD and returns the train & test error for each epoch.
% It is modifed with the code provided by Laurens van der Maaten, 2012, Delft University of Technology


if ~exist('lambda', 'var') || isempty(lambda)
	lambda = 0;
end
if ~exist('no_dims', 'var') || isempty(no_dims)
	no_dims = 2;
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
train_error = zeros(1, svrg_iter);
test_error = zeros(1, svrg_iter);
run_time = zeros(1, svrg_iter);
function_value = zeros(1, svrg_iter);
norm_value = zeros(1, svrg_iter);

% convergence tolerance
tol = 1e-5;
% maximum number of iterations
% max_iter = 1000; 
% learning rate       
% eta = 1;
% best error obtained so far             
old_C = Inf;
C = Inf;
% best embedding kernel found so far           
% best_K = K;             
    
% Initialize gradient
G = zeros(N);

D = bsxfun(@plus, bsxfun(@plus, -2 .* K, diag(K)), diag(K)');
% Compute value of slack variables
residual = D(sub2ind([N N], train_triplets(:,1), train_triplets(:,2))) + 1 - ...
		D(sub2ind([N N], train_triplets(:,1), train_triplets(:,3)));
% Compute value of cost function
slack = max(residual, 0);
vio = (slack>0);

% Perform main learning iterations
iter = 0;
t = 1;

while iter < batch_iter  
	tt = clock;
	% s_{13} = 2 & s_{31} = 2
	G = G + 2*reshape(accumarray(sub2ind([N N], train_triplets(vio, 1), train_triplets(vio, 3)), 2, [N * N 1]), [N, N]);
	G = G + 2*reshape(accumarray(sub2ind([N N], train_triplets(vio, 3), train_triplets(vio, 1)), 2, [N * N 1]), [N, N]);
	% s_{12} = -2 & s_{21} = -2
	G = G - 2*reshape(accumarray(sub2ind([N N], train_triplets(vio, 1), train_triplets(vio, 2)), 2, [N * N 1]), [N N]);
	G = G - 2*reshape(accumarray(sub2ind([N N], train_triplets(vio, 2), train_triplets(vio, 1)), 2, [N * N 1]), [N N]);
	% s_{22} = 1
	G = G + reshape(accumarray(sub2ind([N N], train_triplets(vio, 2), train_triplets(vio, 2)), 1, [N * N 1]), [N N]);
	% s_{33} = -1
	G = G - reshape(accumarray(sub2ind([N N], train_triplets(vio, 3), train_triplets(vio, 3)), 1, [N * N 1]), [N N]);
	K = K - ((eta ./ no_train .* N) .* G + lambda.*eye(N));
	D = bsxfun(@plus, bsxfun(@plus, -2 .* K, diag(K)), diag(K)');
	% Compute value of slack variables
	residual = D(sub2ind([N N], train_triplets(:,1), train_triplets(:,2))) + 1 - ...
			D(sub2ind([N N], train_triplets(:,1), train_triplets(:,3)));
	% Compute value of cost function
	slack = max(residual, 0);
	C = lambda.* trace(K) + (sum(slack(:))./no_train);
	vio = (slack>0);
	% Project kernel back onto the PSD cone
	[V, L] = eig(K);
	V = real(V);
	L = real(L);
	ind = find(diag(L) > 0);
	if isempty(ind)
		warning('Projection onto PSD cone failed. All eigenvalues were negative.'); break
	end
	K = V(:,ind) * L(ind, ind) * V(:,ind)';
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
		X = bsxfun(@times, sqrt(diag(L(ind, ind)))', V(:,ind));
		no_train_viol = sum(D(sub2ind([N N], train_triplets(:,1), train_triplets(:,2))) > ...
						D(sub2ind([N N], train_triplets(:,1), train_triplets(:,3))));
		no_test_viol = sum(D(sub2ind([N N], test_triplets(:,1), test_triplets(:,2))) > ...
						D(sub2ind([N N], test_triplets(:,1), test_triplets(:,3))));
		train_error(t) = no_train_viol ./ no_train;
		test_error(t) = no_test_viol ./ no_test;
		function_value(t) = C;
		norm_value(t) = norm(X, 'fro');
		t = t+1;
	end

end

% Compute low-dimensional embedding as well        
if nargout > 1
	%[X, L, ~] = svd(K);
	X = bsxfun(@times, sqrt(diag(L(1:no_dims, 1:no_dims)))', V(:,1:no_dims));
end