function [C, dC] = tste_k_grad(x, N, triplets, lambda, alpha)


    % Decode current solution
    K = reshape(x, [N N]);
    
    % Compute Gaussian kernel
    K = exp(-bsxfun(@plus, bsxfun(@plus, -2 .* K, diag(K)), diag(K)'));
    base_K = 1+K/alpha;
    t_K = base_K.^(-(alpha+1)/2);
    % Compute value of cost function
    P = t_K(sub2ind([N N], triplets(:,1), triplets(:,2))) ./ ...
       (t_K(sub2ind([N N], triplets(:,1), triplets(:,2))) +  ...
        t_K(sub2ind([N N], triplets(:,1), triplets(:,3))));
    C = -sum(log(max(P(:), realmin)))/size(triplets, 1) + lambda * trace(K);
    
    % Compute gradient if requested
    if nargout > 1
        % Compute gradient
        dC = zeros(N, N);
        for n = 1:size(triplets, 1)
            i = triplets(n, 1);
            j = triplets(n, 2);
            k = triplets(n, 3);
            dij = alpha+K(i, j);
            dik = alpha+K(i, k);
            p1 = (1+(dik/dij))^((alpha+1)/2);
            p2 = (-(alpha+1)/2)*(dik/dij)^(-(alpha+3)/2);
            p = p1*p2;
            dC(i, i) = (dij-dik)*p/(dij^2);
            dC(i, j) = 2*dik*p/(dij^2);
            dC(i, k) = -2*p/dij;
            dC(j, j) = (-dik)*p/(dij^2);
            dC(k, k) = p/dij;
        end
        dC = dC + lambda * speye(N);
        dC = dC(:);
    end
    