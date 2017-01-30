function [J, grad] = linearRegCostFunction(X, y, theta, lambda)

m = length(y); % number of training examples

J = 0;
grad = zeros(size(theta));



J  = (sum((X*theta - y).^2) + (lambda * (sum(theta.^2) - theta(1)^2)))/ (2*m);

Temp = X * theta;
Temp = Temp - y;
grad(1) = (sum(Temp))/m;
for i = 2:size(theta)
    grad(i) = (sum(Temp .* X(:, i)))/m;
    grad(i) = grad(i) + ((lambda * theta(i))/m) ;
end









% =========================================================================


end
