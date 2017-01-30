function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
grad = zeros(size(theta));
s = size(theta);
H = sigmoid(X * theta);
log1 = -log(H);
log2 = -log(1 - H);

J = (sum ( y .* log1 + (1-y) .* log2 ) ) / m + (((sum(theta .^2) - (theta(1)^2)) *lambda) / (2*m));

for i = 1 : s(1)
    if i == 1
    grad(i) = ((sum( (H - y) .* X(:,i))))/ m ;
    else
    grad(i) = ((sum( (H - y) .* X(:,i))))/ m + ((theta(i) * lambda)/ m); 
    end;
    


% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta






% =============================================================

end
