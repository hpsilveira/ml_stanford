function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% Cost regularization
reg = 0;
reg = sum(theta(2:end) .^ 2);

% Gradient regularization
grad_reg = zeros(size(theta));
grad_reg = grad_reg + theta;
grad_reg(1, 1) = 0;

% Cost function
J = (X * theta - y) .^ 2;
J = sum(J(:)) / (2 * m);
J = J + (lambda * reg) / (2 * m);

% Gradient
grad = (X' * (X * theta - y)) / m;
grad = grad + ((lambda / m) * grad_reg);

% =========================================================================

grad = grad(:);

end
