function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

n = length(theta);
reg = 0;
for j = 2:n,
  reg = reg + theta(j, 1) ^ 2;
end;

grad_reg = zeros(size(theta));
grad_reg = grad_reg + theta;
grad_reg(1, 1) = 0;

for i = 1:m,
  J = J + (-y(i, :) * log(sigmoid(theta'* X(i, :)')) - (1 - y(i, :)) * log(1 - sigmoid(theta' * X(i, :)'))) / (m);
end;

J = J + (lambda * reg) / (2 * m)

for i = 1:m,
  grad = grad + ((sigmoid(theta' * X(i, :)') - y(i, :)) / m) * X(i, :)';
end;

grad = grad + ((lambda / m) .* grad_reg);

% =============================================================

end
