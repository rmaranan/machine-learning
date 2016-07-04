function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

%Compute J by taking the sum of all Js increments

for i = 1:m
  J = J + 1/m *(-y(i)*log(sigmoid(theta'*X(i,:)'))-(1-y(i))*log(1-sigmoid(theta'*X(i,:)')))
end

%interm = 0
%for i = 1:m
%  interm(i) = -y(i)*log(sigmoid(X(i)))-(1-y(i))*log(1-sigmoid(X(i)));
%end

%J = 1/m * sum(interm)

% Compute gradient 

for j = 1:length(theta)
  %grad(j) = 1/m * sum((sigmoid(theta'*X(:,j)'*theta(i))-y(:)).*X(:,j));
  grad(j) = 1/m * sum((sigmoid(theta'*X(j,:)')-y(:))'*X(:,j))
end



% =============================================================

end
