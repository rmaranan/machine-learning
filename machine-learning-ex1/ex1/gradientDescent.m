function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

%theta = theta - (alpha / m) * sum(((theta(1)+theta(2)*X(:,2))-y).*X(:,2))
%theta = theta - (alpha/m) * sum(

%for i=1:m
%(theta(1) + theta(2)'.*X(m,2)-y).*X(m,2))
%end

    g = 0;
    k = 0;
    for i = 1 : m
        k = k + theta(1) + theta(2) * X(i,2) - y(i)*X(i,1);
        g = g + (theta(1) + theta(2) * X(i,2) - y(i)) * X(i, 2);
    end
    theta = theta - alpha / m * [k;g]
   


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
