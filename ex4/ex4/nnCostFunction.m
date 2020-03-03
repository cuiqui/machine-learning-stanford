function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Feedforward
y_matrix = eye(num_labels)(y,:);
X_with_bias = [ones(rows(X), 1) X];
z_2 = zeros(m, hidden_layer_size);
a_2 = zeros(m, hidden_layer_size+1);
a_3 = zeros(m, num_labels);

for i=1:m
  tmp = Theta1 * X_with_bias(i, :)';
  z_2(i, :) = tmp';
  a_2(i, :) = [1; arrayfun(@(z) sigmoid(z), z_2(i, :))']';
  z_3 = Theta2 * a_2(i, :)';
  a_3(i, :) = arrayfun(@(z) sigmoid(z), z_3);
endfor

%temp = 0;
%for i=1:m
%  for k=1:num_labels
%    temp += y_matrix(i, k) * log(a_3(i, k)) + (1-y_matrix(i, k)) * log (1 - (a_3(i, k)));
%  endfor
%endfor

% replace first column of theta with 0 for regularization
r_theta1 = Theta1;
r_theta1(:, 1) = 0;
r_theta2 = Theta2;
r_theta2(:, 1) = 0;

inner_sum = sum(y_matrix .* log(a_3) + (1-y_matrix) .* log(1-a_3), 2);
J_no_reg = (-1/m) * sum(inner_sum);
reg = (lambda / (2 * m)) * (sum(sum(r_theta1 .^ 2 , 2)) + ... 
       sum(sum(r_theta2 .^ 2 , 2)));
J = J_no_reg + reg;

% Backpropagation

d_3 = a_3 - y_matrix;
d_2 = (d_3 * Theta2(:, 2:end)) .* sigmoidGradient(z_2);
D_1 = d_2' * X_with_bias;
D_2 = d_3' * a_2;
Theta1_grad = (1/m) * D_1 + (lambda / m) * r_theta1;
Theta2_grad = (1/m) * D_2 + (lambda / m) * r_theta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
