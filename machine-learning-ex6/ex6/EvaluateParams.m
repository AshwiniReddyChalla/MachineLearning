%% Machine Learning Online Class
%  Exercise 6 | Support Vector Machines
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     gaussianKernel.m
%     dataset3Params.m
%     processEmail.m
%     emailFeatures.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% =============== Part 6: Visualizing Dataset 3 ================
%  The following code will load the next dataset into your environment and 
%  plot the data. 
%

fprintf('Loading and Visualizing Data ...\n')

% Load from ex6data3: 
% You will have X, y in your environment
load('ex6data3.mat');

% Plot training data
plotData(X, y);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

%  This is a different dataset that you can use to experiment with. Try
%  different values of C and sigma here.
% 

% Load from ex6data3: 
% You will have X, y in your environment
load('ex6data3.mat');

% Try different SVM Parameters here
%[C, sigma] = dataset3Params(X, y, Xval, yval);
C = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
sigma = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];

gooderror = Inf;
goodsigma = 0;
goodC = 0;
for i= 1: length(C)
    for j= 1:length(sigma)
      model= svmTrain(X, y, C(i), @(x1, x2) gaussianKernel(x1, x2, sigma(j)));
      model.kernelFunction;
      ypred = svmPredict( model,Xval);
      error = sum(abs(ypred - yval));
      fprintf('c=%f sigma=%f error=%f\n',C(i),sigma(j), error);
      if error < gooderror
         gooderror = error;
         goodsigma = sigma(j);
         goodC = C(i);
      end          
    end
end
    
% Train the SVM
fprintf('goodsigma = %f\n',goodsigma);
fprintf('goodC = %f\n',goodC);
model= svmTrain(X, y, goodC, @(x1, x2) gaussianKernel(x1, x2, goodsigma));
visualizeBoundary(X, y, model);

fprintf('Program paused. Press enter to continue.\n');
pause;

