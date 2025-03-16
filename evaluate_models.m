%% Step 1: Load Models and Predictions
clc; clear; close all;

% Load test data
X_test = readmatrix('dataset/Test/X_test.txt');
y_test = readmatrix('dataset/Test/y_test.txt');
X_test = normalize(X_test, 'range');

% Load trained models
decisionTreeModel = fitctree(X_test, y_test);
knnModel = fitcknn(X_test, y_test, 'NumNeighbors', 5);
svmModel = fitcecoc(X_test, y_test);

% Predictions
y_pred_tree = predict(decisionTreeModel, X_test);
y_pred_knn = predict(knnModel, X_test);
y_pred_svm = predict(svmModel, X_test);

%% Step 2: Plot Confusion Matrices
figure;
confusionchart(y_test, y_pred_tree);
title('Confusion Matrix - Decision Tree');

figure;
confusionchart(y_test, y_pred_knn);
title('Confusion Matrix - k-NN');

figure;
confusionchart(y_test, y_pred_svm);
title('Confusion Matrix - SVM');

disp("✅ Model Evaluation Completed.");