%% Step 1: Load Preprocessed Data
clc; clear; close all;

% Load prepared data
X_train = readmatrix('dataset/Train/X_train.txt');
y_train = readmatrix('dataset/Train/y_train.txt');
X_test = readmatrix('dataset/Test/X_test.txt');
y_test = readmatrix('dataset/Test/y_test.txt');

% Normalize data
X_train = normalize(X_train, 'range');
X_test = normalize(X_test, 'range');

%% Step 2: Train Decision Tree Classifier
disp("Training Decision Tree...");
decisionTreeModel = fitctree(X_train, y_train);
y_pred_tree = predict(decisionTreeModel, X_test);
acc_tree = sum(y_pred_tree == y_test) / length(y_test) * 100;
disp("✅ Decision Tree Accuracy: " + acc_tree + "%");

%% Step 3: Train k-Nearest Neighbors (k-NN)
disp("Training k-NN (k=5)...");
knnModel = fitcknn(X_train, y_train, 'NumNeighbors', 5);
y_pred_knn = predict(knnModel, X_test);
acc_knn = sum(y_pred_knn == y_test) / length(y_test) * 100;
disp("✅ k-NN Accuracy: " + acc_knn + "%");

%% Step 4: Train Support Vector Machine (SVM)
disp("Training SVM...");
svmModel = fitcecoc(X_train, y_train);
y_pred_svm = predict(svmModel, X_test);
acc_svm = sum(y_pred_svm == y_test) / length(y_test) * 100;
disp("✅ SVM Accuracy: " + acc_svm + "%");

%% Step 5: Display Results
results = table(["Decision Tree"; "k-NN (k=5)"; "SVM"], [acc_tree; acc_knn; acc_svm], ...
                'VariableNames', {'Model', 'Accuracy (%)'});
disp(results);