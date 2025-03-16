%% Step 1: Load Preprocessed Data (Top 20 Features)
clc; clear; close all;

% Load selected feature dataset
X_train = readmatrix('dataset/Train/X_train_selected.txt');
y_train = readmatrix('dataset/Train/y_train_selected.txt');
X_test = readmatrix('dataset/Test/X_test_selected.txt');
y_test = readmatrix('dataset/Test/y_test_selected.txt');

% Normalize data (ensuring consistency)
X_train = normalize(X_train, 'range');
X_test = normalize(X_test, 'range');

%% Step 2: Train Decision Tree Classifier
disp("Training Decision Tree with Top 30 Features...");
decisionTreeModel = fitctree(X_train, y_train);
y_pred_tree = predict(decisionTreeModel, X_test);
acc_tree = sum(y_pred_tree == y_test) / length(y_test) * 100;
disp("✅ Decision Tree Accuracy (Top 30 Features): " + acc_tree + "%");

%% Step 3: Train k-Nearest Neighbors (k-NN)
disp("Training k-NN (k=5) with Top 30 Features...");
knnModel = fitcknn(X_train, y_train, 'NumNeighbors', 5);
y_pred_knn = predict(knnModel, X_test);
acc_knn = sum(y_pred_knn == y_test) / length(y_test) * 100;
disp("✅ k-NN Accuracy (Top 30 Features): " + acc_knn + "%");

%% Step 4: Train Support Vector Machine (SVM)
disp("Training SVM with Top 30 Features...");
svmModel = fitcecoc(X_train, y_train);
y_pred_svm = predict(svmModel, X_test);
acc_svm = sum(y_pred_svm == y_test) / length(y_test) * 100;
disp("✅ SVM Accuracy (Top 30 Features): " + acc_svm + "%");

%% Step 5: Display Results
results_selected = table(["Decision Tree"; "k-NN (k=5)"; "SVM"], [acc_tree; acc_knn; acc_svm], ...
                'VariableNames', {'Model', 'Accuracy (%)'});
disp(results_selected);
