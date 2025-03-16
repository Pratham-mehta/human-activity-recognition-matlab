%% Step 1: Load Data
clc; clear; close all;

% Load prepared training and test data
X_train = readmatrix('dataset/Train/X_train.txt');
y_train = readmatrix('dataset/Train/y_train.txt');
X_test = readmatrix('dataset/Test/X_test.txt');
y_test = readmatrix('dataset/Test/y_test.txt');

% Normalize the data (ensuring all values are within the same range)
X_train = normalize(X_train, 'range');
X_test = normalize(X_test, 'range');

% Load feature importance ranking
feature_importance = readmatrix('feature_importance.txt');
features = readtable('dataset/features.txt', 'ReadVariableNames', false);
feature_names = features.Var1;

% Select top 30 features based on importance ranking
top_k = 30;
top_features_idx = feature_importance(1:top_k);

% Reduce dataset to selected top 30 features
X_train_selected = X_train(:, top_features_idx);
X_test_selected = X_test(:, top_features_idx);

% Save the reduced dataset
writematrix(X_train_selected, 'dataset/Train/X_train_selected.txt');
writematrix(X_test_selected, 'dataset/Test/X_test_selected.txt');
writematrix(y_train, 'dataset/Train/y_train_selected.txt');
writematrix(y_test, 'dataset/Test/y_test_selected.txt');

% Display confirmation
disp("✅ Top 30 features selected and dataset saved.");