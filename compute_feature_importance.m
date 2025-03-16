%% Step 1: Load Data
clc; clear; close all;

% Load prepared training data
X_train = readmatrix('dataset/Train/X_train.txt');
y_train = readmatrix('dataset/Train/y_train.txt');

% Normalize the data (ensuring all values are within the same range)
X_train = normalize(X_train, 'range');

%% Step 2: Train a Decision Tree for Feature Importance
disp("🔍 Training Decision Tree for Feature Importance...");
treeModel = fitctree(X_train, y_train);

% Extract feature importance scores
feature_scores = predictorImportance(treeModel);

%% Step 3: Visualize Feature Importance
% Sort features based on importance
[sorted_importance, sorted_idx] = sort(feature_scores, 'descend');

% Save feature indices (sorted by importance)
writematrix(sorted_idx, 'feature_importance.txt');

% Load feature names
features = readtable('dataset/features.txt', 'ReadVariableNames', false, 'TextType', 'string');
feature_names = features.Var1; % Assuming features.txt has one feature per row

% Get top 20 most important features
top_k = 30;
top_features = feature_names(sorted_idx(1:top_k));
top_importance = sorted_importance(1:top_k);

% Plot the top important features
figure;
barh(top_importance);
set(gca, 'yticklabel', top_features);
xlabel('Importance Score');
ylabel('Feature');
title('Top 30 Most Important Features');
grid on;

disp("✅ Feature importance analysis completed!");
