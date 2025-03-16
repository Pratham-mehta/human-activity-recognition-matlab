%% Step 1: Load Data
clc; clear; close all;

% Load prepared data
X_train = readmatrix('dataset/Train/X_train.txt');
y_train = readmatrix('dataset/Train/y_train.txt');
X_test = readmatrix('dataset/Test/X_test.txt');
y_test = readmatrix('dataset/Test/y_test.txt');

% Normalize data
X_train = normalize(X_train, 'range');
X_test = normalize(X_test, 'range');

%% Step 2: Tune k-NN for Best 'k'
disp("🔄 Tuning k-NN...");

k_values = [1, 3, 5, 7, 9, 15, 20]; % Different values of k to test
accuracies = zeros(length(k_values), 1);

for i = 1:length(k_values)
    k = k_values(i);
    disp("Training k-NN with k=" + k + "...");
    
    % Train k-NN model
    knnModel = fitcknn(X_train, y_train, 'NumNeighbors', k);
    
    % Predict on test data
    y_pred_knn = predict(knnModel, X_test);
    
    % Compute accuracy
    accuracies(i) = sum(y_pred_knn == y_test) / length(y_test) * 100;
    
    disp("✅ Accuracy for k=" + k + ": " + accuracies(i) + "%");
end

%% Step 3: Plot Accuracy vs k
figure;
plot(k_values, accuracies, '-o', 'LineWidth', 2);
xlabel('k (Number of Neighbors)');
ylabel('Accuracy (%)');
title('k-NN Hyperparameter Tuning');
grid on;

% Find best k
[best_acc, best_idx] = max(accuracies);
best_k = k_values(best_idx);
disp("🏆 Best k: " + best_k + " with accuracy: " + best_acc + "%");