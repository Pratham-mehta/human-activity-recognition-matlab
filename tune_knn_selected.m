%% Step 1: Load Data
clc; clear; close all;

% Load reduced dataset with top 30 features
X_train = readmatrix('dataset/Train/X_train_selected.txt');
y_train = readmatrix('dataset/Train/y_train_selected.txt');
X_test = readmatrix('dataset/Test/X_test_selected.txt');
y_test = readmatrix('dataset/Test/y_test_selected.txt');

% Define k values to test
k_values = [1, 3, 5, 7, 9, 11, 15, 20];
accuracies = zeros(length(k_values), 1);

disp("🔄 Tuning k-NN on Top 30 Features...");

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

%% Step 2: Plot Accuracy vs k
figure;
plot(k_values, accuracies, '-o', 'LineWidth', 2);
xlabel('k (Number of Neighbors)');
ylabel('Accuracy (%)');
title('k-NN Hyperparameter Tuning (Top 30 Features)');
grid on;

% Find the best k
[best_acc, best_idx] = max(accuracies);
best_k = k_values(best_idx);
disp("🏆 Best k: " + best_k + " with accuracy: " + best_acc + "%");

% Save the best k value for training models
save('best_k.mat', 'best_k');