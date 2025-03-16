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

%% Step 2: Tune SVM Kernel
disp("🔄 Tuning SVM...");

kernels = ["linear", "rbf", "polynomial"];
accuracies = zeros(length(kernels), 1);

for i = 1:length(kernels)
    kernel = kernels(i);
    disp("Training SVM with " + kernel + " kernel...");
    
    % Create an SVM template with the selected kernel
    t = templateSVM('KernelFunction', kernel);

    % Train SVM model using fitcecoc
    svmModel = fitcecoc(X_train, y_train, 'Learners', t);
    
    % Predict on test data
    y_pred_svm = predict(svmModel, X_test);
    
    % Compute accuracy
    accuracies(i) = sum(y_pred_svm == y_test) / length(y_test) * 100;
    
    disp("✅ Accuracy for " + kernel + " kernel: " + accuracies(i) + "%");
end

%% Step 3: Plot Accuracy vs Kernel
figure;
bar(categorical(kernels), accuracies);
xlabel('Kernel Type');
ylabel('Accuracy (%)');
title('SVM Hyperparameter Tuning');
grid on;

% Find best kernel
[best_acc, best_idx] = max(accuracies);
best_kernel = kernels(best_idx);
disp("🏆 Best Kernel: " + best_kernel + " with accuracy: " + best_acc + "%");
