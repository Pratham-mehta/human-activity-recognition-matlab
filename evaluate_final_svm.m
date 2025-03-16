%% Step 1: Load Data
clc; clear; close all;

% Load reduced dataset (560 selected features)
X_train = readmatrix('dataset/Train/X_train_selected_rfe.txt');
y_train = readmatrix('dataset/Train/y_train_selected.txt');
X_test = readmatrix('dataset/Test/X_test_selected_rfe.txt');
y_test = readmatrix('dataset/Test/y_test_selected.txt');

% Normalize data
X_train = normalize(X_train, 'range');
X_test = normalize(X_test, 'range');

%% Step 2: Train Final SVM Model
disp("🔄 Training Final SVM Model (C=10, Linear Kernel)...");
finalSVMModel = fitcecoc(X_train, y_train, 'Learners', templateSVM('KernelFunction', 'linear', 'BoxConstraint', 10));

% Predict on test data
y_pred_svm = predict(finalSVMModel, X_test);

%% Step 3: Compute Metrics
disp("✅ Evaluating Model Performance...");

% Compute confusion matrix
confMat = confusionmat(y_test, y_pred_svm);

% Compute precision, recall, and F1-score
num_classes = size(confMat, 1);
precision = zeros(num_classes, 1);
recall = zeros(num_classes, 1);
f1_score = zeros(num_classes, 1);

for i = 1:num_classes
    TP = confMat(i, i);
    FP = sum(confMat(:, i)) - TP;
    FN = sum(confMat(i, :)) - TP;
    
    precision(i) = TP / (TP + FP + eps);
    recall(i) = TP / (TP + FN + eps);
    f1_score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i) + eps);
end

% Compute overall accuracy
accuracy = sum(diag(confMat)) / sum(confMat(:)) * 100;

%% Step 4: Display Results
disp("🎯 Final Model Evaluation:");
disp("✅ Overall Accuracy: " + accuracy + "%");
disp("🔹 Precision: "), disp(precision);
disp("🔹 Recall: "), disp(recall);
disp("🔹 F1-Score: "), disp(f1_score);

% Display confusion matrix
figure;
heatmap(confMat);
xlabel('Predicted Class');
ylabel('True Class');
title('Confusion Matrix - Final SVM Model');
disp("📊 Confusion Matrix Generated.");