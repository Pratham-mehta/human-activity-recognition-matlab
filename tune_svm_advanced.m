%% Advanced SVM Hyperparameter Tuning (C & Kernel)
clc; clear; close all;

% Load reduced dataset with optimal features
best_features = readmatrix('dataset/best_features.txt');
X_train = readmatrix('dataset/Train/X_train.txt');
y_train = readmatrix('dataset/Train/y_train.txt');
X_test = readmatrix('dataset/Test/X_test.txt');
y_test = readmatrix('dataset/Test/y_test.txt');

% Reduce dataset to selected features
X_train = X_train(:, best_features);
X_test = X_test(:, best_features);

% Normalize the data
X_train = normalize(X_train, 'range');
X_test = normalize(X_test, 'range');

% Define SVM parameters to tune
C_values = [0.1, 1, 10, 100];
kernels = ["linear", "polynomial", "rbf"];

best_acc = 0;
best_C = 1;
best_kernel = "linear";

for i = 1:length(kernels)
    for j = 1:length(C_values)
        C = C_values(j);
        kernel = kernels(i);
        
        disp("Training SVM with C=" + C + " and kernel=" + kernel + "...");
        
        % Train SVM Model
        t = templateSVM('KernelFunction', kernel, 'BoxConstraint', C);
        svmModel = fitcecoc(X_train, y_train, 'Learners', t);
        
        % Predict on test data
        y_pred = predict(svmModel, X_test);
        acc = sum(y_pred == y_test) / length(y_test) * 100;
        
        % Track best model
        if acc > best_acc
            best_acc = acc;
            best_C = C;
            best_kernel = kernel;
        end
        
        disp("✅ Accuracy for C=" + C + " kernel=" + kernel + ": " + acc + "%");
    end
end

disp("🏆 Best SVM Model: C=" + best_C + " Kernel=" + best_kernel + " Accuracy: " + best_acc + "%");