%% Recursive Feature Elimination (RFE) for Feature Selection
clc; clear; close all;

% Load prepared training data
X_train = readmatrix('dataset/Train/X_train.txt');
y_train = readmatrix('dataset/Train/y_train.txt');
X_test = readmatrix('dataset/Test/X_test.txt');
y_test = readmatrix('dataset/Test/y_test.txt');

% Normalize the data (ensuring all values are within the same range)
X_train = normalize(X_train, 'range');
X_test = normalize(X_test, 'range');

% Load feature names
features = readtable('dataset/features.txt', 'ReadVariableNames', false, 'TextType', 'string');
feature_names = features.Var1;

% Start with all features
num_features = size(X_train, 2);
selected_features = 1:num_features;

% Initialize accuracy tracking
best_acc = 0;
best_features = selected_features;

for i = 1:10 % Remove least important features iteratively
    disp("🔍 Iteration " + i + " - Features Left: " + length(selected_features));
    
    % Train Decision Tree
    treeModel = fitctree(X_train(:, selected_features), y_train);
    
    % Get feature importance
    feature_scores = predictorImportance(treeModel);
    
    % Sort features by importance
    [~, sorted_idx] = sort(feature_scores, 'descend');
    
    % Remove least important feature
    selected_features(sorted_idx(end)) = [];
    
    % Train SVM with the remaining features
    svmModel = fitcecoc(X_train(:, selected_features), y_train);
    y_pred = predict(svmModel, X_test(:, selected_features));
    acc = sum(y_pred == y_test) / length(y_test) * 100;
    
    % Track best feature subset
    if acc > best_acc
        best_acc = acc;
        best_features = selected_features;
    end
    
    disp("✅ SVM Accuracy with " + length(selected_features) + " features: " + acc + "%");
end

% Save best feature subset
writematrix(best_features, 'dataset/best_features.txt');

% Save reduced dataset using best features
X_train_selected = X_train(:, best_features);
X_test_selected = X_test(:, best_features);

writematrix(X_train_selected, 'dataset/Train/X_train_selected_rfe.txt');
writematrix(y_train, 'dataset/Train/y_train_selected.txt');
writematrix(X_test_selected, 'dataset/Test/X_test_selected_rfe.txt');
writematrix(y_test, 'dataset/Test/y_test_selected.txt');

disp("🏆 Best feature subset found with " + length(best_features) + " features (Accuracy: " + best_acc + "%)");
disp("✅ Reduced dataset saved!");