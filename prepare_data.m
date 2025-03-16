%% Step 1: Load Data
clc; clear; close all;

% Load training data
X_train = readmatrix('dataset/Train/X_train.txt'); % Feature vectors
y_train = readmatrix('dataset/Train/y_train.txt'); % Activity labels

% Load test data
X_test = readmatrix('dataset/Test/X_test.txt');
y_test = readmatrix('dataset/Test/y_test.txt');

%% Step 2: Normalize the Data (Scaling between 0 and 1)
X_train = normalize(X_train, 'range');
X_test = normalize(X_test, 'range');

%% Step 3: Display Data Information
disp("✅ Data Prepared:");
disp("Training Samples: " + size(X_train, 1));
disp("Test Samples: " + size(X_test, 1));
disp("Number of Features: " + size(X_train, 2));