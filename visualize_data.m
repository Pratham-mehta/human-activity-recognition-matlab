%% Step 1: Load Training Data
clc; clear; close all;

X_train = readmatrix('dataset/Train/X_train.txt'); % Feature vectors
y_train = readmatrix('dataset/Train/y_train.txt'); % Activity labels

%% Step 2: Visualize First Sample (First 50 Features)
figure;
plot(X_train(1, 1:50)); % Plot first 50 features of first sample
title('Sensor Data Visualization (First 50 Features)');
xlabel('Feature Index');
ylabel('Feature Value');
grid on;

%% Step 3: Visualize Activity Distribution
figure;
histogram(y_train, 'BinMethod', 'integers');
title('Activity Distribution in Training Data');
xlabel('Activity Label');
ylabel('Frequency');
grid on;