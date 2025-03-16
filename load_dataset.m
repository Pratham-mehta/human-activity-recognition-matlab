%% Step 1: Load Features from 'features.txt'
clc; clear; close all; % Clear workspace and close figures

features = readtable('dataset/features.txt', 'Delimiter', '\t', 'ReadVariableNames', false, 'TextType', 'string');
features = features.Var1;

%% Step 2: Load Training Data
X_train = readmatrix('dataset/Train/X_train.txt'); % Feature vectors
y_train = readmatrix('dataset/Train/y_train.txt'); % Activity labels
subjects_train = readmatrix('dataset/Train/subject_id_train.txt'); % Subject IDs

%% Step 3: Load Activity Labels
activity_labels = readtable('dataset/activity_labels.txt', 'Delimiter', ' ', 'ReadVariableNames', false);
activity_labels.Properties.VariableNames = {'ID', 'Activity'};

%% Step 4: Display Dataset Information
disp("✅ Loaded Data:");
disp("Number of Training Samples: " + size(X_train, 1));
disp("Number of Features: " + size(X_train, 2));
disp("Unique Activities:");
disp(unique(y_train)');