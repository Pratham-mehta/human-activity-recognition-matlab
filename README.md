# Smartphone-Based Recognition of Human Activities & Postural Transitions

##  Project Overview
This project aims to classify human activities and postural transitions using smartphone sensor data. The dataset used is from the **UCI Machine Learning Repository**: [Smartphone-Based Recognition of Human Activities & Postural Transitions](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions). The dataset consists of accelerometer and gyroscope readings collected from a smartphone placed on subjects performing various activities.

### Why This Project?
Human activity recognition (HAR) has applications in **healthcare monitoring, fitness tracking, and smart environments**. By classifying activities such as walking, standing, and sitting, we can develop intelligent applications that assist users in daily life. This project focuses on training and optimizing machine learning models to **accurately classify human activities** based on smartphone sensor data.

---

## Dataset Description
- **Source**: UCI Machine Learning Repository
- **Features**: 561 features extracted from accelerometer and gyroscope data
- **Classes**: 12 human activities (e.g., Walking, Sitting, Standing, Lying Down, etc.)
- **Training Samples**: 7,767
- **Testing Samples**: 3,162

### Activity Labels:
1. Standing
2. Sitting
3. Lying down
4. Walking
5. Walking upstairs
6. Walking downstairs
7. Stand-to-sit
8. Sit-to-stand
9. Sit-to-lie
10. Lie-to-sit
11. Lie-to-stand
12. Stand-to-lie

---

## Data Preprocessing & Feature Engineering
### **1.Loading and Normalizing the Data**
- The dataset is loaded from `dataset/Train/` and `dataset/Test/`.
- Features are normalized to a range between **0 and 1**.

### **2.Initial Data Visualization**
- Exploratory Data Analysis (EDA) was conducted to understand feature distributions.
- Data was visualized to analyze activity-wise feature separation.

---

## Initial Model Training
Three baseline models were trained to establish a starting point:

| Model             | Accuracy (%) |
|------------------|-------------|
| Decision Tree    | 60.56       |
| k-NN (k=5)      | 87.25       |
| SVM (Linear)    | 89.88       |

- **k-NN and SVM performed well**, but Decision Tree had low accuracy.

---

## Hyperparameter Tuning & Feature Selection
We experimented with different approaches to improve model performance:

### **1.Hyperparameter Tuning**
- **k-NN**: We tested different values of **k** and found **k=7** optimal.
- **SVM**: We experimented with different kernel functions (`linear`, `polynomial`, `rbf`) and values of **C (Regularization Parameter)**.

### **2.Feature Selection Experiments**
We attempted multiple feature selection strategies:
1. **Top 20 Features** → Slight improvement in Decision Tree, but **k-NN and SVM accuracy dropped**.
2. **Top 30 Features** → Improved **k-NN accuracy**, but SVM performance dropped significantly.
3. **Recursive Feature Elimination (RFE)** → Selected **560 most relevant features**, improving overall accuracy.

---

## Final Model Selection & Evaluation
After performing **RFE-based feature selection** and **advanced hyperparameter tuning**, the final model was trained using **SVM with C=10 and a linear kernel**.

### **Final Model Performance:**
| Model             | Accuracy (%) |
|------------------|-------------|
| Final SVM Model (C=10, Linear) | 91.77 |

#### Confusion Matrix:
- The confusion matrix revealed strong classification performance, especially in static activities.
- Some misclassification occurred in transitions like `sit-to-stand` and `stand-to-lie`.

#### **Precision, Recall, and F1-Score:**
These metrics provided additional insights into per-class performance, helping analyze strengths and weaknesses.

---

## How to Run This Project on MATLAB Online
### **1.Clone the Repository:**
```bash
# Clone the repository
https://github.com/yourusername/smartphone-activity-recognition.git
cd smartphone-activity-recognition
```

### **2.Open MATLAB Online:**
- Go to [MATLAB Online](https://matlab.mathworks.com/)
- Upload the dataset and scripts

### **3.Run the Scripts in Order:**
1. **Prepare the Data:** `prepare_data.m`
2. **Train Initial Models:** `train_models.m`
3. **Hyperparameter Tuning:** `tune_knn.m`, `tune_svm.m`
4. **Feature Selection:** `compute_feature_importance.m`, `select_features.m`
5. **Train Final Model:** `train_models_selected.m`
6. **Evaluate Final Model:** `evaluate_final_svm.m`

---

## GitHub Repository Structure
```
smartphone-activity-recognition/
│── dataset/                     # Raw dataset
│   ├── Train/
│   ├── Test/
│   ├── features.txt
│── prepare_data.m                # Data preprocessing script
│── train_models.m                 # Initial model training
│── tune_knn.m                     # k-NN Hyperparameter tuning
│── tune_svm.m                     # SVM Hyperparameter tuning
│── compute_feature_importance.m   # Feature importance analysis
│── select_features.m              # Feature selection based on importance
│── rfe_feature_selection.m        # Recursive Feature Elimination (RFE)
│── tune_svm_advanced.m            # Advanced SVM tuning (C & Kernel)
│── train_models_selected.m        # Final training on selected features
│── evaluate_final_svm.m           # Final model evaluation and metrics
│── README.md                      # Documentation
```

---

## Key Takeaways & Lessons Learned
- **Feature selection is critical** – Selecting the right features improved **Decision Tree performance** but initially hurt SVM and k-NN.
- **Hyperparameter tuning is essential** – Fine-tuning `k` in k-NN and `C, Kernel` in SVM significantly improved accuracy.
- **RFE is powerful** – Recursive Feature Elimination helped retain **560 best features**, boosting **SVM accuracy to 91.77%**.
- **Confusion matrices help analysis** – We identified **where misclassifications occurred** and addressed them.

---

## Future Scope
- **Deep Learning Approach**: Testing **LSTM-based HAR models** for sequence-based activity recognition.
- **On-Device Deployment**: Implementing lightweight models for **real-time classification** on smartphones.
- **Data Augmentation**: Using **synthetic data generation** to improve classifier robustness.

---

## Contributing & Contact
Want to contribute or have questions? Feel free to reach out!

📧 Email: pm3483@nyu.edu
🔗 GitHub: [Pratham-mehta](https://github.com/pratham-mehta)  

Let's build smarter activity recognition systems together!

