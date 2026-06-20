# ML-Classification-Project
Employee Attrition Prediction Using Machine Learning

👨‍💼 Employee Attrition Prediction Using Machine Learning
📌 Project Overview

Employee attrition can significantly impact an organization's productivity and costs. This project uses machine learning techniques to predict whether an employee is likely to leave the company based on HR analytics data.

The project covers the complete machine learning workflow, including data preprocessing, exploratory data analysis (EDA), model training, evaluation, hyperparameter tuning, and model persistence.

🎯 Objectives
Understand factors influencing employee attrition.
Analyze employee demographics and workplace characteristics.
Build predictive machine learning models for attrition prediction.
Compare multiple classification algorithms.
Identify the most important features affecting employee turnover.
🛠️ Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
XGBoost
Joblib
Jupyter Notebook
📊 Exploratory Data Analysis

The following analyses were performed:

Attrition Distribution
Age Distribution
Monthly Income Distribution
Age vs Attrition Analysis
Education Level Distribution
Pairplot Visualization
Age vs Monthly Income Scatter Analysis
⚙️ Data Preprocessing
Removed unnecessary columns
Encoded categorical variables using Label Encoding
Selected relevant features
Split data into training and testing sets
Standardized numerical features using StandardScaler
🤖 Machine Learning Models Evaluated
Logistic Regression
K-Nearest Neighbors (KNN)
Decision Tree
Random Forest
Support Vector Machine (SVM)
Naive Bayes
AdaBoost
Gradient Boosting
Extra Trees Classifier
XGBoost Classifier
📈 Evaluation Metrics

Models were evaluated using:

Accuracy
Precision
Recall
F1-Score
Confusion Matrix
Classification Report
🔧 Hyperparameter Tuning

GridSearchCV was used to optimize Logistic Regression parameters and improve model performance.

🔍 Feature Importance Analysis

Feature importance analysis was performed to identify the factors most strongly associated with employee attrition.

Key features include:

Monthly Income
Overtime
Years at Company
Job Role
Department
Job Satisfaction
Age
