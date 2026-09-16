# Employee Attrition Prediction & AI Model Explainability

## 📌 Project Overview

This project focuses on predicting employee attrition using Machine Learning and explaining the predictions using Explainable AI (XAI) techniques.

The project uses the IBM HR Employee Attrition dataset to analyze employee-related factors and identify patterns associated with employee turnover. Machine Learning models are trained to predict whether an employee is likely to leave the organization.

Along with prediction, SHAP (SHapley Additive exPlanations) is used to explain how individual features contribute to the model's predictions. This makes the Machine Learning model more interpretable and helps understand the factors influencing employee attrition.

## 🎯 Objectives

- Analyze employee data and identify patterns related to attrition.
- Perform data cleaning and preprocessing.
- Explore relationships between employee characteristics and attrition.
- Build Machine Learning models for employee attrition prediction.
- Handle class imbalance using balanced Machine Learning models.
- Evaluate model performance using classification metrics.
- Use SHAP for model explainability.
- Generate individual and global feature explanations.
- Export prediction and feature-importance results.
- Visualize important findings using charts and a Tableau dashboard.

## 📊 Dataset

The project uses the **IBM HR Employee Attrition dataset**.

- Number of records: 1,470
- Number of original features: 35
- Target variable: `Attrition`
- `No` → 0
- `Yes` → 1

The dataset contains employee information such as:

- Age
- BusinessTravel
- Department
- JobRole
- JobLevel
- MonthlyIncome
- OverTime
- StockOptionLevel
- TotalWorkingYears
- YearsAtCompany
- YearsInCurrentRole
- YearsWithCurrManager
- JobSatisfaction
- EnvironmentSatisfaction
- WorkLifeBalance
- and other employee-related attributes.

## 🧹 Data Preprocessing

The following columns were removed because they do not provide useful information for prediction:

- `EmployeeCount`
- `Over18`
- `StandardHours`
- `EmployeeNumber`

The target variable `Attrition` was converted into binary values:

- `No` → 0
- `Yes` → 1

The dataset was divided into training and testing sets using an 80/20 split with stratification and `random_state=42`.

### Feature Preprocessing

Numerical features were standardized using `StandardScaler`.

Categorical features were converted into numerical form using `OneHotEncoder` with:

- `drop='first'`
- `handle_unknown='ignore'`

A `ColumnTransformer` and Machine Learning pipeline were used to organize the preprocessing and modeling workflow.

## 🤖 Machine Learning Models

The project implements:

### 1. Logistic Regression

Logistic Regression was used as one of the classification models for predicting employee attrition.

### 2. Random Forest

Random Forest was used as a tree-based classification model.

### 3. Balanced Models

Because the dataset contains considerably more employees who stayed than employees who left, balanced versions of the models were also evaluated using:

```python
class_weight="balanced"