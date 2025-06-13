# 🏦 Loan Approval Prediction using Machine Learning

## 📌 Project Overview
This project uses machine learning to predict whether a loan application will be approved based on applicant information such as marital status, income, credit history, etc. It automates the decision-making process and helps financial institutions efficiently evaluate loan eligibility :contentReference[oaicite:1]{index=1}.

---

## 📁 Dataset Description
- **Source**: `LoanApprovalPrediction.csv` (from GeeksforGeeks tutorial)
- **Number of features**: 13
- **Key Features**:
  1. `Loan_ID` – unique identifier
  2. `Gender`
  3. `Married`
  4. `Dependents`
  5. `Education`
  6. `Self_Employed`
  7. `ApplicantIncome`
  8. `CoapplicantIncome`
  9. `LoanAmount`
  10. `Loan_Amount_Term`
  11. `Credit_History`
  12. `Property_Area`
  13. `Loan_Status` (target: Y/N)

---

## 🛠 Tools and Libraries
- Python 3  
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn  

---

## 🔧 Installation
```bash
pip install pandas numpy matplotlib seaborn scikit-learn

# 1. Import libraries
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 2. Load dataset
data = pd.read_csv('LoanApprovalPrediction.csv')

# 3. Explore data
print(data.head(), data.info(), data.describe())

# 4. Preprocessing
data.drop('Loan_ID', axis=1, inplace=True)
for col in data.select_dtypes(include='object'):
    data[col].fillna(data[col].mode()[0], inplace=True)
data.fillna(data.select_dtypes(include='number').mean(), inplace=True)

# Label encoding of categorical variables
le = LabelEncoder()
for col in data.select_dtypes(include='object'):
    data[col] = le.fit_transform(data[col])

# 5. Exploratory Data Analysis
plt.figure(figsize=(12,8))
for i, col in enumerate(data.select_dtypes(include='object').columns):
    plt.subplot(4,2,i+1)
    sns.countplot(data[col])
plt.tight_layout()
plt.show()

# 6. Feature correlation
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.show()

# 7. Model training
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize numerical features
num_cols = X.select_dtypes(include='number').columns
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Train models
models = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(random_state=42)
}
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(f"{name} Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

# 8. Model evaluation
# Compare accuracy and review classification reports for metrics like precision, recall, F1-score.
