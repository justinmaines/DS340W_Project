# Library Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Reading data
data = pd.read_csv("data/Loan_status_2007-2020Q3.gzip")

# Data preprocessing
data_subset = data[["loan_amnt", "term", "int_rate", "grade", "sub_grade", "home_ownership", "purpose", "dti", "loan_status", "revol_bal"]].dropna()
data_subset["loan_status"] = np.where((data_subset["loan_status"] == "Fully Paid") | (data_subset["loan_status"] == "Current"), 0, 1)
data_subset[["term", "grade", "sub_grade", "home_ownership", "purpose"]] = data_subset[["term", "grade", "sub_grade", "home_ownership", "purpose"]].apply(LabelEncoder().fit_transform)
data_subset["int_rate"] = data_subset["int_rate"].str.strip("%")
data_subset["int_rate"] = (data_subset["int_rate"].astype('float64'))/100
X = data_subset.drop("loan_status", axis = 1)
y = data_subset["loan_status"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 42)

# Model Building
DT = DecisionTreeClassifier()
DT.fit(X_train, y_train)
DT_y_pred = DT.predict(X_test)

RF = RandomForestClassifier(n_estimators = 100)
RF.fit(X_train, y_train)
RF_y_pred = RF.predict(X_test)

# Model Evaluation
print(f"Decision Tree results:\n{classification_report(y_test, DT_y_pred)})")
print(f"Random Forest results:\n{classification_report(y_test, RF_y_pred)})")
