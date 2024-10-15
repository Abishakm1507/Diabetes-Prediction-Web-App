import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load the data
data = pd.read_csv('diabetes.csv')

print(data.head())

# Data Preprocessing
X = data.drop(columns='Outcome')
y = data['Outcome']

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=0)

# Hyperparameter tuning for Random Forest
rf = RandomForestClassifier(random_state=0)
rf_param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf_grid = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy')
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# Hyperparameter tuning for XGBoost

xgb = XGBClassifier(eval_metric='logloss')
xgb_param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7]
}

# GridSearchCV for hyperparameter tuning
xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=5, scoring='accuracy')
xgb_grid.fit(X_train, y_train)

# Best XGBoost model after tuning
best_xgb = xgb_grid.best_estimator_

# Checking for and handling NaN/Infinity values
np.isnan(X_train).sum()
np.isnan(y_train).sum()
np.isinf(X_train).sum()
np.isinf(y_train).sum()

# Handling NaN/Infinity by replacing with mean (if applicable)
imputer = SimpleImputer(strategy='mean')
X_train = imputer.fit_transform(X_train)

# KNN model with hyperparameter tuning for optimal k
knn = KNeighborsClassifier()
knn_param_grid = {
    'n_neighbors': range(1, 31)
}
knn_grid = GridSearchCV(knn, knn_param_grid, cv=5, scoring='accuracy')
knn_grid.fit(X_train, y_train)

# Best KNN model after tuning
best_knn = knn_grid.best_estimator_

# Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('rf', best_rf),
    ('xgb', best_xgb),
    ('knn', best_knn)],
    voting='hard')

voting_clf.fit(X_train, y_train)
# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler object
pickle.dump(scaler, open("scaler.pkl", "wb"))


# Make pickle file of our model
pickle.dump(voting_clf, open("model.pkl", "wb"))
