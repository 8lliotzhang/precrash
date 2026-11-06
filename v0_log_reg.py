import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss, accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

print("v0 logistic regression start")

cats = [
    'C_MNTH', 'C_HOUR', 'C_RCFG', 'C_WTHR', 'C_RSUR',
    'C_RALN', 'C_TRAF', 'V_YEAR', 'P_SEX', 'P_AGE', 'P_SAFE'
]

# Read categoricals as raw strings so "23" never becomes "23.0" in only one file
dtypes = {c: str for c in cats}

trainset = pd.read_csv('data/v0_train.csv', dtype=dtypes, low_memory=False)
X_train = trainset.drop(columns=['P_ISEV'])
y_train = trainset['P_ISEV']          # Series, not a one-column DataFrame

testset = pd.read_csv('data/v0_eval.csv', dtype=dtypes, low_memory=False)
X_test = testset.drop(columns=['P_ISEV'])
y_test = testset['P_ISEV']

for frame in (X_train, X_test):
    frame[cats] = frame[cats].fillna('__MISSING__')

numeric = [c for c in X_train.columns if c not in cats]

preprocessor = ColumnTransformer([
    ('categorical', OneHotEncoder(handle_unknown='ignore'), cats),
    ('numeric', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric)
])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=2000, solver='lbfgs'))
])
model.fit(X_train, y_train)

classes = list(model.named_steps['classifier'].classes_)

y_pred_proba = model.predict_proba(X_test)
y_pred_class = model.predict(X_test)

print("Log Loss:", log_loss(y_test, y_pred_proba, labels=classes))
print("Accuracy:", accuracy_score(y_test, y_pred_class))

print(classification_report(y_test, y_pred_class, zero_division=0))
print(confusion_matrix(y_test, y_pred_class, labels=[0, 1, 2]))

# Average fatality probability by actual class
fatality_col = classes.index(2)
p_fatal = pd.Series(y_pred_proba[:, fatality_col], index=y_test.index)

print(
    p_fatal.groupby(y_test)
    .mean()
    .rename(index={0: "No injury", 1: "Injury", 2: "Fatality"})
)

# Binary fatality metrics
y_fatal = (y_test == 2).astype(int)
print(f"Fatality ROC-AUC: {roc_auc_score(y_fatal, p_fatal):.5f}")
print(f"Fatality PR-AUC: {average_precision_score(y_fatal, p_fatal):.5f}")

print(type(y_pred_proba))
print(y_pred_proba.shape)
print(y_pred_proba[:5])

# Fatality probabilities for actual fatality cases (original row index preserved)
fatality_probs = p_fatal[y_test == 2].rename("Fatality probability")

print(fatality_probs.describe(percentiles=[.10, .25, .50, .75, .90, .95, .99]))

min_idx = fatality_probs.idxmin()
max_idx = fatality_probs.idxmax()

print("Minimum:")
print("Index:", min_idx)
print("Probability:", fatality_probs.min())

print("\nMaximum:")
print("Index:", max_idx)
print("Probability:", fatality_probs.max())

# remember: in eval_dead, ++2 to the given idx for the matching row: idx starts at 0, row 1 is header
print("done :3")