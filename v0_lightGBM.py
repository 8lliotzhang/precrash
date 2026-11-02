import pandas as pd
import lightgbm as lgb
from sklearn.metrics import log_loss, accuracy_score, classification_report, confusion_matrix, roc_auc_score, average_precision_score

import numpy as np
print("v0 lightGBM start")

# same initialization setup block here
trainset = pd.read_csv('data/v0_train.csv', low_memory=False)
X_train = trainset.drop(columns=['P_ISEV'])
y_train = trainset[['P_ISEV']]
testset = pd.read_csv('data/v0_eval.csv', low_memory=False)
X_test = testset.drop(columns=['P_ISEV'])
y_test = testset[['P_ISEV']]

lgbtrain = lgb.Dataset(X_train, label=y_train) 
# specialized LGBM data input thing ig
lgbtest = lgb.Dataset(X_test, label=y_test, reference = lgbtrain)

cats = [
    'C_MNTH', 'C_HOUR', 'C_RCFG', 'C_WTHR', 'C_RSUR',
    'C_RALN', 'C_TRAF', 'V_YEAR', 'P_SEX', 'P_AGE', 'P_SAFE'
]
for col in cats:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

param = {
    "learning_rate": 0.04,
    "metric": "multi_logloss",
    "task": "train",
    "objective": "multiclass",
    "num_class": 3,
    "early_stopping": 100

}

num_rounds = 1000
bst = lgb.train(param, lgbtrain, num_rounds, valid_sets=[lgbtest])

y_pred_proba = bst.predict(X_test)
y_pred_class = np.argmax(y_pred_proba, axis=1)
print("Log Loss:", log_loss(y_test, y_pred_proba))
print("Accuracy:", accuracy_score(y_test, y_pred_class))

print(classification_report(y_test, y_pred_class))
print(confusion_matrix(y_test, y_pred_class))

print(
    pd.DataFrame({
        "actual": y_test.squeeze().to_numpy(),
        "p_fatality": y_pred_proba[:, 2]
    })
    .groupby("actual")["p_fatality"]
    .mean()
    .rename(index={
        0: "No injury",
        1: "Injury",
        2: "Fatality"
    })
)

# Convert multiclass labels into binary fatality labels
y_fatal = (y_test == 2).astype(int)

# Extract fatality probabilities
p_fatal = y_pred_proba[:, 2]

# Calculate ROC-AUC and Precision-Recall AUC for fatality
auc = roc_auc_score(y_fatal, p_fatal)
print(f"Fatality ROC-AUC: {auc:.5f}")
pr_auc = average_precision_score(y_fatal, p_fatal)
print(f"Fatality PR-AUC: {pr_auc:.5f}")
# - huh
print(type(y_pred_proba))
print(y_pred_proba.shape)
print(y_pred_proba[:5])
# - huh
fatality_mask = (np.asarray(y_test).ravel() == 2)

fatality_probs = pd.Series(
    y_pred_proba[fatality_mask, 2],
    name="Fatality probability"
)

print(fatality_probs.describe(
    percentiles=[.10, .25, .50, .75, .90, .95, .99]
))
min_idx = np.argmin(fatality_probs)
max_idx = np.argmax(fatality_probs)

print("Minimum:")
print("Index:", fatality_probs.index[min_idx])
print("Probability:", fatality_probs.iloc[min_idx])

print("\nMaximum:")
print("Index:", fatality_probs.index[max_idx]) 
print("Probability:", fatality_probs.iloc[max_idx])
# remember: in eval_dead, ++2 to the given idx for the matching row: idx starts at 0,  row 1 is header 
print("done :3")