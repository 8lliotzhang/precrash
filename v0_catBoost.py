import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    log_loss, accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, average_precision_score, precision_recall_curve
)
from sklearn.model_selection import train_test_split

print("v0 catboost start")

cats = [
    'C_MNTH', 'C_HOUR', 'C_RCFG', 'C_WTHR', 'C_RSUR',
    'C_RALN', 'C_TRAF', 'V_YEAR', 'P_SEX', 'P_AGE', 'P_SAFE'
]
dtypes = {c: str for c in cats}

trainset = pd.read_csv('data/v0_train.csv', dtype=dtypes, low_memory=False)
X_train = trainset.drop(columns=['P_ISEV'])
y_train = trainset['P_ISEV']          # Series, not a one-column DataFrame

testset = pd.read_csv('data/v0_eval.csv', dtype=dtypes, low_memory=False)
X_test = testset.drop(columns=['P_ISEV'])
y_test = testset['P_ISEV']

# CatBoost needs categoricals with no NaN
for frame in (X_train, X_test):
    frame[cats] = frame[cats].fillna('__MISSING__')

# Hold out part of the TRAIN data for early stopping AND threshold tuning
# so the eval set stays untouched until the final scoring
X_fit, X_val, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
)

# specialized CatBoost data input thing
cbtrain = Pool(X_fit, label=y_fit, cat_features=cats)
cbval = Pool(X_val, label=y_val, cat_features=cats)
cbtest = Pool(X_test, label=y_test, cat_features=cats)

# [0 = no injury, 1 = injury, 2 = fatality].
# Fatalities = rare so each fatality row counts FATALITY_WEIGHT times as much
# as a no-injury row during training.... try values between 10 and 50 and compare??
FATALITY_WEIGHT = 20
class_weights = [1, 1, FATALITY_WEIGHT]

bst = CatBoostClassifier(
    loss_function='MultiClass',
    eval_metric='MultiClass',          # multi-class log loss
    learning_rate=0.04,
    iterations=1000,
    early_stopping_rounds=100,         # stop if validation loss hasn't improved for 100 rounds
    class_weights=class_weights,       # upweight the rare fatality class
    random_seed=42,
    verbose=100
)
bst.fit(cbtrain, eval_set=cbval, use_best_model=True)

# Class order CatBoost uses for the probability columns (should be [0, 1, 2])
classes = list(bst.classes_)
fatality_col = classes.index(2)

# Plain argmax rarely predicts the rare class...
# so.. flag a crash as a fatality whenever P(fatality) >= threshold
# and pick the threshold that
# maximizes F1 for the fatality class on data the model hasn't trained on
val_proba = bst.predict_proba(cbval)
p_val_fatal = val_proba[:, fatality_col]
y_val_fatal = (y_val == 2).astype(int)

# precision_recall_curve returns one more precision/recall value than thresholds
# so the final point (precision=1, recall=0) is dropped with [:-1]
precision, recall, thresholds = precision_recall_curve(y_val_fatal, p_val_fatal)
f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
best_i = np.argmax(f1_scores)
fatality_threshold = thresholds[best_i]

print(f"Best iteration: {bst.get_best_iteration()}")
print(f"Validation fatalities: {int(y_val_fatal.sum())} (threshold is noisy if this is small)")
print(f"Chosen fatality threshold: {fatality_threshold:.5f} "
      f"(val precision {precision[best_i]:.3f}, recall {recall[best_i]:.3f})")

# Predictions on the eval set
y_pred_proba = bst.predict_proba(cbtest)
y_pred_class = np.asarray(classes)[np.argmax(y_pred_proba, axis=1)]   # plain argmax
 
p_fatal = pd.Series(y_pred_proba[:, fatality_col], index=y_test.index)

# ignore the fatality column and pick the more likely of no injury / injury
proba_no_fatal = y_pred_proba.copy()
proba_no_fatal[:, fatality_col] = -np.inf
y_pred_thresh = np.asarray(classes)[np.argmax(proba_no_fatal, axis=1)]
# override with "fatality" wherever P(fatality) clears the tuned threshold
y_pred_thresh[p_fatal.to_numpy() >= fatality_threshold] = 2

# the average fatality probabilities below are NOT calibrated risk estimates
print("Log Loss:", log_loss(y_test, y_pred_proba, labels=classes))
print("Accuracy:", accuracy_score(y_test, y_pred_class))

print(classification_report(y_test, y_pred_class, zero_division=0))
print(confusion_matrix(y_test, y_pred_class, labels=[0, 1, 2]))

# ---------- NEW: Evaluation with the tuned fatality threshold ----------
print("\n--- With tuned fatality threshold ---")
print("Accuracy:", accuracy_score(y_test, y_pred_thresh))
print(classification_report(y_test, y_pred_thresh, zero_division=0))
print(confusion_matrix(y_test, y_pred_thresh, labels=[0, 1, 2]))

# Average fatality probability by actual class
print(
    p_fatal.groupby(y_test)
    .mean()
    .rename(index={0: "No injury", 1: "Injury", 2: "Fatality"})
)

# Convert multiclass labels into binary fatality labels
y_fatal = (y_test == 2).astype(int)

# ROC-AUC and PR-AUC for fatality 
auc = roc_auc_score(y_fatal, p_fatal)
print(f"Fatality ROC-AUC: {auc:.5f}")
pr_auc = average_precision_score(y_fatal, p_fatal)
print(f"Fatality PR-AUC: {pr_auc:.5f}")
# - huh
print(type(y_pred_proba))
print(y_pred_proba.shape)
print(y_pred_proba[:5])
# - huh

# Fatality probabilities for actual fatality cases 
fatality_probs = p_fatal[y_test == 2].rename("Fatality probability")

print(fatality_probs.describe(
    percentiles=[.10, .25, .50, .75, .90, .95, .99]
))

min_idx = fatality_probs.idxmin()
max_idx = fatality_probs.idxmax()

print("Minimum:")
print("Index:", min_idx)
print("Probability:", fatality_probs.min())

print("\nMaximum:")
print("Index:", max_idx)
print("Probability:", fatality_probs.max())
# remember: in eval_dead, ++2 to the given idx for the matching row: idx starts at 0,  row 1 is header
print("done :3")