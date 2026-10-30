# THIS MODEL SUCKS!!!!
from sklearn.utils import compute_sample_weight
from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss

print("v0 train start")

trainset = pd.read_csv('data/v0_train.csv', low_memory=False)
X_train = trainset.drop(columns=['P_ISEV'])
y_train = trainset[['P_ISEV']]
testset = pd.read_csv('data/v0_eval.csv', low_memory=False)
X_test = testset.drop(columns=['P_ISEV'])
y_test = testset[['P_ISEV']]

# sadly! XGBoost didnt filter out the non-numerics; convert to categories.
# y's were prev put into numerics so thats okay
for col in X_train.select_dtypes(include=['object', 'string']).columns:
    X_train[col] = X_train[col].astype('category')
    
for col in X_train.columns:
    if isinstance(X_train[col].dtype, pd.CategoricalDtype):
        X_test[col] = X_test[col].astype(X_train[col].dtype)
    else:
        X_test[col] = X_test[col].astype(X_train[col].dtype)

labels = y_train.squeeze()
sample_weights = compute_sample_weight(class_weight="balanced", y=labels)
sample_weights[labels.to_numpy() == 2] *= 100 
# boost death rates because otherwise it will NEVER predict death
bst = XGBClassifier(
    objective = 'multi:softprob',
    num_class = 3,
    n_estimators = 2000, 
    max_depth = 8, 
    learning_rate = 0.02,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 3,
    reg_lambda = 1.0,
    enable_categorical = True,
    tree_method = 'hist',
    eval_metric = 'mlogloss',
    early_stopping_rounds = 60,
    random_state = 67,
    n_jobs = -1,
    )

bst.fit(
    X_train, y_train,
    sample_weight = sample_weights,
    eval_set = [(X_test, y_test)],
    verbose = 50
    )
proba = bst.predict_proba(X_test)
preds = bst.predict(X_test)

# check scores:
print(f"Log loss : {log_loss(y_test, proba):.4f}")
print(f"Accuracy : {accuracy_score(y_test, preds):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, preds, target_names=['No injury', 'Injury', 'Fatality']))

#special check: whats the avg softprob for fatality for each case?
print(pd.DataFrame({"actual": y_test.squeeze().to_numpy(), "p_fatality": proba[:, 2]})
      .groupby("actual")["p_fatality"].mean()
      .rename(index={0: "No injury", 1: "Injury", 2: "Fatality"})) 
    # wowza!! it works!!! when fatality is predicted, the softprob for fatality is significantly higher than BOTH no-injury and injury (0.011, vs 0.015, vs 0.056)!!!!
print("done :3")

