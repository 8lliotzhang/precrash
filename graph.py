import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve, roc_curve, average_precision_score,
    roc_auc_score, ConfusionMatrixDisplay
)

# name shown on the plots -> predictions file
models = {
    'Logistic regression': 'outputs/logreg_preds.csv',
    'LightGBM': 'outputs/lgbm_preds.csv',
    'CatBoost': 'outputs/catboost_preds.csv',
}

# Load whichever files exist, so you can graph before every model has been run
results = {name: pd.read_csv(path) for name, path in models.items() if os.path.exists(path)}
if not results:
    raise SystemExit("No prediction files found in outputs/. Run a model script first.")

# atality PR and ROC curves, all models overlaid 
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for name, df in results.items():
    y_fatal = (df['y_true'] == 2).astype(int)

    prec, rec, _ = precision_recall_curve(y_fatal, df['p_fatal'])
    ap = average_precision_score(y_fatal, df['p_fatal'])
    axes[0].plot(rec, prec, label=f'{name} (AP = {ap:.3f})')

    fpr, tpr, _ = roc_curve(y_fatal, df['p_fatal'])
    auc = roc_auc_score(y_fatal, df['p_fatal'])
    axes[1].plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')

# Baseline for PR = overall fatality rate (same eval set for every model)
base_rate = (next(iter(results.values()))['y_true'] == 2).mean()
axes[0].axhline(base_rate, linestyle='--', color='gray', label='Baseline')
axes[0].set(xlabel='Recall', ylabel='Precision', title='Fatality precision-recall')
axes[0].legend()

axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray')
axes[1].set(xlabel='False positive rate', ylabel='True positive rate', title='Fatality ROC')
axes[1].legend()

plt.tight_layout()
plt.savefig('outputs/curves.png', dpi=150)

# One confusion matrix per model
fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 4.5), squeeze=False)
for ax, (name, df) in zip(axes[0], results.items()):
    ConfusionMatrixDisplay.from_predictions(
        df['y_true'], df['y_pred'], labels=[0, 1, 2],
        display_labels=['No injury', 'Injury', 'Fatality'],
        normalize='true', ax=ax, colorbar=False
    )
    ax.set_title(name)
plt.tight_layout()
plt.savefig('outputs/confusion.png', dpi=150)

# CatBoost feature importance
if os.path.exists('outputs/catboost_importance.csv'):
    imp = pd.read_csv('outputs/catboost_importance.csv', index_col=0).squeeze('columns')
    plt.figure(figsize=(8, 6))
    imp.nlargest(15).sort_values().plot.barh()
    plt.title('CatBoost top 15 feature importances')
    plt.tight_layout()
    plt.savefig('outputs/importance.png', dpi=150)

plt.show()