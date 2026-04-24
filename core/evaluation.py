import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_recall_curve, auc, confusion_matrix

def calculate_fraud_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    """Calculates F2, PR-AUC and False Positive Rate."""
    
    # F2-score prioritizes recall (catching fraud)
    f2 = fbeta_score(y_true, y_pred, beta=2)
    
    # Precision-Recall AUC is better for imbalanced data than ROC-AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    
    # Calculate False Positive Rate (customer friction)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    fpr = fp / (fp + tn)
    
    return {
        "f2_score": f2,
        "pr_auc": pr_auc,
        "fpr": fpr,
        "recall": recall[1] if len(recall) > 1 else 0, # Catch rate
        "precision": precision[1] if len(precision) > 1 else 0 # Precision
    }
