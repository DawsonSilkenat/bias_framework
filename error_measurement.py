from sklearn.metrics import confusion_matrix


def get_error_metrics(y_true, y_pred) -> dict[str, float]:
    # Computes a number of error metrics as a batch and returns a dictionary containing the error metrics names and values
    # We use the variable names TN, FP, FN, TP for true negative, false positive, false negative, and true positive for easy comparison with formula
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics = dict()
    
    metrics["accuracy"]   = (tp + tn) / (tp + fp + tn + fn)
    metrics["precision"]  = tp / (tp + fp)
    
    metrics["recall"]     = tp / (tp + fn)
    
    
    metrics["alarm rate"] = fp / (fp + tn)
    
    
def test_metrics(y_test, y_pred):
    # Found the code the Bias Mitigation Methods paper used to generate their metrics. Want to verify mine are the same
    from sklearn.metrics import accuracy_score,recall_score,precision_score, f1_score,roc_auc_score,matthews_corrcoef
    accuracy        = accuracy_score(y_test, y_pred)
    recall1         = recall_score(y_test, y_pred, pos_label=1)
    recall0         = recall_score(y_test, y_pred, pos_label=0)
    recall_macro    = recall_score(y_test, y_pred, average='macro')
    precision1      = precision_score(y_test, y_pred, pos_label=1)
    precision0      = precision_score(y_test, y_pred, pos_label=0)
    precision_macro = precision_score(y_test, y_pred, average='macro')
    f1score1        = f1_score(y_test, y_pred, pos_label=1)
    f1score0        = f1_score(y_test, y_pred, pos_label=0)
    f1score_macro   = f1_score(y_test, y_pred, average='macro')
    mcc             = matthews_corrcoef(y_test, y_pred)
    
    
    