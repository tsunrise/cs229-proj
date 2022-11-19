import numpy as np
import sklearn.metrics as sklearn
import torch
def evaluate_performance(y_true, y_pred):
    metrics = [accuracy_emr, accuracy_samples,
               precision_micro, precision_macro, precision_weighted, precision_samples,
               recall_micro, recall_macro, recall_weighted, recall_samples,
               f1_micro, f1_macro, f1_weighted, f1_samples]

    results = {}
    for m in metrics:
        k, v = m(y_true, y_pred)
        results[k] = v
    return results

def accuracy_emr(y_true, y_pred):
    label = "exact_match_ratio"
    return label, sklearn.accuracy_score(y_true, y_pred)

def accuracy_samples(y_true, y_pred):
    label = "hamming_score"
    return label, np.mean(np.sum(1*(y_true==y_pred), axis=1) / y_true.shape[1])

def precision_micro(y_true, y_pred):
    label = "precision_micro"
    return label, sklearn.precision_score(y_true, y_pred, average='micro', zero_division=1)

def precision_macro(y_true, y_pred):
    label = "precision_macro"
    return label, sklearn.precision_score(y_true, y_pred, average='macro', zero_division=1)

def precision_weighted(y_true, y_pred):
    label = "precision_weighted"
    return label, sklearn.precision_score(y_true, y_pred, average='weighted', zero_division=1)

def precision_samples(y_true, y_pred):
    label = "precision_samples"
    return label, sklearn.precision_score(y_true, y_pred, average='samples', zero_division=1)

def recall_micro(y_true, y_pred):
    label = "recall_micro"
    return label, sklearn.recall_score(y_true, y_pred, average='micro', zero_division=1)

def recall_macro(y_true, y_pred):
    label = "recall_macro"
    return label, sklearn.recall_score(y_true, y_pred, average='macro', zero_division=1)

def recall_weighted(y_true, y_pred):
    label = "recall_weighted"
    return label, sklearn.recall_score(y_true, y_pred, average='weighted', zero_division=1)

def recall_samples(y_true, y_pred):
    label = "recall_samples"
    return label, sklearn.recall_score(y_true, y_pred, average='samples', zero_division=1)

def f1_micro(y_true, y_pred):
    label = "f1_micro"
    return label, sklearn.f1_score(y_true, y_pred, average='micro', zero_division=1)

def f1_macro(y_true, y_pred):
    label = "f1_macro"
    return label, sklearn.f1_score(y_true, y_pred, average='macro', zero_division=1)

def f1_weighted(y_true, y_pred):
    label = "f1_weighted"
    return label, sklearn.f1_score(y_true, y_pred, average='weighted', zero_division=1)

def f1_samples(y_true, y_pred):
    label = "f1_samples"
    return label, sklearn.f1_score(y_true, y_pred, average='samples', zero_division=1)

def logits_to_multi_hot(logits):
    if isinstance(logits, list):
        logits = np.array(logits)
    elif isinstance(logits, torch.Tensor):
        logits = logits.detach().cpu().numpy()
    elif isinstance(logits, np.ndarray):
        pass
    else:
        raise Exception("logits_to_multi_hot: unsupported type for logits")
    return (logits > 0).astype(int)