"""
uncertainty.py — Belirsizlik (Uncertainty) ve Kalibrasyon Metrikleri

İçerik:
- Expected Calibration Error (ECE)
- Predictive Entropy hesaplama
- Reliability Diagram (Kalibrasyon Eğrisi) çizimi
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def calculate_ece(probs, labels, n_bins=10):
    """
    Expected Calibration Error (ECE) hesaplar.
    
    Args:
        probs (np.ndarray): Sınıf olasılıkları (N, C).
        labels (np.ndarray): Gerçek etiketler (N,).
        n_bins (int): Bin sayısı.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
    return ece

def calculate_entropy(probs):
    """
    Predictive Entropy hesaplar. (N, C) -> (N,)
    """
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=1)

def plot_reliability_diagram(probs, labels, n_bins=10, save_path=None):
    """Kalibrasyon eğrisini (Reliability Diagram) çizer."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels)

    bin_accs = []
    bin_confs = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        if in_bin.sum() > 0:
            bin_accs.append(accuracies[in_bin].mean())
            bin_confs.append(confidences[in_bin].mean())
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2)

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
    plt.plot(bin_confs, bin_accs, marker='o', color='blue', label='Model')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def evaluate_metrics(probs, labels):
    """
    Accuracy, F1, ve AUC-ROC hesaplar.
    Binary classification varsayımıyla.
    """
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    
    if probs.shape[1] == 2:
        auc = roc_auc_score(labels, probs[:, 1])
    else:
        auc = roc_auc_score(labels, probs, multi_class='ovr')
        
    ece = calculate_ece(probs, labels)
    
    return {
        "Accuracy": acc,
        "F1-Score": f1,
        "AUC-ROC": auc,
        "ECE": ece
    }
