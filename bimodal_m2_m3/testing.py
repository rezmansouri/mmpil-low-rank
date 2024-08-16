import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

EPS = 1e-9

def tss(yhat, y):
    '''empty'''
    yhat = np.array(yhat)
    y = np.array(y)
    tp = ((y == 1) & (yhat == 1)).sum()
    fn = ((y == 1) & (yhat == 0)).sum()
    fp = ((y == 0) & (yhat == 1)).sum()
    tn = ((y == 0) & (yhat == 0)).sum()
    score = (tp / (tp + fn + EPS)) - (fp / (fp + tn + EPS))
    return score


def hss(yhat, y):
    '''empty'''
    yhat = np.array(yhat)
    y = np.array(y)
    tp = ((y == 1) & (yhat == 1)).sum()
    fn = ((y == 1) & (yhat == 0)).sum()
    fp = ((y == 0) & (yhat == 1)).sum()
    tn = ((y == 0) & (yhat == 0)).sum()
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn) + EPS
    score = numerator / denominator
    return score


def test(val_loader, test_loader, model, result_save_path):

    # finding best decision threshold on balanced validation data
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            modalities, y = batch
            out = model(modalities)
            out = out.cpu().numpy().squeeze()
            y = y.cpu().numpy().squeeze()
            y_pred.extend(list(out))
            y_true.extend(list(y))

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    print('optimal threshold:', optimal_threshold)
    
    # hss tss on unbalanced test data
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            modalities, y = batch
            out = model(modalities)
            out[out>=optimal_threshold] = 1
            out[out<optimal_threshold] = 0
            out = out.cpu().numpy().squeeze()
            y = y.cpu().numpy().squeeze()
            y_pred.extend(list(out))
            y_true.extend(list(y))
    
    tss_ = tss(y_pred, y_true)
    hss_ = hss(y_pred, y_true)
    print('tss:', tss_, 'hss:', hss_)
    
    df = pd.DataFrame(
        {
            'optimal_threshold_on_validation': [optimal_threshold],
            'tss_on_testing': [tss_],
            'hss_on_testing': [hss_],
        }
    )
    
    df.to_csv(result_save_path)
