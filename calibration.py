import torch
import numpy as np
from typing import Dict
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    threshold: float
    precision: float
    recall: float
    f1_score: float
    signal_ratio: float
    noise_ratio: float
    true_positives: int
    false_positives: int
    false_negatives: int


class ThresholdCalibrator:
    def __init__(self, min_precision=0.7, min_recall=0.5):
        self.min_precision = min_precision
        self.min_recall = min_recall
        self.curve = []
        
    def calibrate(self, preds, targets, thresholds=None):
        if thresholds is None:
            thresholds = np.linspace(preds.min(), preds.max(), 100)
        
        best = None
        best_f1 = 0
        
        for t in thresholds:
            result = self._eval_threshold(preds, targets, t)
            self.curve.append(result)
            
            if result.precision >= self.min_precision and result.recall >= self.min_recall:
                if result.f1_score > best_f1:
                    best_f1 = result.f1_score
                    best = result
        
        if best is None:
            best = max(self.curve, key=lambda x: x.f1_score)
        
        return best
    
    def _eval_threshold(self, preds, targets, thresh):
        pred_bin = (preds >= thresh).astype(int)
        
        tp = ((pred_bin == 1) & (targets == 1)).sum()
        fp = ((pred_bin == 1) & (targets == 0)).sum()
        fn = ((pred_bin == 0) & (targets == 1)).sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        
        sig_ratio = pred_bin.sum() / len(pred_bin)
        
        return CalibrationResult(
            threshold=thresh,
            precision=prec,
            recall=rec,
            f1_score=f1,
            signal_ratio=sig_ratio,
            noise_ratio=1 - sig_ratio,
            true_positives=int(tp),
            false_positives=int(fp),
            false_negatives=int(fn)
        )
    
    def get_pr_curve(self):
        prec = np.array([r.precision for r in self.curve])
        rec = np.array([r.recall for r in self.curve])
        thresh = np.array([r.threshold for r in self.curve])
        return prec, rec, thresh


class SignalNoiseController:
    def __init__(self, target_signal_ratio=0.3, tolerance=0.05):
        self.target_ratio = target_signal_ratio
        self.tolerance = tolerance
        
    def find_threshold(self, preds):
        sorted_p = np.sort(preds)
        idx = int(len(sorted_p) * (1 - self.target_ratio))
        return sorted_p[idx]
    
    def evaluate_quality(self, preds, targets, thresh):
        signals = preds >= thresh
        sig_ratio = signals.sum() / len(signals)
        
        sig_preds = preds[signals]
        sig_targets = targets[signals]
        
        if len(sig_preds) > 0:
            sig_acc = (sig_preds >= thresh).astype(int) == sig_targets
            sig_qual = sig_acc.mean()
        else:
            sig_qual = 0
        
        noise_preds = preds[~signals]
        noise_targets = targets[~signals]
        
        if len(noise_preds) > 0:
            noise_acc = (noise_preds < thresh).astype(int) == (1 - noise_targets)
            noise_qual = noise_acc.mean()
        else:
            noise_qual = 0
        
        return {
            'signal_ratio': sig_ratio,
            'signal_quality': sig_qual,
            'noise_ratio': 1 - sig_ratio,
            'noise_quality': noise_qual,
            'overall_quality': sig_ratio * sig_qual + (1 - sig_ratio) * noise_qual
        }


class EmpiricalStoppingCriteria:
    def __init__(self, patience=10, min_delta=0.001, metric='f1_score'):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        self.history = []
        self.best = -np.inf
        self.wait = 0
        
    def should_stop(self, metrics):
        val = metrics.get(self.metric, 0)
        self.history.append(val)
        
        if val > self.best + self.min_delta:
            self.best = val
            self.wait = 0
        else:
            self.wait += 1
        
        return self.wait >= self.patience
    
    def get_curve(self):
        return np.array(self.history)
    
    def is_converged(self, window=5):
        if len(self.history) < window:
            return False
        return np.var(self.history[-window:]) < self.min_delta ** 2


class PrecisionRecallOptimizer:
    def __init__(self, precision_weight=0.5):
        self.prec_w = precision_weight
        self.rec_w = 1 - precision_weight
        
    def optimize(self, preds, targets, thresholds=None):
        if thresholds is None:
            thresholds = np.percentile(preds, np.linspace(0, 100, 50))
        
        best_score = -np.inf
        best_thresh = thresholds[0]
        best_metrics = {}
        
        for t in thresholds:
            pred_bin = (preds >= t).astype(int)
            
            tp = ((pred_bin == 1) & (targets == 1)).sum()
            fp = ((pred_bin == 1) & (targets == 0)).sum()
            fn = ((pred_bin == 0) & (targets == 1)).sum()
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            score = self.prec_w * prec + self.rec_w * rec
            
            if score > best_score:
                best_score = score
                best_thresh = t
                best_metrics = {
                    'precision': prec,
                    'recall': rec,
                    'f1_score': 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0,
                    'weighted_score': score
                }
        
        return best_thresh, best_metrics


class EvaluationLoop:
    def __init__(self, calibrator, stopping):
        self.calibrator = calibrator
        self.stopping = stopping
        self.results = []
        
    def run(self, model, val_loader, max_iter=100):
        for i in range(max_iter):
            preds_list = []
            targets_list = []
            
            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    x, y = batch
                    p = model(x)
                    preds_list.append(p.cpu().numpy())
                    targets_list.append(y.cpu().numpy())
            
            preds = np.concatenate(preds_list)
            targets = np.concatenate(targets_list)
            
            result = self.calibrator.calibrate(preds, targets)
            self.results.append(result)
            
            metrics = {
                'f1_score': result.f1_score,
                'precision': result.precision,
                'recall': result.recall
            }
            
            if self.stopping.should_stop(metrics):
                break
        
        return {
            'best_result': max(self.results, key=lambda x: x.f1_score),
            'iterations': len(self.results),
            'converged': self.stopping.is_converged(),
            'curve': self.stopping.get_curve()
        }


def calibrate_regime_thresholds(vae_hmm, data, true_regimes, K):
    vae_hmm.eval()
    with torch.no_grad():
        probs = torch.softmax(vae_hmm.encode(data), dim=1).cpu().numpy()
    
    thresholds = {}
    for k in range(K):
        cal = ThresholdCalibrator(min_precision=0.6, min_recall=0.5)
        targets = (true_regimes == k).astype(int)
        preds = probs[:, k, :].mean(axis=1)
        result = cal.calibrate(preds, targets)
        thresholds[k] = result.threshold
    
    return thresholds


def evaluate_with_tradeoffs(preds, targets, weights=np.linspace(0, 1, 11)):
    import pandas as pd
    
    results = []
    for w in weights:
        opt = PrecisionRecallOptimizer(precision_weight=w)
        thresh, metrics = opt.optimize(preds, targets)
        metrics['precision_weight'] = w
        metrics['threshold'] = thresh
        results.append(metrics)
    
    return pd.DataFrame(results)
