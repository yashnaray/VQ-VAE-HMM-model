import torch
import numpy as np
from calibration import (
    ThresholdCalibrator, 
    SignalNoiseController,
    EmpiricalStoppingCriteria,
    PrecisionRecallOptimizer,
    EvaluationLoop,
    calibrate_regime_thresholds,
    evaluate_with_tradeoffs
)
from VQ_VAE_HMM_fixed import VAE_HMM


def example_threshold_calibration():
    # simulate predictions and targets
    np.random.seed(42)
    predictions = np.random.randn(1000)
    targets = (predictions > 0.5).astype(int)
    
    # basic calibration
    calibrator = ThresholdCalibrator(min_precision=0.7, min_recall=0.5)
    result = calibrator.calibrate(predictions, targets)
    
    print("Threshold Calibration Results:")
    print(f"  Optimal threshold: {result.threshold:.4f}")
    print(f"  Precision: {result.precision:.4f}")
    print(f"  Recall: {result.recall:.4f}")
    print(f"  F1 Score: {result.f1_score:.4f}")
    print(f"  Signal ratio: {result.signal_ratio:.4f}")
    print(f"  Noise ratio: {result.noise_ratio:.4f}")
    
    # precision-recall curve
    precisions, recalls, thresholds = calibrator.get_precision_recall_curve()
    print(f"\nPrecision-Recall curve computed with {len(thresholds)} points")


def example_signal_noise_control():
    np.random.seed(42)
    predictions = np.random.randn(1000)
    targets = (predictions > 0).astype(int)
    
    # control signal ratio
    controller = SignalNoiseController(target_signal_ratio=0.3, tolerance=0.05)
    threshold = controller.find_threshold(predictions)
    
    print("\nSignal-Noise Control:")
    print(f"  Target signal ratio: 30%")
    print(f"  Calibrated threshold: {threshold:.4f}")
    
    # evaluate quality
    quality = controller.evaluate_signal_quality(predictions, targets, threshold)
    print(f"  Actual signal ratio: {quality['signal_ratio']:.4f}")
    print(f"  Signal quality: {quality['signal_quality']:.4f}")
    print(f"  Noise quality: {quality['noise_quality']:.4f}")
    print(f"  Overall quality: {quality['overall_quality']:.4f}")


def example_empirical_stopping():
    # simulate training loop
    stopping = EmpiricalStoppingCriteria(patience=10, min_delta=0.001, metric='f1_score')
    
    print("\nEmpirical Stopping Criteria:")
    for epoch in range(50):
        # simulate improving then plateauing metrics
        f1 = 0.5 + 0.3 * (1 - np.exp(-epoch / 10)) + np.random.normal(0, 0.01)
        metrics = {'f1_score': f1, 'precision': 0.7, 'recall': 0.6}
        
        if stopping.should_stop(metrics):
            print(f"  Stopped at epoch {epoch}")
            print(f"  Best F1: {stopping.best_value:.4f}")
            print(f"  Converged: {stopping.is_converged()}")
            break
    
    curve = stopping.get_convergence_curve()
    print(f"  Total iterations: {len(curve)}")


def example_precision_recall_tradeoff():
    np.random.seed(42)
    predictions = np.random.randn(1000)
    targets = (predictions > 0).astype(int)
    
    print("\nPrecision-Recall Tradeoff Analysis:")
    
    # try different precision weights
    for pw in [0.3, 0.5, 0.7]:
        optimizer = PrecisionRecallOptimizer(precision_weight=pw)
        threshold, metrics = optimizer.optimize_threshold(predictions, targets)
        
        print(f"\n  Precision weight: {pw:.1f}")
        print(f"    Threshold: {threshold:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall: {metrics['recall']:.4f}")
        print(f"    F1: {metrics['f1_score']:.4f}")


def example_evaluation_loop():
    # simulate model and data
    class DummyModel:
        def eval(self): pass
        def __call__(self, x):
            return torch.randn(len(x))
    
    class DummyLoader:
        def __iter__(self):
            for _ in range(5):
                x = torch.randn(32, 10)
                y = torch.randint(0, 2, (32,))
                yield x, y
    
    model = DummyModel()
    val_loader = DummyLoader()
    
    calibrator = ThresholdCalibrator(min_precision=0.6, min_recall=0.5)
    stopping = EmpiricalStoppingCriteria(patience=5, min_delta=0.01)
    
    loop = EvaluationLoop(calibrator, stopping)
    results = loop.run(model, val_loader, max_iterations=20)
    
    print("\nEvaluation Loop Results:")
    print(f"  Iterations: {results['iterations']}")
    print(f"  Converged: {results['converged']}")
    print(f"  Best F1: {results['best_result'].f1_score:.4f}")
    print(f"  Best threshold: {results['best_result'].threshold:.4f}")


def example_regime_calibration():
    # simulate VAE-HMM and data
    vae_hmm = VAE_HMM(input_dim=5, hidden_dim=32, K=3, hidden_dim2=16, u_dim=4)
    data = torch.randn(1, 5, 100)
    true_regimes = np.random.randint(0, 3, 100)
    
    print("\nRegime Threshold Calibration:")
    thresholds = calibrate_regime_thresholds(vae_hmm, data, true_regimes, K=3)
    
    for regime, thresh in thresholds.items():
        print(f"  Regime {regime}: threshold = {thresh:.4f}")


def example_tradeoff_analysis():
    np.random.seed(42)
    predictions = np.random.randn(1000)
    targets = (predictions > 0).astype(int)
    
    print("\nPrecision-Recall Tradeoff Table:")
    df = evaluate_with_tradeoffs(predictions, targets)
    print(df[['precision_weight', 'threshold', 'precision', 'recall', 'f1_score']].to_string(index=False))


if __name__ == "__main__":
    print("=" * 60)
    print("THRESHOLD CALIBRATION EXAMPLES")
    print("=" * 60)
    
    example_threshold_calibration()
    example_signal_noise_control()
    example_empirical_stopping()
    example_precision_recall_tradeoff()
    example_evaluation_loop()
    example_regime_calibration()
    example_tradeoff_analysis()
