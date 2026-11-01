"""Pure NumPy helpers for train-only origin fitting and held-out residual scores."""
import numpy as np


def profile_origin(predicted_all, observed_train, relative_sigma=0.01,
                   model_sigma=0.02) -> float:
    """Return origin offset (s) relative to the caller's earliest TRAIN reference.

    Predictions are absolute nonnegative propagation times at all global stations.
    Observations use that TRAIN reference, with NaN for every unavailable pick
    (including holdout). Only finite TRAIN positions enter the weighted fit.
    """
    predicted = np.asarray(predicted_all, dtype=float)
    observed = np.asarray(observed_train, dtype=float)
    if predicted.ndim != 1 or observed.shape != predicted.shape:
        raise ValueError('predictions and observations must be matching 1D arrays')
    if not np.all(np.isfinite(predicted)) or np.any(predicted < 0):
        raise ValueError('predictions must be finite nonnegative propagation seconds')
    if np.any(np.isinf(observed)):
        raise ValueError('missing observations must be NaN, not infinity')
    for name, value in (('relative_sigma', relative_sigma), ('model_sigma', model_sigma)):
        if (isinstance(value, (bool, np.bool_)) or np.ndim(value) != 0
                or not np.isfinite(value) or value < 0):
            raise ValueError(f'{name} must be a finite nonnegative scalar')
    if model_sigma == 0:
        raise ValueError('model_sigma must be positive')
    train = np.isfinite(observed)
    if np.count_nonzero(train) < 2:
        raise ValueError('at least two finite TRAIN observations are required')
    times = predicted[train]
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        variance = np.square(relative_sigma * times) + np.square(model_sigma)
    if not np.all(np.isfinite(variance)) or np.any(variance <= 0):
        raise ValueError('TRAIN noise variances must be finite and positive')
    # Scaling all precisions equally preserves the fit and avoids reciprocal overflow.
    precision = variance.min() / variance
    offset = float(np.average(observed[train] - times, weights=precision))
    if not np.isfinite(offset):
        raise ValueError('fitted origin must be finite')
    return offset


def summarize(residuals) -> dict:
    """Score nonempty finite 1D residuals (prediction minus observation), in seconds.

    Reject nonfinite scores rather than silently dropping failed predictions.
    """
    values = np.asarray(residuals, dtype=float)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError('residuals must be a nonempty finite 1D array')
    absolute = np.abs(values)
    return {
        'count': int(values.size),
        'rmse_s': float(np.sqrt(np.mean(np.square(values)))),
        'mae_s': float(np.mean(absolute)),
        'median_abs_s': float(np.median(absolute)),
        'p90_abs_s': float(np.percentile(absolute, 90)),
        'bias_s': float(np.mean(values)),
        'fraction_abs_gt_0p1_s': float(np.mean(absolute > 0.1)),
        'fraction_abs_gt_0p2_s': float(np.mean(absolute > 0.2)),
    }
