# Mean signed anomaly amplitude versus pick noise

For each sphere: sum[(Vp_est-background)*(Vp_true-background)] / sum[(Vp_true-background)^2]; arithmetic mean of six signed coefficients. 1=correct amplitude, 0=no template projection, negative=opposite contrast; not clipped and not a complete reconstruction-error metric.

The figure shows percentages; 100% is the target, not a guaranteed optimum for other errors. Known flat4 background is subtracted, not fitted. Fast/slow signs are normalized by each true template. Equal weight per sphere; no absolute values or clipping. Noise artifacts, background bias, shape and location errors can affect the projection, and averaging can hide differences between spheres. A larger value is not necessarily a better reconstruction; use alongside RMSE.

Same 18 saved final models, six paired noise levels, fixed inversion settings. No solves. Rebuild: `python -B paper/make_geysers_noise_quality.py --amplitude`.
