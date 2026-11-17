# Latest real-pick comparison for §4.7

This evaluates fixed final 500-m models from
`runs/geysers_real_refine500_fmm125_20261110_110138`, plus the original
1-km-cell linear 3.5–6.0 km/s initial model. No velocity inversion is rerun.
The final models were selected by the original internal-control stopping,
not by the synthetic L-curve rule or by external-holdout performance.

## Prediction protocol

- 3471 events and 31 BG stations; 1–20 April 2011.
- Preserve the exact original inversion mask: **38676 training picks**.
- Exclude both **4297 internal control picks** and **4775 external holdout picks**
  from event relocation and origin fitting.
- Use the queue's hash-verified frozen numerical Python, common 125-m scikit-fmm
  travel-time fields, 50 refined candidate seeds separated by 7 grid steps,
  and MAP location for **all** velocity models.
- Likelihood standard deviations: relative 0.01, absolute 0.05 s, model 0.02 s.
  Fit each origin with training-only weights `1 / noise.sigmas(T)**2`.
- Save predictions for every station before opening the external holdout.
- Score all 4775 external picks from 3186 events; no outlier removal or refit.
- Recomputed internal-control RMSE must match the original stopping value to
  absolute tolerance 1e-9 s for both final models.

This isolates predictive performance of the recovered **velocity models** under
one common relocation rule. It is not soft posterior-predictive scoring.
Catalog coordinates are never read during fitting or held-out scoring.

`summary.csv` contains pooled pick metrics; `summary.json` adds source hashes,
protocol, model provenance, gains and paired event-bootstrap intervals.
`holdout_predictions.csv` contains every scored residual (prediction minus pick).
`*_predictions.npz` also contains training-only fitted event coordinates.
`*_fit.json` binds each prediction artifact to its velocity model, training mask,
noise, runtime and evaluator hashes.

The paired bootstrap resamples scored events with all their held-out picks and
keeps hard/soft pairing (2000 replicates, seed 42). Its interval is conditional on
this split and these fitted models. The external split was already inspected for
older models; it is **not** a pristine tuning-independent test. Withheld picks of
training events are also not a test on wholly unseen events.

## Related descriptive comparisons

- `../geysers_real_structure_comparison.{json,csv}`: volume-sampled Vp differences
  from Guo–Thurber 2021 where published coverage is finite and interpolated scaled
  resolvability is strictly greater than 0.4. These are **model differences**, not
  errors against truth. The 125-m sampling is checked against 250-m sampling.
- `../geysers_real_location_shifts_latest.{json,csv}`: fitted MAP shifts relative
  to the catalog, initial-model fits, and between hard/soft. Catalog coordinates
  are read only after all fits; distances are not accuracy metrics. Compatibility
  of catalog and station vertical datums has not been independently established.
- `../../figures/real_picks/geysers_real_location_shifts_latest.{png,pdf}`:
  descriptive XY/XZ/YZ projections and relocation vectors.

## Reproduction

Run from `projects/EMTomo` with the repository virtualenv Python. Set BLAS, OMP
and Numba thread counts to 1 to avoid oversubscription when using 40 processes.

```sh
python -B studies/geysers_real_picks/evaluate_latest_holdout.py --fit initial --workers 40
python -B studies/geysers_real_picks/evaluate_latest_holdout.py --fit hard --workers 40
python -B studies/geysers_real_picks/evaluate_latest_holdout.py --fit soft --workers 40
python -B studies/geysers_real_picks/evaluate_latest_holdout.py --score-only
python -B paper/paper_scripts/make_geysers_real_structure_metrics.py
python -B paper/paper_scripts/make_geysers_real_location_shifts.py
```

Existing fits are reused only when the evaluator and all bound inputs match.
Original inversion runs and their stopping checkpoints are never modified.
