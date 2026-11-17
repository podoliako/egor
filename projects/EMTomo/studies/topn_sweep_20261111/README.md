# Top-N sweep, 2026-11-11

20 new inversions / 14 new comparisons, 12 fixed updates each, control_fraction=0 to match historical controls. Four concurrent inversions with 12 workers each; numerical library threads limited to one. Frozen numerical runtime in snapshot/; manifest.json records hashes and all configurations. Existing runs and historical CSV are untouched.

- Section 2, one-sided stations: soft K50/top10, K50/top25, K100/top100; new hard K100/top100. First two reuse historical hard K50/top50.
- Section 3, two station clusters: same sweep.
- Section 4, real geometry: standard actual pick noise only, coverage power2, new hard/soft pairs K=topN=30,100,150,200. Assumed noise unchanged.
- Section 5, vertical fault: soft K128/top50,75,100,128. Reuse gfss28-h10, baseline soft gfss28-s21; temperature1, dmin2, coverage power0, original layered initial model. The baseline was chosen by smallest hard avg_abs_pct_dev at the existing L-curve checkpoint, not oracle iteration minimization.

## Monitoring

`queue_status.json`: queue state and finished case exit codes.
`cases/<case>/result.json`: running/completed/failed, PID, run ID.
`cases/<case>/worker.log`: numerical progress.
All runs use shared EMTomo/runs and prefix `topnsweep11-` for the viewer.

Queue is detached from terminal timeout. Do not launch it again or remove reservation files to retry. Failed cases are recorded, not automatically retried; independent remaining cases continue.

## Export

New output: `paper/exp_data/hard_soft_lcurve_avg_abs_pct_dev_topn_sweep.csv`.
Contains 52 rows: 38 historical comparisons +14 new comparisons. Pending/running/failed new pairs have blank metric cells. Separate hard/soft K and topN columns remove ambiguity when controls are reused. Historical cell values are retained, including historical hard_control_reuse_count; new rows count reuse within the expanded table. Reused controls are not independent replicates.

The queue invokes export.py after all workers terminate. To refresh statuses manually from repository root:

```sh
OPENBLAS_NUM_THREADS=1 venv/bin/python projects/EMTomo/studies/topn_sweep_20261111/export.py
```

The exporter caches only new residual extraction in this study, reuses historical caches, fixes training-only L-curve choices before truth-based evaluation, and retains V_k -> quality[k-1] timing. Original homogeneous-complex-anomaly plot exclusions remain unchanged. No new plot is generated automatically.

Validation: all20 configurations passed input/model/grid preparation, exact initial-array comparison and truth-model hash checks before launch. Export and existing L-curve unit tests:16 passed. Runtime completion requires12 complete markers and final_model.npy.
