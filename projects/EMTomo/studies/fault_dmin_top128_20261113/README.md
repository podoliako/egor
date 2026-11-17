# Fault dmin sweep at K128/top128

Four new inversions: hard and soft for minimum distance4 and7. Uses the exact frozen numerical snapshot from topn_sweep_20261111, original gfsoftscreen28_26st_noise02_3776ev data, layered initial model, T1, coverage power0, 12 fixed updates, no heldout split, unchanged noise and solver settings. Baseline soft is the completed K128/top128/dmin2 run recorded in manifest.json.

All four runs execute concurrently with12 workers each and one numerical-library thread per process (48 workers total). Previous two studies were complete before launch. Shared viewer run prefix: faultdmin13-.

Monitoring: queue_status.json, cases/<case>/result.json, cases/<case>/worker.log. Detached queue PID: launch.json. No automatic retries; preserve reservation files.

Validation: snapshot hashes, exact initial arrays, truth-model hash, event counts and matching hard/soft configurations checked for all four cases. Completion requires12 complete iteration markers and final_model.npy.

On completion exporter writes separate paper/exp_data/hard_soft_lcurve_avg_abs_pct_dev_fault_dmin_top128.csv with38 historical rows +2 new pairs. The existing combined CSV and figures are not overwritten automatically. These must be merged with both the completed topN sweep and top50 distance sweep when plotting the expanded comparison.
