# Fault minimum-distance validation

Four new runs: hard/soft at dmin4 and dmin7. Each uses K128, top50, T1, coverage power0, 12 fixed updates, control_fraction0, 12 workers, the original layered start and gfsoftscreen28_26st_noise02_3776ev observations. All noise settings are unchanged. Only candidate mode, minimum distance and run name differ from the completed top50 baseline config.

The entire numerical runtime snapshot is copied from topn_sweep_20261111 and checksum-verified, not regenerated from current live code. The baseline dmin2 run IDs are recorded in manifest.json. Its hard has top40, which does not change MAP selection at fixed K/dmin.

Two simultaneous runs (24 workers total), leaving 24 workers for the existing K200 pair. dmin4 runs first, followed by dmin7. Results go to the shared EMTomo/runs with prefix faultdmin12-. Never retry by deleting reservation files; inspect failures first.

Monitoring: queue_status.json, cases/<case>/result.json, cases/<case>/worker.log. Queue PID is in launch.json.

Preflight checked all four experiments, exact initial arrays, truth hashes, unchanged source checksums, and hard/soft config differences. Successful worker completion requires 12 complete iteration markers and final_model.npy.

After all four workers finish, the existing exporter writes a separate paper/exp_data/hard_soft_lcurve_avg_abs_pct_dev_fault_dmin.csv (38 historical pairs + two new pairs). It does not overwrite the active top-N CSV. Baseline top50/dmin2 remains in the top-N CSV. A combined distance-sweep analysis should use that baseline and the two new pairs, compare equal-update trajectories as well as L-curve-selected errors, and inspect hypothesis effective count/spread before interpreting changes in gain. No plot is generated automatically.
