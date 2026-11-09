# Hypocentre-error 90th percentile versus pick noise

Full catalogue (3776 events), Euclidean 3D error in metres, NumPy linear percentile. For hard and soft, use the maximum-weight saved hypothesis (MAP); no averaging of soft locations. The saved coordinates are continuous fine-grid indices: (positions+0.5)*125 m. Last saved E-step, iter_11, BEFORE the final velocity update; no new relocation or solve. Original 1%+10 ms baseline and five additional noise levels; all three configurations fixed. One paired seed, no uncertainty intervals. Raw per-event errors and MAP coordinates are in the event CSV.

Rebuild: `python -B paper/paper_scripts/make_geysers_noise_hypocentre_p90.py`.
