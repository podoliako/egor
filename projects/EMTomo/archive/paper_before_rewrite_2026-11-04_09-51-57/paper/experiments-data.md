# Experiment sources and metric reproduction

`experiments.tex` is included by `main.tex`. The paper's synthetic-result tables use the saved comparisons below; none of the underlying inversions or oracle fits was rerun while drafting the section.

## Primary comparisons

All three directories exist under `/home/egor/tomodd_benchmark/runs/`:

| Directory | Selected checkpoints |
| --- | --- |
| `comparison_nx4_nx6_nx8_locnoise2km_seed42` | EMTomo after 12 updates, tomoDD iteration 3 |
| `comparison_flat4_synthetic_em8_tomodd3_zero_noise` | EMTomo after 8 updates, tomoDD iteration 3 |
| `comparison_flat4_geysers_em12_tomodd3_zero_noise` | EMTomo after 12 updates, tomoDD iteration 3 |

Use each directory's `metrics.json` and `README.md`; the two layered comparisons additionally contain `provenance.json` with source paths and artifact hashes. TomoDD run paths and selected velocity/relocation history files are identified there. In the synthetic zero-start case the source run continued to 10 iterations: its final state is **not** the iteration-3 result used in the paper.

- Checkerboard velocity errors in the original JSON are in **km/s**; paper table values multiply by 1000. They use 432,000 common 2-km samples, not native parameter-space errors.
- Layered velocity errors use all 501,760 physical 125-m truth centers, in **m/s**. The six sphere supports contain 15,314 samples altogether.
- Layered location errors in the primary JSON are in **km**; paper values multiply by 1000.
- `zero_noise` distinguishes zero/noisy **initial hypocenters**. It does not mean noiseless picks. Pick-noise terms are independent and add in quadrature.
- Initial model fields differ under constant-cell versus native trilinear interpolation even when their interior parameter values are identical.
- Source coordinates already include the Geysers elevation transformation; do not apply another correction. The shared station/catalog vertical datum remains an assumption.

## Added metrics

`evaluate_metrics.py` reads original artifacts and generates `supplemental_metrics.json`. It verifies available source/checkpoint hashes, geometry, masks, checkpoint selection, and agreement with primary comparison RMSE and hard-location scores. It computes:

- Signed sphere-template amplitudes against the **known physical layered background** for primary reconstructions, native initial fields, and both representation oracles.
- Energy score, MAP RMS/median/P95, weighted-mean RMS error, and RMS spread for saved hard/soft event hypotheses.
- Common-grid velocity RMSE for all three additional Geysers soft runs (K=5/10/15), with all configuration differences recorded. These are a candidate-count sensitivity scan, not an isolated soft-weight ablation.

Saved continuous location indices convert to metres as `(positions + 0.5) * fine_cell_size_m`; `weight_indices` are not refined event coordinates. Catalog order is checked against source event rows and hashed.

From `projects/EMTomo/paper`, using the existing Python environment with NumPy/SciPy and access to the original artifact paths:

```sh
python3 -B -m unittest -v test_evaluate_metrics.py
python3 -B evaluate_metrics.py --check
```

There are 17 unit tests. `--check` recomputes without writing; evaluation is limited to 120 seconds. Without `--check`, the script creates its report exclusively and refuses to overwrite an existing file. It does not import or call a forward/inverse solver. Native interpolation/history/mask helpers are loaded from the recorded benchmark analysis source; the source hash and helper names are saved in the report.

## Version and checkpoint limitations

- The checkerboard EMTomo runs are labeled **v1.1**, not the **v1.4** framework described in Method. Runtime code hashes were not recorded. The last archived pre-run revision, `98718ef2a744f290799cb72fe2d7068c982d9789`, uses station cell-center seeding and unweighted pairwise residuals. This source/run association is supported by dates and metadata, not conclusively established by runtime hashes. Verify or rerun before claiming validation of the current implementation.
- The layered runs record `skfmm_order2_exact_station_homogeneous_sphere`. The saved Geysers runtime snapshot precedes the working tree's later `...sphere_v2` handling of velocities inside the seeding sphere; the original runs were not recomputed using that change.
- The legacy artificial-geometry hard run does not record ingestion-time event/arrival hashes. Its current comparison hashes, source/model identity, station coordinates, and full catalog count are checked, but original ingestion cannot be retrospectively hash-proven.
- EMTomo location scores are from the E-step before the last reported velocity update (synthetic `iter_7`, Geysers `iter_11`). They are **not** final relocations in the selected updated velocity. A final location-only pass remains needed for synchronized checkpoints.
- No matched artificial-geometry soft run was found among the 468 run metadata files inspected at drafting time.
- Checkerboard EMTomo event weights exist, but their location metrics are not included in the primary comparison and have not been added here.

## Figure copies

The following existing PNGs were copied without changing their image content:

| Paper copy | Primary comparison source |
| --- | --- |
| `figures/checkerboard_y25km.png` | checkerboard `comparison_y25km.png` |
| `figures/flat4_synthetic_z1p4375km.png` | synthetic layered `slice_z1.5km.png` |
| `figures/flat4_geysers_z1p4375km.png` | Geysers layered `slice_z1.5km.png` |

The layered images' requested 1.5-km plane is actually sampled at the nearest truth center, **1.4375 km**, as stated in their captions. Figure checkerboard velocity units are km/s; layered velocity units are m/s.

## Metric references

- PyKonal: White et al. (2020), DOI `10.1785/0220190318` (bibliographic details checked against Crossref).
- Probabilistic hypocenter summaries: Lomax et al. (2000) and HypoSVI, Smith et al. (2022). HypoSVI's methods explicitly distinguish the dominant-mode location and posterior uncertainty; this is not a claim that it uses our energy score.
- Energy score: Gneiting & Raftery (2007), DOI `10.1198/016214506000001437`. The score here is evaluated exactly for each saved weighted finite distribution, rather than estimated with a finite-sample unbiased ensemble correction.
