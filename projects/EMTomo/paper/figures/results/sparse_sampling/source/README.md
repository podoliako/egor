# Dense versus sparse candidate sampling: one saved event

Rebuild from the EMTomo root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python -B paper/figures/results/sparse_sampling/source/make_sparse_sampling.py
```

`../png/` contains PNG previews `dense_vs_sparse_event.png` and the identical `dense_vs_sparse_event_final.png`. This directory contains matching PDFs, the script, plotted arrays, numerical likelihood field, and provenance JSONs. No article files or inversion outputs are modified.

The figure has three columns: dense candidates, sparse candidates, then full-set X and depth CDFs stacked vertically. There is no overall title, subtitle or footer. Panel headings and colorbar/axis labels identify the data.

## Additional user-selected event EVT_0905

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 python -B paper/figures/results/sparse_sampling/source/make_sparse_sampling.py --event-id EVT_0905 --xlim 0 7 --full-depth --filled-truth --distance-style
```

This uses saved **event index 8**, joined by ID in the same pilot subset of the one-sided 4.2 experiment. Outputs are `../png/dense_vs_sparse_EVT_0905{,_final,_all_candidates_final}.png` (identical renderings), matching PDFs and `dense_vs_sparse_EVT_0905.json`, `plotted_data_EVT_0905.npz`, `likelihood_field_EVT_0905.{npz,json}` here. The default EVT_0401 image/cache/data files are not overwritten.

Both maps show **X = 0–7 km and full depth 0–12 km**. The true hypocentre is a **filled burgundy star (`#800020`), without outline**, projected onto XZ. The same dense-weighted-mean rule selects the likelihood plane at **Y = 2.4375 km**. With `--distance-style`, **all 10000 dense and all 50 sparse candidates** are projected onto XZ, without any Y restriction. For diagnostic comparison only, the saved slab masks still identify the earlier 1001/3 candidates; those masks do not control display. The right CDFs also use all 10000/50 saved candidates. There is no additional candidate selection to make this event resemble EVT_0401.

For this new event, the initial-model station fields were recomputed without rerunning localization/inversion; both sets of saved candidate probabilities were reproduced numerically. The event-specific likelihood cache is fingerprinted and reused on subsequent runs. The current `--distance-style` for EVT_0905 maps **likelihood 0 to white and likelihood 1 to dark blue**. Candidates are **opaque grayscale points** with `gray = 0.8 * abs(Y - plane_Y) / global_max_distance`, using the same distance normalization across both full candidate sets. Near-plane points are black; far points are light gray. Far points are drawn first so near-plane black points remain visible. This encodes geometric distance, **not candidate probability or weight**, and is not an alpha/transparency effect. There is **one vertical likelihood colorbar to the left**, and no proximity/weight scale. Map widths are calculated from their metric aspect ratio; fixed small physical gaps remove unused space between map panels, with the stacked CDFs immediately to their right. Arrays store full candidate distances and gray values, all-true display masks, and separate diagnostic slab masks. Likelihood values, weights, plane, burgundy star and CDFs are unchanged. Without this flag the earlier grayscale background and weight-colored points remain available; the details below describe that default EVT_0401 styling unless stated otherwise.

## Which experiment and event?

- Experiment: `onesided_nx6_24x12x12_18st_10ev_noise01_pick005_seed42`.
- Event: **EVT_0401**, `event_0`, the first event in the saved input order. It was not selected by looking for the closest matching distributions.
- Common checkpoint: `iter_0/model.npy`, before any velocity update; uniform 5000 m/s, coarse 1-km cells, q=8, 125-m computational grid.
- Common data: the same 18 observed picks, station coordinates and timing-error model.
- Dense run: `onesided-pilot-soft10000`, 10,000 tested/retained candidates, `min_dist=1`.
- Sparse run: `onesided-pilot-soft50-dmin7`, 50 tested/retained candidates, `min_dist=7`.

Full run IDs, input/checkpoint hashes and weight-file hashes are recorded in `dense_vs_sparse_event.json` and `likelihood_field.json`. Candidate coordinates and normalized weights are read from the **original saved weights.npz files**, not generated for the illustration.

The separation is the minimum **Chebyshev distance between seed cells**, before continuous refinement: 125 m for min_dist=1 and 875 m for min_dist=7. Final refined positions need not preserve that separation. The maps show only candidates in a shared **250 m thick slab** (two fine cells), centred on **Y = 6062.5 m** with half-width **125 m**. The plane is the nearest fine-grid cell centre to the dense set's full weighted mean Y, not chosen for visual agreement. Inclusive selection uses a 1e-6 m tolerance for floating-point boundary noise. This retains **7354 / 10000 dense** and **50 / 50 sparse** candidates, without weight filtering, jitter, truth-error filtering or random subsampling. All original coordinates/weights and the Boolean slab masks are exported.

Candidate **small filled point markers** use one shared light-blue-to-dark-blue scale of `w / max(w)` within each **full run**. The displayed common color limits span the observed selected relative weights (approximately 0.715–1), allowing the small weight differences to be seen. These are not raw probabilities comparable between a 10,000- and a 50-hypothesis set; no saved weights are altered. Candidates use the small `.` marker, opaque weight-colored fill and no outline, with sizes 10 pt² for dense and 35 pt² for sparse. Candidate point values need not match the background underneath: saved weights refer to each candidate's own 3-D position, while the background is at fixed Y, and their relative-likelihood reference maxima differ.

Both maps include an **unfilled orange star** at the actual synthetic hypocentre, X = 15023.063 m, Y = 5595.818 m, Z = 7023.789 m. Its XZ projection is drawn even though its Y lies **466.682 m** away from the plane, outside the candidate slab; the legend explicitly calls it an XZ projection. Truth is used only for this annotation and to ensure the star fits within plot bounds, never to choose candidates, weights, likelihood, event or section plane. Its original 3-D coordinates are exported in NPZ/JSON.

## The background is the same numerical field

Neither pilot run saved a full likelihood volume or station fields. The script therefore reconstructs the station fields once using production `eikonal.py`, the hash-identical initial model and station coordinates. It then evaluates the same marginalized-origin Gaussian likelihood for this one event. There is **no new localization, optimization, ray tracing or velocity inversion**.

To check reconstruction, likelihoods are sampled at all 10,000 and 50 saved continuous candidate locations, and their normalized probabilities must reproduce the saved weights within the specified numerical tolerance. The cached field is reused only when its input/code fingerprints and cache SHA-256 still match.

The background is now the **actual fixed-Y likelihood slice** at Y = 6062.5 m, not the previous maximum-over-Y profile or a marginalized posterior:

`exp(-(cost[:, plane_y_index, :] - cost[:, plane_y_index, :].min()))`.

This is drawn using opaque **`Greys_r`**, with limits [0, 1]: the maximum is **white** and zero is **black**. There is no arbitrary likelihood rescaling or display-alpha attenuation. Saved 3-D positions within the slab are projected orthogonally onto X–depth, retaining their original weights.

The right-column panels are exact weighted empirical CDFs of the **full original candidate sets**, not conditioned on the map slab and not smoothed KDEs. Their titles explicitly say full-set. Thus the original Wasserstein distances and full-set statistics below are unchanged. Both maps additionally show a **green diamond** for the XZ projection of `sum(w_i * XYZ_i)` using the full saved, normalized weights, and a **red diamond** for the saved candidate with maximum weight. Ties use the first saved maximum (`numpy.argmax`). Marker positions are never jittered. The green diamond is slightly larger so it remains identifiable if the red MAP diamond nearly overlaps it. The NPZ exports both 3-D positions and the MAP candidate index for each run; legends distinguish them from the true-hypocentre star.

## What this example demonstrates, and what it does not

With 200 times fewer retained candidates, the sparse set covers a similar X–depth ridge and gives close weighted X/depth marginals in this example:

- Weighted-mean 3-D location separation: **127 m**.
- One-dimensional Wasserstein distance: **132 m in X**, **172 m in depth**.
- Weighted X standard deviation: **1.387 / 1.444 km**, dense / sparse.
- Weighted depth standard deviation: **1.804 / 1.942 km**.

This is not full 3-D distributional equivalence. In particular, Y standard deviations are **0.105 / 0.013 km**, so the sparse set is markedly narrower across the ridge. Both approximations assign equal prior mass to their retained hypotheses and use likelihood weights, not quadrature-volume correction. Candidate selection is deterministic and refined, not iid posterior sampling. Similar projected CDFs do not establish calibrated uncertainty, identical velocity updates, a speedup of 200 times, or general equivalence across events.

The first event's original location is not used to choose the example or calculate the likelihood field. It is used to draw the truth star and expand plot bounds to keep the star visible. The figure compares candidate representations in a fixed velocity model, not location accuracy against truth.
