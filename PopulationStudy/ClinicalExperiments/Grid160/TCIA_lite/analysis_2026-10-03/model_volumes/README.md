# DIR-Lab volumes for every main model (exported 6 Oct 2026)

All on the 2 mm, 160³ scoring cube. Array axes (Z, Y, X) = front-back, head-foot (+ toward the feet), left-right.
DVFs are vectors in mm (dx, dy, dz), pull convention: warped(x) = input(x + u(x)) should match the target.
Every prediction = plain + left-right mirror average. Seed 1 where two seeds exist (all-82 only has seed 2).

- `common/CXX_input_CT06_T50.mha`, `CXX_target_CT01_T00.mha`, `CXX_dvf_elastix_mm.mha` — shared by all models
- `<model>/CXX_warped_CT06.mha`, `CXX_dvf_mm.mha`, `checkpoint.txt`, `explanatory_figure.png` (cases 1, 4, 8)
- `tre_summary.csv` — DIR-Lab TRE300 per case (mirror)
- `comparison_after_difference.png` — target − warped for all models side by side

`07_small_push_ep40`: push labels live on the input grid, so the pull warp used here does not apply; its 14.9 mm and its
volumes are NOT valid (correct score 4.38 mm). The thin blue band at the bottom edge of "after" images is the warp's
border padding, not anatomy. Volumes are ~6 GB and are not in git; regenerate with `../export_model_volumes.py` then
`../explain_figs.py model_volumes`.
