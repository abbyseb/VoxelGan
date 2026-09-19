# TRE Viewer (`tools/tre_viewer`)

Interactive TRE / DVF / DRR diagnosis for **DIR EXPERIMENTS** runs.

**Branch:** `TRE-VIZ`  
**Default display frame (later phases):** pack-native mm  
**Stack:** dedicated `.venv` here (not LEARN-GUI).

## Start here

The target platform is **Linux** with a desktop display, **Python 3.10+**,
DIR-Lab landmark packs, and prepared experiment runs.
The raw CTs, landmarks, fields and checkpoints are not included in this code checkout.
Set `DIRLAB_ROOT` to the folder containing `Case1Pack`, `Case2Pack`, etc.
The coordinate adapter is `DIR EXPERIMENTS/scripts/eval_a1_tre.py`.

### Setup

On a minimal Ubuntu/Debian installation, install the Qt/OpenGL/font libraries first:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv libegl1 libgl1-mesa-dri libopengl0 \
  libglib2.0-0 libfontconfig1 fonts-dejavu-core libdbus-1-3 libxkbcommon-x11-0 \
  libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
  libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 libxcb-shape0
# For headless tests, also install: sudo apt-get install -y xvfb xauth
```

```bash
cd "DIR EXPERIMENTS/tools/tre_viewer"
bash setup_venv.sh
# If python3 is older than 3.10:
# PYTHON=python3.11 bash setup_venv.sh
```

### Commands

```bash
cd "DIR EXPERIMENTS/tools"
./tre_viewer/.venv/bin/python -m tre_viewer doctor
./tre_viewer/.venv/bin/python -m tre_viewer list-runs
./tre_viewer/.venv/bin/python -m tre_viewer verify --arm A1 --case 1
./tre_viewer/.venv/bin/python -m tre_viewer per-landmark --arm A1 --case 1 --field elastix_mha
```

`list-runs`, `verify`, `per-landmark` and `view` accept `--runs-dir /path/to/runs`.
Run `python -m tre_viewer --help` for commands or add `--debug` **before** the
command to show a full traceback. Help and run discovery do not require patient data.

`verify` recalculates TRE and compares the selected **75 or 300** landmark set
against its matching reference. It returns 0 only when all requested comparisons
pass, 1 for absent reference files/runs or skipped comparisons, and 2 for failed
checks (including missing fields/pairs in an existing summary). An absent
reference file is reported as **SKIP**, never as a successful verification.

The experiment's primary arm KPI is **75-point Sampled4D TRE**. The 300-point
option is for pack QA/debugging and must not be reported as an arm score.

### Success criteria

- `list-runs` prints A1 cases; empty A2/A3 arms produce no rows (no crash)
- `verify --case 1`: identity mean matches A0; elastix/voxelmap means match `tre_summary.json` to `1e-6`
- `per-landmark` writes `runs/DIR_Cxx/tre/per_landmark_75_T00_T50_<field>.npz`

## Phase 1 (volume TRE viewer)

napari GUI: pack-mm `GTVol` CT, axial/coronal/sagittal, truth/pred landmarks,
TRE colour + rings + error vectors, summary dock, worst-landmark jump (`W`).

```bash
cd "DIR EXPERIMENTS/tools"
./tre_viewer/.venv/bin/python -m tre_viewer view --arm A3 --case 1
# or point at any runs folder:
./tre_viewer/.venv/bin/python -m tre_viewer view \
  --runs-dir "../arms/A3_synth_conditioned/runs" --case 5 --field voxelmap
```

**In the Controls dock**
- **Runs folder** + **Browse…** / **Scan folder** — pick an arm root, `…/runs`, or parent of `DIR_Cxx`
- **Case / patient** dropdown — switch C01–C10 without restarting; shows TRE75 from `tre_summary.json`
- Select the field, phase pair and 75/300 landmarks, then click **Apply data / refresh**.
- Slice plane, displacement display, checkboxes and arrow settings update immediately.
- The case label reports the **loaded** field, pair and landmark set.
- A failed case or landmark load preserves the previous display and shows the error.

Keys: `W` jump worst TRE · `Shift+W` worst identity · `A`/`C`/`S` orient · **Apply data / refresh** in Controls.

### Success criteria

- Smoke: 75 truth points + vectors; C01 identity ≈ 3.913 mm
- UI mean TRE matches `verify` / `tre_summary.json` for the selected field
- Jump-to-worst centres the slice on that landmark

## Phase 2 (DVF + warp + best.pt cache)

Overlays (Controls dock / layer list):
- `DVF |u| mm` / component (SI/AP/LR) — resampled to pack for display
- `DVF arrows` — decimated quiver on current slice
- `warped source`, `target − warped`, `target − source (identity)`
- **Space** blinks target ↔ warped
- Arrows follow the current slice and use pack-voxel lengths; R3 fields are reoriented before display.
- Failed overlay/projection loads clear stale images. Without a lung mask, MAE is labeled **whole volume**.
- Reverse (`T50_T00`) landmark TRE remains available. Reverse **image** overlays require an inverse DVF and are disabled until one is supplied; the forward field cannot be reused for that warp.
- Missing VoxelMap cache → infer once from `best.pt` (LEARN torch if needed) and write `tre/voxelmap_dvf_phase01_mean.npy`

For inference, set `VOXELMAP_CLINICAL_ROOT` to your VoxelMap_Clinical checkout.
If torch/model dependencies live in another environment, set
`TRE_VIEWER_INFERENCE_PYTHON=/path/to/that/venv/bin/python`.

```bash
./tre_viewer/.venv/bin/python -m tre_viewer smoke --case 1
# expect warp_improves: true, mae_warped_lung ≪ mae_ident_lung
```


## Respiratory Phase Performance

Open the **Phase Performance** tab (or **Window → TRE Panels → Show Phase Performance**).
Choose **Direct synthesizer** or **Downstream VoxelMap**, then click **Evaluate runs**.
Evaluation runs in the background across the loaded runs folder; **Cancel** keeps completed cells.
Run names such as `DIR_C01_muhist` and `DIR_C01_g160ft` appear separately.

- **Phase curves:** selected run's 75-point mean and p95 TRE, identity mean, and improvement
  (`identity − model`, positive is better). Gaps indicate unavailable scores.
- **Patient × phase:** mean, p95, or improvement heatmap. Negative improvement means worse
  than identity. The detail text identifies the selected run's highest-error scored phase.
- **Landmark trajectory:** observed and predicted signed displacement from T50 in mm,
  on official xyz axes (approximately LR/AP/SI). Landmark IDs are zero-based, matching the TRE panel.
- Click a heatmap cell or plotted phase to load that run/phase's real CT and available
  landmark overlays. The landmark selector and TRE worst-landmark jump stay linked.
- **Check synth CT** loads the saved synthesized CT and a signed `synth − real` HU layer.
  It also reports whole-volume HU MAE, including background. This check works without
  annotations and is an image-similarity diagnostic, not a substitute for TRE.
- **Export JSON** saves scores, status/reasons, coverage and landmark trajectories.
  Missing scores are JSON `null`, never zero. **Return to KPI view** restores the usual
  T00→T50 view. Historical image/DVF/DRR overlays are disabled during phase inspection
  so they cannot reuse a field belonging to a different phase or mapping direction.

All new phase scores map **T50 → target phase**, using **75 corresponding Sampled4D
landmarks only**. T50 is an explicitly marked zero-motion reference, excluded from rankings.
Only phases with both reference and target annotations are scored; coverage is discovered
from the pack, not assumed to include all ten phases. Rankings describe the scored phases
of the selected run; they do not establish clinical adequacy or statistically significant differences.

Direct synth evaluation requires `synth_meta.json` and
`<scan_id>/train/_synth_dvf_infer_XX.npy`, as written by `prepare_a3_dir_case.py`.
It uses the raw pull field, **not** the negated `DVF_sub_XX.mha` training label.
The evaluator reproduces the preparation script's resize-then-warp sampling and solves
`q + u(q) = p` for each reference landmark, with a maximum residual of **0.01 mm**.
If any of the 75 solutions fails to converge or leaves the image grid, the phase is marked
invalid and excluded from scoring. Convergence does not prove the field is globally invertible.
The legacy T00→T50 KPI still uses its existing first-order inverse approximation; its values
are not interchangeable with the new phase curves.

Downstream VoxelMap evaluation reads each phase's own
`tre/voxelmap_dvf_phaseXX_mean.npy` (`XX=01…10`, zyx grid, xyz components).
A phase-01 cache is never substituted for another phase. The GUI does not run GPU inference.
Create missing caches with the batch command in the environment containing PyTorch and
VoxelMap dependencies:

```bash
cd "DIR EXPERIMENTS/tools"
export DIRLAB_ROOT=/path/to/dirlab_packs
export VOXELMAP_CLINICAL_ROOT=/path/to/VoxelMap_Clinical
python -m tre_viewer phase-performance \
  --runs-dir /path/to/A3/runs --stage synth --output /tmp/synth_phases.json
python -m tre_viewer phase-performance \
  --runs-dir /path/to/A3/runs --stage voxelmap --infer-voxelmap \
  --real-runs-dir /path/to/A1/runs --device cuda --stride 10 \
  --output /tmp/voxelmap_phases.json
```

Use `--case 1` to restrict evaluation. Inference is only attempted for annotated,
non-reference phases. Existing caches are reused. Newly inferred caches include a JSON
sidecar recording the phase, reference, frame, checkpoint, real data directory, stride and
projection count. A3 inference uses **real A1** projections, following `eval_a3_tre.py`;
it never defaults to the synthetic training projections. Without `--real-runs-dir`, it
uses the existing TRE summary's evaluation directory or the canonical A1 runs folder.
The command exits 0 when at least one non-reference cell is scored and no cell is invalid;
missing annotations/fields remain explicit in the export. It exits 1 if nothing is scored
or any cell is invalid. Setup/output errors exit 2.

## Window / panels

Multipane layout (napari docks, overridden for a usable default):
- **Center:** CT + TRE overlays (primary canvas)
- **Right tabs:** **TRE** (summary / worst list) ↔ **Controls** (cases + overlays) ↔ **Phase Performance** (respiratory phase graphs) — tabified so they don’t fight for height; Controls scrolls if needed
- **Bottom:** clickable **DRR** / **RTK** launchers → separate maximized full-page windows (Shift+D / Shift+T)
- Soft size hints only (no hard min widths that clip). Drag edges / float titles as usual.
- Close (X) hides a panel; **P** or Window→TRE Panels restores.

| Key / menu | Action |
|---|---|
| Drag dock edge / title | Resize / float / move panes |
| Panel title-bar `×` / `–` | Hide the panel (never destroys it) |
| `P` or **Window → TRE Panels → Show all** | Reopen closed panels |
| **Window → TRE Panels → Show DRR/TRE/Controls** | Reopen one panel |
| **Window → TRE Panels → Hide …** | Hide one panel |
| `F11` | Toggle fullscreen ↔ maximized |
| **Window → TRE Panels → Open DRR full page** (`Shift+D`) | Maximized DRR scrub window |
| **Window → TRE Panels → Open RTK landmarks full page** (`Shift+T`) | Maximized RTK overlay window |
| **Window → Export PNG → CT canvas** (`Ctrl+E`) | Save main CT view as PNG |
| **Window → Export PNG → Full main window** (`Ctrl+Shift+E`) | Save whole main window as PNG |
| DRR / RTK page **Export PNG…** (`Ctrl+S`) | Save that page’s figure as PNG |

PNG defaults go under ``<run>/tre/exports/``.

## Regression tests

From the repository root:

```bash
"DIR EXPERIMENTS/tools/tre_viewer/.venv/bin/python" -m pip install -r "DIR EXPERIMENTS/tools/tre_viewer/requirements-dev.txt"
"DIR EXPERIMENTS/tools/tre_viewer/.venv/bin/python" -m pytest "DIR EXPERIMENTS/tools/tre_viewer/tests" -q
```

Tests use synthetic volumes and landmarks. On a headless Linux machine, install
the libraries listed above plus Xvfb/xauth, then use Mesa software rendering:

```bash
QT_QPA_PLATFORM=xcb QT_API=pyqt6 LIBGL_ALWAYS_SOFTWARE=1 \
  xvfb-run -a -s '-screen 0 1280x1024x24' \
  "DIR EXPERIMENTS/tools/tre_viewer/.venv/bin/python" -m pytest "DIR EXPERIMENTS/tools/tre_viewer/tests" -q
```

Real-case `verify` and `smoke` still need your prepared data.
Landmark caches are rebuilt when their DVF metadata, landmarks,
coordinate adapter or frame changes; verification always bypasses cached results.

Starts **maximized** (not tiny). Bottom dock has clickable **DRR** / **RTK** buttons for the same pages. If the launcher vanished, press **`P`**.
Panels are rebuilt from scratch if their dock was destroyed, so `P` always works:

```bash
QT_QPA_PLATFORM=offscreen ./tre_viewer/.venv/bin/python -m tre_viewer panel-smoke --case 1
# close→reopen regression: dead dock, dead content widget, P key, no stale refs
```


Reuses `eval_a1_tre`: official↔pack z-flip, optional R3 remap, sub-128 sampling.  
Landmark push-forward T00→T50 uses **−disp**; image warp uses **+disp**.  
`tre/voxelmap_dvf_phase01_mean.npy` is already `(z,y,x,3)` — not HWD.  
**Display** is pack-native `GTVol` mm; DVF math still uses the run’s compute `frame` (`r3`/`native`).
