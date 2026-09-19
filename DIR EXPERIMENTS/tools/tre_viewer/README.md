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


## Window / panels

Multipane layout (napari docks, overridden for a usable default):
- **Center:** CT + TRE overlays (primary canvas)
- **Right tabs:** **TRE** (summary / worst list) ↔ **Controls** (cases + overlays) — tabified so they don’t fight for height; Controls scrolls if needed
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
