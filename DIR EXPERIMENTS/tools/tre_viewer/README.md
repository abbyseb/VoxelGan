# TRE Viewer (`tools/tre_viewer`)

Interactive TRE / DVF / DRR diagnosis for **DIR EXPERIMENTS** runs.

**Branch:** `TRE-VIZ`  
**Default display frame (later phases):** pack-native mm  
**Stack:** dedicated `.venv` here (not LEARN-GUI).

## Phase 0 (this milestone)

Scaffold + run discovery + thin adapter over `scripts/dirlab_tre.py` / `eval_a1_tre.py` + per-landmark cache + verify.

### Setup

```bash
cd "DIR EXPERIMENTS/tools/tre_viewer"
bash setup_venv.sh
```

### Commands

```bash
cd "DIR EXPERIMENTS/tools"
./tre_viewer/.venv/bin/python -m tre_viewer list-runs
./tre_viewer/.venv/bin/python -m tre_viewer verify --arm A1 --case 1
./tre_viewer/.venv/bin/python -m tre_viewer per-landmark --arm A1 --case 1 --field elastix_mha
```

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
- Then field / pair / 75·300 / overlays as before

Keys: `W` jump worst TRE · `Shift+W` worst identity · `A`/`C`/`S` orient · **Reload overlays** in Controls.

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
- Missing VoxelMap cache → infer once from `best.pt` (LEARN torch if needed) and write `tre/voxelmap_dvf_phase01_mean.npy`

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
