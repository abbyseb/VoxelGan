#!/usr/bin/env python3
"""Lung-mask padding: static survey PNGs + PyQtGraph interactive viewer.

Interactive GUI (recommended):
  python scripts/viz_mask_padding.py --gui --patient P6

Controls:
  Patient combo / ← → keys   switch patient
  Mouse wheel / Z slider     scroll axial slices
  Pad slider / [ ] keys      dilate (+) / erode (−)
  Rotation combo / R key     rotate CT + masks together (0/90/180/270°)
  Space                      jump to lung mid-Z
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"
OUT = ROOT / "plots" / "mask_pad"
LIDC_VOLS = ROOT / "AnatomyPretrain" / "data" / "lidc-idri" / "volumes"


def load_mha(path: Path) -> np.ndarray:
    try:
        import SimpleITK as sitk
    except ImportError:
        import itk

        return np.asarray(itk.imread(str(path)), dtype=np.float32)
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def window_ct(x: np.ndarray, lo=None, hi=None) -> np.ndarray:
    x = x.astype(np.float32)
    if lo is None or hi is None:
        lo = float(np.percentile(x, 1))
        hi = float(np.percentile(x, 99))
    if hi <= lo:
        hi = lo + 1e-6
    return np.clip((x - lo) / (hi - lo), 0, 1)


def rot180_vol(vol: np.ndarray) -> np.ndarray:
    """Rotate each axial slice 180° (display upright)."""
    return np.rot90(vol, 2, axes=(1, 2))


def rotate_axial_vol(vol: np.ndarray, k: int) -> np.ndarray:
    """Rotate each axial (Y,X) slice by k * 90° (k in 0..3). Same for CT and masks."""
    k = int(k) % 4
    if k == 0:
        return vol
    return np.rot90(vol, k, axes=(1, 2))


def lung_mid_z(mask: np.ndarray) -> int:
    zz = np.where(mask.any(axis=(1, 2)))[0]
    if zz.size == 0:
        return mask.shape[0] // 2
    return int(zz[len(zz) // 2])


def save_mask_mha(mask_bool: np.ndarray, out_path: Path, ref_mha: Path | None = None):
    """Write uint8 mask; copy geometry from ref Mask_Lung.mha when available."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = mask_bool.astype(np.uint8)
    try:
        import SimpleITK as sitk
    except ImportError:
        import itk

        img = itk.GetImageFromArray(arr)
        if ref_mha is not None and ref_mha.exists():
            ref = itk.imread(str(ref_mha))
            img.SetSpacing(ref.GetSpacing())
            img.SetOrigin(ref.GetOrigin())
            img.SetDirection(ref.GetDirection())
        itk.imwrite(img, str(out_path))
        return
    img = sitk.GetImageFromArray(arr)
    if ref_mha is not None and ref_mha.exists():
        ref = sitk.ReadImage(str(ref_mha))
        img.CopyInformation(ref)
    sitk.WriteImage(img, str(out_path))


def morph_pad(mask: np.ndarray, pad_vox: int) -> np.ndarray:
    """3D EDT dilate/erode on lung bbox (+margin)."""
    m = mask.astype(bool)
    if pad_vox == 0:
        return m
    r = abs(int(pad_vox))
    coords = np.argwhere(m)
    if coords.size == 0:
        return m
    margin = r + 2
    z0, y0, x0 = np.maximum(coords.min(axis=0) - margin, 0)
    z1, y1, x1 = np.minimum(coords.max(axis=0) + margin + 1, np.array(m.shape))
    sub = m[z0:z1, y0:y1, x0:x1]
    out = np.zeros_like(m)
    if pad_vox > 0:
        dist = ndi.distance_transform_edt(~sub)
        out[z0:z1, y0:y1, x0:x1] = dist <= r
        out |= m
    else:
        dist = ndi.distance_transform_edt(sub)
        out[z0:z1, y0:y1, x0:x1] = dist > r
    return out


def morph_pad_slice(mask_sl: np.ndarray, pad_vox: int) -> np.ndarray:
    m = mask_sl.astype(bool)
    if pad_vox == 0:
        return m
    r = abs(int(pad_vox))
    if pad_vox > 0:
        return ndi.distance_transform_edt(~m) <= r
    return ndi.distance_transform_edt(m) > r


# ---------------------------------------------------------------------------
# Static survey (matplotlib, offline)
# ---------------------------------------------------------------------------

def list_patients(root: Path, globpat: str) -> list[str]:
    return sorted(p.name for p in root.glob(globpat) if p.is_dir() and not p.name.startswith("."))


def ct_mask_paths(root: Path, pid: str, phase: int, ct_name: str | None) -> tuple[Path, Path]:
    pdir = root / pid
    ct_path = pdir / (ct_name if ct_name else f"GTVol_{phase:02d}.mha")
    return ct_path, pdir / "Mask_Lung.mha"


def run_survey(
    patients: list[str],
    pads: list[int],
    phase: int = 5,
    root: Path | None = None,
    ct_name: str | None = None,
):
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    n_pad = len(pads)
    cols = 1 + n_pad
    fig, axs = plt.subplots(
        len(patients), cols, figsize=(3.2 * cols, 2.8 * len(patients)), squeeze=False
    )

    def draw(ax, ct_sl, m0, mp, title):
        ax.imshow(ct_sl, cmap="gray", origin="lower", aspect="equal", vmin=0, vmax=1)
        if m0.any():
            ax.contour(m0, levels=[0.5], colors="lime", linewidths=0.8)
        if mp.any():
            ax.contour(mp, levels=[0.5], colors="cyan", linewidths=0.9, linestyles="--")
        ring = mp.astype(bool) ^ m0.astype(bool)
        if ring.any():
            ax.contour(ring.astype(float), levels=[0.5], colors="orange", linewidths=0.5, alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    data_root = root if root is not None else RAW
    for i, pid in enumerate(patients):
        ct_path, mask_path = ct_mask_paths(data_root, pid, phase, ct_name)
        ct = load_mha(ct_path)
        mask = load_mha(mask_path) > 0
        z = lung_mid_z(mask)
        lo, hi = float(np.percentile(ct, 1)), float(np.percentile(ct, 99))
        ct_sl = np.rot90(window_ct(ct[z], lo, hi), 2)
        m0 = np.rot90(mask[z].astype(float), 2)
        draw(axs[i, 0], ct_sl, m0, m0, f"{pid} orig z={z}")
        for j, pad in enumerate(pads):
            mp = morph_pad(mask, pad)
            draw(
                axs[i, j + 1],
                ct_sl,
                m0,
                np.rot90(mp[z].astype(float), 2),
                f"{pid} pad={pad:+d}vx",
            )
        print(f"{pid}: lung mid z={z}  pads={pads}", flush=True)

    fig.suptitle(
        f"Lung mask padding survey"
        + ("" if ct_name else f"  phase {phase:02d}")
        + "  lime=original  cyan=padded  orange=Δ  (180° rot)",
        fontsize=12,
    )
    fig.tight_layout()
    tag = "lidc" if ct_name else f"all_patients"
    out = OUT / f"{tag}_mask_pad_{'_'.join(str(p) for p in pads)}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"wrote {out}", flush=True)

    for pid in patients:
        ct_path, mask_path = ct_mask_paths(data_root, pid, phase, ct_name)
        ct = load_mha(ct_path)
        mask = load_mha(mask_path) > 0
        z = lung_mid_z(mask)
        lo, hi = float(np.percentile(ct, 1)), float(np.percentile(ct, 99))
        ct_sl = np.rot90(window_ct(ct[z], lo, hi), 2)
        m0 = np.rot90(mask[z].astype(float), 2)
        fig, axs = plt.subplots(1, cols, figsize=(3.2 * cols, 3.2))
        axs = np.atleast_1d(axs)
        draw(axs[0], ct_sl, m0, m0, f"{pid} orig")
        for j, pad in enumerate(pads):
            mp = morph_pad(mask, pad)
            draw(axs[j + 1], ct_sl, m0, np.rot90(mp[z].astype(float), 2), f"pad={pad:+d}")
        fig.suptitle(f"{pid} lung mid z={z}")
        fig.tight_layout()
        fig.savefig(OUT / f"{pid}_mask_pad.png", dpi=130)
        plt.close(fig)


# ---------------------------------------------------------------------------
# PyQtGraph GUI
# ---------------------------------------------------------------------------

def run_gui(
    patient: str,
    phase: int = 5,
    init_pad: int = 8,
    root: Path | None = None,
    ct_name: str | None = None,
    patient_glob: str = "P*",
):
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets

    data_root = root if root is not None else RAW
    patients = list_patients(data_root, patient_glob)
    if not patients:
        raise SystemExit(f"no patients under {data_root} matching {patient_glob}")
    if patient not in patients:
        patient = patients[0]
        print(f"falling back to {patient}", flush=True)

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

    class MaskPadViewer(QtWidgets.QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle(
                "LIDC lung mask pad (PyQtGraph)"
                if ct_name
                else "PopulationStudy — lung mask pad (PyQtGraph)"
            )
            self.resize(1100, 900)

            self.phase = phase
            self.ct = None          # native attenuation
            self.mask = None        # native bool
            self.ct_win = None      # windowed native float32
            self.ct_disp = None     # windowed + rotated for display
            self.mask_disp = None
            self.mid_z = 0
            self.rot_k = 2  # default 180° (matches survey PNGs)
            self._pad_cache = {}
            self._updating = False

            central = QtWidgets.QWidget()
            self.setCentralWidget(central)
            layout = QtWidgets.QHBoxLayout(central)

            # ---- left controls ----
            panel = QtWidgets.QWidget()
            panel.setFixedWidth(260)
            form = QtWidgets.QVBoxLayout(panel)

            form.addWidget(QtWidgets.QLabel("<b>Patient</b>"))
            self.combo = QtWidgets.QComboBox()
            self.combo.addItems(patients)
            self.combo.setCurrentText(patient)
            self.combo.currentTextChanged.connect(self.on_patient)
            form.addWidget(self.combo)

            form.addWidget(QtWidgets.QLabel("<b>Phase</b>"))
            self.phase_spin = QtWidgets.QSpinBox()
            self.phase_spin.setRange(1, 10)
            self.phase_spin.setValue(phase)
            self.phase_spin.valueChanged.connect(self.on_phase)
            form.addWidget(self.phase_spin)
            if ct_name:
                self.phase_spin.setEnabled(False)
                self.phase_spin.setToolTip("LIDC volumes have a single CT (no 4D phases)")

            form.addWidget(QtWidgets.QLabel("<b>Rotation (CT + mask)</b>"))
            self.rot_combo = QtWidgets.QComboBox()
            self.rot_combo.addItems(["0°", "90°", "180°", "270°"])
            self.rot_combo.setCurrentIndex(self.rot_k)
            self.rot_combo.currentIndexChanged.connect(self.on_rotation)
            form.addWidget(self.rot_combo)

            form.addWidget(QtWidgets.QLabel("<b>Z slice</b>"))
            self.z_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.z_slider.setMinimum(0)
            self.z_slider.setMaximum(1)
            self.z_slider.valueChanged.connect(self.on_z)
            form.addWidget(self.z_slider)
            self.z_label = QtWidgets.QLabel("z=0")
            form.addWidget(self.z_label)

            form.addWidget(QtWidgets.QLabel("<b>Pad (voxels)</b>  +dilate / −erode"))
            self.pad_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
            self.pad_slider.setMinimum(-20)
            self.pad_slider.setMaximum(30)
            self.pad_slider.setValue(init_pad)
            self.pad_slider.valueChanged.connect(self.on_pad)
            form.addWidget(self.pad_slider)
            self.pad_label = QtWidgets.QLabel(f"pad={init_pad:+d}")
            form.addWidget(self.pad_label)

            self.chk_3d = QtWidgets.QCheckBox("3D pad (slower, accurate)")
            self.chk_3d.setChecked(False)
            self.chk_3d.stateChanged.connect(self.on_pad_mode)
            form.addWidget(self.chk_3d)

            btn_mid = QtWidgets.QPushButton("Jump to lung mid-Z (Space)")
            btn_mid.clicked.connect(self.jump_mid)
            form.addWidget(btn_mid)

            btn_save = QtWidgets.QPushButton("Save padded mask (Ctrl+S)")
            btn_save.setToolTip(
                "Writes 3D padded Mask_Lung in native orientation "
                "(not display-rotated) under masks_padded/"
            )
            btn_save.clicked.connect(self.save_padded_mask)
            form.addWidget(btn_save)

            form.addWidget(QtWidgets.QLabel(
                "<small>"
                "lime = original&nbsp;&nbsp;cyan = padded<br>"
                "Wheel / Z slider = slices<br>"
                "← → = prev/next patient<br>"
                "[ ] = pad −1 / +1<br>"
                "R = cycle rotation (CT+mask)<br>"
                "Ctrl+S = save padded mask"
                "</small>"
            ))
            form.addStretch(1)
            layout.addWidget(panel)

            # ---- image view ----
            right = QtWidgets.QVBoxLayout()
            self.view = pg.ImageView(view=pg.PlotItem())
            self.view.ui.roiBtn.hide()
            self.view.ui.menuBtn.hide()
            # ImageView built-in time slider = Z for 3D stacks
            self.view.sigTimeChanged.connect(self.on_view_time)
            right.addWidget(self.view)

            self.status = QtWidgets.QLabel("")
            right.addWidget(self.status)
            layout.addLayout(right, stretch=1)

            # overlay ImageItems for masks (same transform as main image)
            self.overlay_orig = pg.ImageItem()
            self.overlay_pad = pg.ImageItem()
            self.view.getView().addItem(self.overlay_orig)
            self.view.getView().addItem(self.overlay_pad)
            # green / cyan with alpha via LUT
            self.overlay_orig.setZValue(10)
            self.overlay_pad.setZValue(11)
            self._set_overlay_luts()

            self.load_patient(patient)
            self.show()

        def _set_overlay_luts(self):
            # 0 transparent, 1 colored edge-ish fill
            lut_g = np.zeros((256, 4), dtype=np.ubyte)
            lut_g[128:, 1] = 220
            lut_g[128:, 3] = 90
            lut_c = np.zeros((256, 4), dtype=np.ubyte)
            lut_c[128:, 1] = 200
            lut_c[128:, 2] = 255
            lut_c[128:, 3] = 110
            self.overlay_orig.setLookupTable(lut_g)
            self.overlay_pad.setLookupTable(lut_c)

        def keyPressEvent(self, event):
            key = event.key()
            if key == QtCore.Qt.Key.Key_Left:
                i = max(0, self.combo.currentIndex() - 1)
                self.combo.setCurrentIndex(i)
            elif key == QtCore.Qt.Key.Key_Right:
                i = min(self.combo.count() - 1, self.combo.currentIndex() + 1)
                self.combo.setCurrentIndex(i)
            elif key == QtCore.Qt.Key.Key_BracketLeft:
                self.pad_slider.setValue(self.pad_slider.value() - 1)
            elif key == QtCore.Qt.Key.Key_BracketRight:
                self.pad_slider.setValue(self.pad_slider.value() + 1)
            elif key == QtCore.Qt.Key.Key_R:
                self.rot_combo.setCurrentIndex((self.rot_combo.currentIndex() + 1) % 4)
            elif key == QtCore.Qt.Key.Key_S and (
                event.modifiers() & QtCore.Qt.KeyboardModifier.ControlModifier
            ):
                self.save_padded_mask()
            elif key == QtCore.Qt.Key.Key_Space:
                self.jump_mid()
            elif key in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Comma):
                self.z_slider.setValue(self.z_slider.value() + 1)
            elif key in (QtCore.Qt.Key.Key_Down, QtCore.Qt.Key.Key_Period):
                self.z_slider.setValue(self.z_slider.value() - 1)
            else:
                super().keyPressEvent(event)

        def apply_display_rotation(self, keep_z: bool = True):
            """Rebuild CT + mask display volumes with the same axial rotation."""
            if self.ct_win is None or self.mask is None:
                return
            z_keep = int(self.z_slider.value()) if keep_z and self.ct_disp is not None else None
            self.rot_k = int(self.rot_combo.currentIndex()) % 4
            self.ct_disp = rotate_axial_vol(self.ct_win, self.rot_k).astype(np.float32)
            self.mask_disp = rotate_axial_vol(self.mask.astype(np.float32), self.rot_k)
            self._pad_cache.clear()

            self._updating = True
            self.view.setImage(self.ct_disp, autoLevels=False, levels=(0.0, 1.0))
            self.z_slider.setMaximum(self.ct_disp.shape[0] - 1)
            z = self.mid_z if z_keep is None else int(np.clip(z_keep, 0, self.ct_disp.shape[0] - 1))
            self.z_slider.setValue(z)
            self.view.setCurrentIndex(z)
            self.z_label.setText(f"z={z} / {self.ct_disp.shape[0] - 1}")
            self._updating = False
            self.refresh_overlays()

        def load_patient(self, pid: str):
            self._updating = True
            self.status.setText(f"Loading {pid}…")
            QtWidgets.QApplication.processEvents()
            ph = int(self.phase_spin.value())
            ct_path, mask_path = ct_mask_paths(data_root, pid, ph, ct_name)
            ct = load_mha(ct_path)
            mask = load_mha(mask_path) > 0
            lo, hi = float(np.percentile(ct, 1)), float(np.percentile(ct, 99))
            self.ct = ct
            self.mask = mask
            self.ct_win = window_ct(ct, lo, hi).astype(np.float32)
            self.mid_z = lung_mid_z(mask)
            self._pad_cache.clear()
            self._updating = False
            # rotate CT + mask together (default 180°)
            self.apply_display_rotation(keep_z=False)
            self.status.setText(
                f"{pid}  phase {ph:02d}  shape={tuple(self.ct.shape)}  "
                f"lung mid-Z={self.mid_z}  rot={self.rot_k * 90}°"
            )

        def on_patient(self, pid: str):
            if not pid:
                return
            self.load_patient(pid)

        def on_phase(self, _v):
            self.load_patient(self.combo.currentText())

        def on_rotation(self, _idx):
            self.apply_display_rotation(keep_z=True)

        def on_z(self, z: int):
            if self._updating or self.ct_disp is None:
                return
            self._updating = True
            self.view.setCurrentIndex(int(z))
            self.z_label.setText(f"z={z} / {self.ct_disp.shape[0] - 1}")
            self._updating = False
            self.refresh_overlays()

        def on_view_time(self, *_args):
            if self._updating or self.ct_disp is None:
                return
            z = int(self.view.currentIndex)
            if self.z_slider.value() != z:
                self._updating = True
                self.z_slider.setValue(z)
                self.z_label.setText(f"z={z} / {self.ct_disp.shape[0] - 1}")
                self._updating = False
            self.refresh_overlays()

        def on_pad(self, v: int):
            self.pad_label.setText(f"pad={v:+d}")
            self.refresh_overlays()

        def on_pad_mode(self, _s):
            self._pad_cache.clear()
            self.refresh_overlays()

        def jump_mid(self):
            self.z_slider.setValue(self.mid_z)

        def save_padded_mask(self):
            """Save 3D padded lung mask in native orientation for downstream prep."""
            if self.mask is None:
                return
            pid = self.combo.currentText()
            pad = int(self.pad_slider.value())
            self.status.setText(f"Computing 3D pad={pad:+d} for {pid}…")
            QtWidgets.QApplication.processEvents()
            padded = morph_pad(self.mask, pad)
            sign = "p" if pad >= 0 else "m"
            name = f"Mask_Lung_pad{sign}{abs(pad):02d}.mha"
            default = (data_root.parent / "padded_masks" / pid / name)
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save padded lung mask",
                str(default),
                "MetaImage (*.mha);;All (*)",
            )
            if not path:
                self.status.setText("Save cancelled")
                return
            out = Path(path)
            ref = (data_root / pid / "Mask_Lung.mha")
            try:
                save_mask_mha(padded, out, ref_mha=ref)
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Save failed", str(e))
                self.status.setText(f"Save failed: {e}")
                return
            n = int(padded.sum())
            self.status.setText(f"Saved {out}  ({n} voxels, pad={pad:+d}, native orientation)")
            QtWidgets.QMessageBox.information(
                self,
                "Saved",
                f"Wrote padded mask:\n{out}\n\n"
                f"pad={pad:+d} vx (3D EDT)\n"
                f"voxels={n}\n"
                f"orientation = native (not display-rotated)",
            )

        def _padded_mask_disp(self, pad: int) -> np.ndarray:
            use_3d = self.chk_3d.isChecked()
            key = (pad, use_3d, self.rot_k)
            if key in self._pad_cache:
                return self._pad_cache[key]
            if pad == 0:
                out = self.mask_disp > 0.5
            elif use_3d:
                padded = morph_pad(self.mask, pad)
                out = rotate_axial_vol(padded.astype(np.float32), self.rot_k) > 0.5
            else:
                out = np.empty_like(self.mask_disp, dtype=bool)
                for z in range(self.mask_disp.shape[0]):
                    out[z] = morph_pad_slice(self.mask_disp[z] > 0.5, pad)
            self._pad_cache[key] = out
            return out

        def refresh_overlays(self):
            if self.ct_disp is None:
                return
            z = int(self.z_slider.value())
            pad = int(self.pad_slider.value())
            m0 = (self.mask_disp[z] > 0.5).astype(np.float32)
            mp_vol = self._padded_mask_disp(pad)
            mp = mp_vol[z].astype(np.float32)

            self.overlay_orig.setImage(m0 * 255.0, autoLevels=False, levels=(0, 255))
            self.overlay_pad.setImage(mp * 255.0, autoLevels=False, levels=(0, 255))

            cov = float(mp.sum())
            orig = float(m0.sum())
            self.status.setText(
                f"{self.combo.currentText()}  phase {self.phase_spin.value():02d}  "
                f"z={z}/{self.ct_disp.shape[0]-1}  rot={self.rot_k * 90}°  "
                f"pad={pad:+d}  mask px orig={orig:.0f} padded={cov:.0f}  "
                f"{'3D' if self.chk_3d.isChecked() else '2D'} pad"
            )

    win = MaskPadViewer()
    print(
        "PyQtGraph GUI ready — CT+mask share rotation (default 180°). "
        "R or Rotation combo; wheel/Z slices; ←→ patients; [ ] pad; Space mid-Z.",
        flush=True,
    )
    app.exec()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--survey", action="store_true", help="Write static padding comparison PNGs")
    ap.add_argument("--gui", action="store_true", help="PyQtGraph interactive viewer")
    ap.add_argument("--pads", default="8,16", help="Dilation radii for --survey")
    ap.add_argument("--patient", default="P6")
    ap.add_argument("--patients", default="")
    ap.add_argument("--phase", type=int, default=5)
    ap.add_argument("--init_pad", type=int, default=8)
    ap.add_argument("--root", type=Path, default=None, help="Patient folder parent (default: PopulationStudy/raw)")
    ap.add_argument("--patient-glob", default="P*")
    ap.add_argument("--ct-name", default=None, help="CT filename inside each patient folder (e.g. CT.mha)")
    ap.add_argument(
        "--lidc",
        action="store_true",
        help="LIDC-IDRI R231 volumes: AnatomyPretrain/data/lidc-idri/volumes",
    )
    args = ap.parse_args()

    if args.lidc:
        args.root = LIDC_VOLS
        args.patient_glob = "LIDC-IDRI-*"
        args.ct_name = "CT.mha"
        if args.patient in ("P6", ""):
            args.patient = ""
        OUT_LIDC = ROOT / "AnatomyPretrain" / "plots" / "mask_pad"
        global OUT
        OUT = OUT_LIDC

    data_root = args.root if args.root is not None else RAW
    if args.patients:
        patients = [p.strip() for p in args.patients.split(",") if p.strip()]
    else:
        patients = list_patients(data_root, args.patient_glob)
    if not patients:
        raise SystemExit(f"no patients under {data_root}")
    if not args.patient:
        args.patient = patients[0]
    pads = [int(x) for x in args.pads.split(",") if x.strip()]

    if not args.survey and not args.gui:
        args.gui = True

    if args.survey:
        run_survey(
            patients,
            pads,
            phase=args.phase,
            root=data_root,
            ct_name=args.ct_name,
        )
    if args.gui:
        run_gui(
            args.patient,
            phase=args.phase,
            init_pad=args.init_pad,
            root=data_root,
            ct_name=args.ct_name,
            patient_glob=args.patient_glob,
        )


if __name__ == "__main__":
    main()
