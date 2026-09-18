"""Napari TRE Viewer — Phase 1 volume + landmark overlays."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_TOOLS = Path(__file__).resolve().parent.parent
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

from tre_viewer.data import (  # noqa: E402
    CASE_INFO,
    PhasePair,
    RunRef,
    default_field,
    discover_runs_in_folder,
    field_warp_bundle,
    load_pack_volume,
    load_tre_summary,
    normalize_runs_dir,
    pair_phases,
    per_landmark,
    resolve_run,
    xyz_to_zyx,
)
from tre_viewer.encodings import (  # noqa: E402
    error_vectors_zyx,
    summary_line,
    tre_to_rgba,
    worst_table,
)
from tre_viewer.warp_dvf import (  # noqa: E402
    decimated_arrows_zyx,
    upsample_dvf_to_pack,
)

# napari dims.order: which axis is the slider (first) for 2D view of last two
# Pack array is (z,y,x). Base pose = 0° in-plane; R cycles +90° via axis swap+flips.
_ORIENT = {
    "axial": (0, 1, 2),  # slider z — rows=y (AP), cols=x (LR)
    "coronal": (1, 0, 2),  # slider y — rows=z (SI), cols=x (LR)
    "sagittal": (2, 0, 1),  # slider x — rows=z (SI), cols=y (AP)
}

# Deleted-wrapper probes, newest binding first. shiboken6 is PySide6-only, so a
# PyQt6 install must fall through to PyQt6.sip — otherwise nothing checks
# validity and a deleted dock looks alive until it raises RuntimeError.
_WRAPPER_PROBES = (
    ("shiboken6", "isValid", True),
    ("shiboken2", "isValid", True),
    ("PyQt6.sip", "isdeleted", False),
    ("PyQt5.sip", "isdeleted", False),
    ("sip", "isdeleted", False),
)
_wrapper_checks: list[tuple] | None = None


def _qt_wrapper_alive(obj) -> bool | None:
    """True/False if a C++-wrapper check applies to ``obj``, else None.

    Safe to call on already-deleted wrappers: no attribute access on ``obj``.
    """
    global _wrapper_checks
    if _wrapper_checks is None:
        import importlib

        _wrapper_checks = []
        for mod, attr, alive_if_true in _WRAPPER_PROBES:
            try:
                fn = getattr(importlib.import_module(mod), attr)
            except Exception:  # noqa: BLE001 — binding not installed
                continue
            _wrapper_checks.append((fn, alive_if_true))
    for fn, alive_if_true in _wrapper_checks:
        try:
            res = bool(fn(obj))
        except TypeError:
            continue  # not a wrapper of *this* binding (e.g. FunctionGui)
        except RuntimeError:
            return False
        return res if alive_if_true else not res
    return None


class TreViewerApp:
    """Interactive pack-mm CT + TRE landmark overlays."""

    def __init__(
        self,
        run: RunRef,
        *,
        field: str | None = None,
        which: str = "75",
        pair: PhasePair = "T00_T50",
        show: bool = True,
    ):
        import napari

        self.run = run
        self.which = which
        self.pair: PhasePair = pair
        self.field = field or default_field(run)
        self._worst_cycle = 0
        # Case browser catalog (folder of DIR_Cxx runs)
        self._catalog: list[RunRef] = [run]
        self._runs_dir = run.run_root.parent
        try:
            discovered = discover_runs_in_folder(self._runs_dir)
            if discovered:
                self._catalog = discovered
                # keep current run object if present in catalog
                for r in discovered:
                    if r.case == run.case and r.run_root == run.run_root:
                        self.run = r
                        break
        except Exception:
            pass

        src_ph, dst_ph = pair_phases(pair)
        self.vol_dst = load_pack_volume(run, dst_ph)  # target (T50 for KPI)
        self.vol_src = load_pack_volume(run, src_ph)

        self.viewer = napari.Viewer(
            title=(
                f"TRE Viewer │ {run.arm} C{run.case:02d} │ "
                f"frame={run.frame} │ {self.field}"
            ),
            show=show,
        )
        scale = self.vol_dst.scale  # (dz,dy,dx) mm

        self.layer_ct = self.viewer.add_image(
            self.vol_dst.data,
            name=f"CT {dst_ph} (target)",
            scale=scale,
            colormap="gray",
            blending="translucent",
        )
        # Window roughly to soft tissue / lung HU
        self.layer_ct.contrast_limits = (-1000.0, 500.0)

        self.layer_src = self.viewer.add_image(
            self.vol_src.data,
            name=f"CT {src_ph} (source)",
            scale=scale,
            colormap="gray",
            blending="additive",
            opacity=0.0,  # off by default; toggle for blink later
            visible=False,
        )
        self.layer_src.contrast_limits = (-1000.0, 500.0)

        # Point sizes are in *data* coords; keep them large enough on 256² FOV.
        # projection_mode='all' → show points from other slices (shrunk).
        _pt = dict(
            scale=scale,
            border_color="white",
            projection_mode="all",
            blending="translucent",
        )
        self.layer_truth = self.viewer.add_points(
            np.zeros((0, 3)),
            name="truth (dst)",
            size=12.0,
            face_color="lime",
            border_width=0.15,
            shading="spherical",
            **_pt,
        )
        self.layer_pred = self.viewer.add_points(
            np.zeros((0, 3)),
            name="pred",
            size=10.0,
            face_color="magenta",
            border_width=0.15,
            symbol="disc",
            **_pt,
        )
        self.layer_src_pts = self.viewer.add_points(
            np.zeros((0, 3)),
            name="source (src)",
            size=8.0,
            face_color="cyan",
            border_width=0.15,
            opacity=0.5,
            visible=False,
            **_pt,
        )
        self.layer_rings = self.viewer.add_points(
            np.zeros((0, 3)),
            name="TRE rings",
            size=18.0,
            face_color=[0, 0, 0, 0],
            border_width=0.35,
            opacity=1.0,
            **_pt,
        )
        self.layer_err = self.viewer.add_vectors(
            np.zeros((0, 2, 3)),
            name="error vectors",
            scale=scale,
            edge_width=1.5,
            length=1.0,
            vector_style="triangle",
            opacity=0.95,
        )

        # --- Phase 2: DVF / warp overlays (pack-shaped upsamples; off by default) ---
        self._blink_warped = False
        self._bundle = None
        z0 = np.zeros_like(self.vol_dst.data, dtype=np.float32)
        self.layer_dvf_mag = self.viewer.add_image(
            z0,
            name="DVF |u| mm",
            scale=scale,
            colormap="magma",
            blending="additive",
            opacity=0.55,
            visible=False,
        )
        self.layer_dvf_comp = self.viewer.add_image(
            z0,
            name="DVF component mm",
            scale=scale,
            colormap="coolwarm",
            blending="additive",
            opacity=0.55,
            visible=False,
        )
        self.layer_warped = self.viewer.add_image(
            z0,
            name="warped source",
            scale=scale,
            colormap="gray",
            blending="translucent",
            opacity=1.0,
            visible=False,
        )
        self.layer_warped.contrast_limits = (-1000.0, 500.0)
        self.layer_diff = self.viewer.add_image(
            z0,
            name="target − warped",
            scale=scale,
            colormap="coolwarm",
            blending="additive",
            opacity=0.7,
            visible=False,
        )
        self.layer_ident_diff = self.viewer.add_image(
            z0,
            name="target − source (identity)",
            scale=scale,
            colormap="coolwarm",
            blending="additive",
            opacity=0.7,
            visible=False,
        )
        self.layer_dvf_arrows = self.viewer.add_vectors(
            np.zeros((0, 2, 3)),
            name="DVF arrows",
            scale=scale,
            edge_width=1.0,
            length=1.0,
            edge_color="cyan",
            visible=False,
        )

        # Show landmarks within ~±3 slices of the slider (world mm margins).
        self._apply_thick_slices()

        # In-plane ornament state: 0/90/180/270 via dims.order swap (all planes).
        # Set before the panels are built — the Controls panel seeds its widget
        # defaults from this live state so a rebuild keeps the current pose.
        self._orient_name = "axial"
        self._rot90_k = 0  # 0,1,2,3 → 0°,90°,180°,270°
        self._flip_lr = False
        self._flip_ud = False
        self._dvf_component = "off"
        self._arrow_step = 6
        self._arrow_gain = 2.0

        # --- docks (LEARN-like multipane: CT center, panels right, DRR bottom) ---
        # Every panel is registered with a *builder*, not just a widget: any
        # napari destroy path (title-bar X, remove_dock_widget) can delete the
        # QtViewerDockWidget, so show_panel must be able to rebuild from scratch.
        self._docks: dict[str, object] = {}
        self._panel_specs: dict[str, dict] = {}
        self._register_dock(
            name="TRE", builder=self._build_tre_panel, area="right", min_w=280
        )
        self._register_dock(
            name="Controls",
            builder=self._build_controls_panel,
            area="right",
            min_w=260,
        )

        # Keybindings
        @self.viewer.bind_key("w")
        def _jump_worst(viewer):  # noqa: ARG001
            self.jump_worst()

        @self.viewer.bind_key("Shift-W")
        def _jump_worst_ident(viewer):  # noqa: ARG001
            self.jump_worst(by_identity=True)

        @self.viewer.bind_key("a")
        def _axial(viewer):  # noqa: ARG001
            self._set_orient("axial")

        @self.viewer.bind_key("c")
        def _coronal(viewer):  # noqa: ARG001
            self._set_orient("coronal")

        @self.viewer.bind_key("s")
        def _sagittal(viewer):  # noqa: ARG001
            self._set_orient("sagittal")

        @self.viewer.bind_key("Space")
        def _blink(viewer):  # noqa: ARG001
            self.blink_target_warped()

        @self.viewer.bind_key("r")
        def _rot90(viewer):  # noqa: ARG001
            self.rotate_inplane(90)

        @self.viewer.bind_key("Shift-R")
        def _rot180(viewer):  # noqa: ARG001
            self.rotate_inplane(180)

        @self.viewer.bind_key("f")
        def _flip_lr(viewer):  # noqa: ARG001
            self.flip_inplane(lr=True)

        @self.viewer.bind_key("Shift-F")
        def _flip_ud(viewer):  # noqa: ARG001
            self.flip_inplane(ud=True)

        @self.viewer.bind_key("0")
        def _reset_orn(viewer):  # noqa: ARG001
            self.reset_inplane()

        self._build_drr_dock()
        self._setup_window_chrome(show=show)
        self._set_orient("axial")
        self._refresh_landmarks()
        self._refresh_dvf_overlays()
        self._refresh_drr_panel()
        # Land on the worst landmark so points are immediately visible.
        if len(self.pl.tre_mm):
            self.jump_to_landmark(int(np.argmax(self.pl.tre_mm)))
        self._add_handedness_overlay()
        self._apply_inplane_ornament()

    # ------------------------------------------------------------------
    def _qt_alive(self, obj) -> bool:
        """True if obj is usable. Never attribute-access a possibly-deleted wrapper first."""
        if obj is None:
            return False
        # 1) Binding-specific validity check FIRST — safe on deleted wrappers,
        #    getattr is NOT (it raises RuntimeError).
        state = _qt_wrapper_alive(obj)
        if state is not None:
            return state
        # 2) magicgui FunctionGui & friends → check the nested QWidget.
        #    object.__getattribute__ bypasses the binding's __getattr__, so it
        #    reports AttributeError (not RuntimeError) even for dead QWidgets —
        #    never treat that as "alive" on its own.
        try:
            native = object.__getattribute__(obj, "native")
        except AttributeError:
            native = None
        except RuntimeError:
            return False
        if native is not None and native is not obj:
            return self._qt_alive(native)
        # 3) Unknown binding: probing a real QObject method is RuntimeError-safe.
        try:
            obj.objectName()
        except RuntimeError:
            return False
        except Exception:  # noqa: BLE001 — not a QObject at all
            pass
        return True

    # ---- panel builders (each one re-creatable from scratch) ----------
    def _build_tre_panel(self):
        """Summary / selection / worst-landmark list (right dock)."""
        from qtpy.QtWidgets import QLabel, QListWidget, QVBoxLayout, QWidget

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setStyleSheet("font-family: monospace; padding: 6px;")
        self.selected_label = QLabel("selected: —")
        self.selected_label.setStyleSheet("font-family: monospace; padding: 4px;")
        self.worst_list = QListWidget()
        self.worst_list.itemClicked.connect(self._on_worst_clicked)

        side = QWidget()
        side.setMinimumWidth(280)
        lay = QVBoxLayout(side)
        lay.addWidget(QLabel("Summary"))
        lay.addWidget(self.summary_label)
        lay.addWidget(self.selected_label)
        lay.addWidget(QLabel("Worst landmarks (click / press W)"))
        lay.addWidget(self.worst_list)
        self._tre_panel = side
        return side

    def _build_controls_panel(self):
        """Folder browse + case dropdown + magicgui overlay controls."""
        from magicgui import magicgui
        from qtpy.QtWidgets import (
            QComboBox,
            QFileDialog,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QVBoxLayout,
            QWidget,
        )

        fields = list(self.run.fields_available) or ["identity"]

        @magicgui(
            call_button="Reload",
            field={"choices": fields},
            pair={"choices": ["T00_T50", "T50_T00"]},
            which={"choices": ["75", "300"]},
            orient={"choices": list(_ORIENT.keys())},
            dvf_component={"choices": ["off", "mag", "SI", "AP", "LR"]},
            auto_call=False,
        )
        def controls(
            field: str = self.field,
            pair: str = self.pair,
            which: str = self.which,
            orient: str = self._orient_name,
            show_src_pts: bool = self.layer_src_pts.visible,
            show_rings: bool = self.layer_rings.visible,
            show_vectors: bool = self.layer_err.visible,
            show_dvf_arrows: bool = self.layer_dvf_arrows.visible,
            show_warped: bool = self.layer_warped.visible,
            show_diff: bool = self.layer_diff.visible,
            show_ident_diff: bool = self.layer_ident_diff.visible,
            dvf_component: str = self._dvf_component,
            arrow_step: int = self._arrow_step,
            arrow_gain: float = self._arrow_gain,
        ):
            self.field = field
            self.pair = pair  # type: ignore[assignment]
            self.which = which
            self.layer_src_pts.visible = show_src_pts
            self.layer_rings.visible = show_rings
            self.layer_err.visible = show_vectors
            self.layer_dvf_arrows.visible = show_dvf_arrows
            self.layer_warped.visible = show_warped
            self.layer_diff.visible = show_diff
            self.layer_ident_diff.visible = show_ident_diff
            self._dvf_component = dvf_component
            self._arrow_step = int(arrow_step)
            self._arrow_gain = float(arrow_gain)
            self._set_orient(orient)
            self._reload_volumes_if_needed()
            self._refresh_landmarks()
            self._refresh_dvf_overlays()
            self._refresh_drr_panel()
            self._apply_inplane_ornament()

        self.controls = controls
        try:
            controls.native.setMinimumWidth(260)
        except Exception:
            pass

        root = QWidget()
        root.setMinimumWidth(280)
        lay = QVBoxLayout(root)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        lay.addWidget(QLabel("Runs folder"))
        path_row = QHBoxLayout()
        self._runs_dir_edit = QLineEdit(str(self._runs_dir))
        self._runs_dir_edit.setPlaceholderText("…/arms/A3_synth_conditioned/runs")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_runs_dir)
        path_row.addWidget(self._runs_dir_edit, stretch=1)
        path_row.addWidget(browse)
        lay.addLayout(path_row)

        scan_btn = QPushButton("Scan folder")
        scan_btn.clicked.connect(self._scan_runs_dir_from_edit)
        lay.addWidget(scan_btn)

        lay.addWidget(QLabel("Case / patient"))
        self._case_combo = QComboBox()
        self._case_combo.setMinimumWidth(240)
        self._case_combo.currentIndexChanged.connect(self._on_case_combo_changed)
        lay.addWidget(self._case_combo)
        self._case_tre_label = QLabel("TRE: —")
        self._case_tre_label.setStyleSheet("font-family: monospace; padding: 2px;")
        self._case_tre_label.setWordWrap(True)
        lay.addWidget(self._case_tre_label)

        lay.addWidget(QLabel("Overlays"))
        try:
            lay.addWidget(controls.native)
        except Exception:
            lay.addWidget(controls)

        self._controls_panel = root
        self._refill_case_combo(select_case=self.run.case)
        self._update_case_tre_label()
        return root

    def _case_combo_label(self, run: RunRef) -> str:
        fields = ",".join(run.fields_available[:3]) or "—"
        summ = "TRE✓" if run.tre_summary_path else "no-sum"
        return f"C{run.case:02d}  [{summ}]  {fields}"

    def _refill_case_combo(self, *, select_case: int | None = None) -> None:
        combo = getattr(self, "_case_combo", None)
        if combo is None or not self._qt_alive(combo):
            return
        combo.blockSignals(True)
        combo.clear()
        for run in self._catalog:
            combo.addItem(self._case_combo_label(run), run.case)
        # select
        want = select_case if select_case is not None else self.run.case
        idx = 0
        for i in range(combo.count()):
            if combo.itemData(i) == want:
                idx = i
                break
        combo.setCurrentIndex(idx)
        combo.blockSignals(False)

    def _browse_runs_dir(self) -> None:
        from qtpy.QtWidgets import QFileDialog

        start = str(self._runs_dir) if self._runs_dir else str(Path.home())
        path = QFileDialog.getExistingDirectory(
            None,
            "Select runs folder (arm root, …/runs, or a DIR_Cxx parent)",
            start,
        )
        if not path:
            return
        self._runs_dir_edit.setText(path)
        self._load_runs_dir(path)

    def _scan_runs_dir_from_edit(self) -> None:
        path = self._runs_dir_edit.text().strip()
        if path:
            self._load_runs_dir(path)

    def _load_runs_dir(self, path: str | Path) -> None:
        try:
            runs_dir = normalize_runs_dir(path)
            catalog = discover_runs_in_folder(runs_dir)
        except Exception as exc:
            self.viewer.status = f"Runs folder error: {exc}"
            return
        if not catalog:
            self.viewer.status = f"No DIR_Cxx runs in {runs_dir}"
            return
        self._runs_dir = runs_dir
        self._catalog = catalog
        if self._qt_alive(getattr(self, "_runs_dir_edit", None)):
            self._runs_dir_edit.setText(str(runs_dir))
        # Prefer same case number if present, else first
        pick = next((r for r in catalog if r.case == self.run.case), catalog[0])
        self._refill_case_combo(select_case=pick.case)
        self._switch_to_run(pick)
        self.viewer.status = f"Loaded {len(catalog)} case(s) from {runs_dir}"

    def _on_case_combo_changed(self, index: int) -> None:
        if index < 0 or not self._catalog:
            return
        case = self._case_combo.itemData(index)
        if case is None:
            return
        case = int(case)
        if case == self.run.case and any(
            r.run_root == self.run.run_root for r in self._catalog if r.case == case
        ):
            # still update TRE label
            self._update_case_tre_label()
            return
        hit = next((r for r in self._catalog if r.case == case), None)
        if hit is None:
            return
        self._switch_to_run(hit)

    def _update_case_tre_label(self) -> None:
        lab = getattr(self, "_case_tre_label", None)
        if lab is None or not self._qt_alive(lab):
            return
        summ = load_tre_summary(self.run)
        if not summ:
            lab.setText(f"C{self.run.case:02d}: no tre_summary.json")
            return
        try:
            arms = summ.get("arms") or {}
            vm = (arms.get("voxelmap") or {}).get("75", {}).get("T00_T50") or {}
            reg = (vm.get("registered") or {}).get("mean")
            ident = (vm.get("identity") or {}).get("mean")
            el = (
                (arms.get("elastix_mha") or {}).get("75", {}).get("T00_T50", {})
                .get("registered", {})
                .get("mean")
            )
            parts = [f"C{self.run.case:02d} TRE75"]
            if reg is not None:
                parts.append(f"VM={reg:.2f}")
            if el is not None:
                parts.append(f"El={el:.2f}")
            if ident is not None:
                parts.append(f"id={ident:.2f}")
            lab.setText("  ".join(parts) + " mm")
        except Exception:
            lab.setText(f"C{self.run.case:02d}: tre_summary present")

    def _sync_field_choices(self) -> None:
        controls = getattr(self, "controls", None)
        if controls is None:
            return
        fields = list(self.run.fields_available) or ["identity"]
        try:
            controls.field.choices = fields
            if self.field in fields:
                controls.field.value = self.field
            else:
                self.field = default_field(self.run)
                controls.field.value = self.field
        except Exception:
            pass

    def _switch_to_run(self, run: RunRef) -> None:
        """Hot-swap patient/case without restarting napari."""
        self.run = run
        if self.field not in run.fields_available:
            self.field = default_field(run)
        self._bundle = None
        self._worst_cycle = 0

        src_ph, dst_ph = pair_phases(self.pair)
        self.vol_dst = load_pack_volume(run, dst_ph)
        self.vol_src = load_pack_volume(run, src_ph)
        scale = self.vol_dst.scale

        self.layer_ct.data = self.vol_dst.data
        self.layer_ct.scale = scale
        self.layer_ct.name = f"CT {dst_ph} (target)"
        self.layer_src.data = self.vol_src.data
        self.layer_src.scale = scale
        self.layer_src.name = f"CT {src_ph} (source)"

        for lyr in (
            self.layer_truth,
            self.layer_pred,
            self.layer_src_pts,
            self.layer_rings,
            self.layer_err,
            self.layer_dvf_mag,
            self.layer_dvf_comp,
            self.layer_warped,
            self.layer_diff,
            self.layer_ident_diff,
            self.layer_dvf_arrows,
        ):
            try:
                lyr.scale = scale
            except Exception:
                pass

        z0 = np.zeros_like(self.vol_dst.data, dtype=np.float32)
        for lyr in (
            self.layer_dvf_mag,
            self.layer_dvf_comp,
            self.layer_warped,
            self.layer_diff,
            self.layer_ident_diff,
        ):
            try:
                lyr.data = z0
            except Exception:
                pass

        self._sync_field_choices()
        self._update_case_tre_label()
        self._apply_thick_slices()
        self._refresh_landmarks()
        self._refresh_dvf_overlays()
        self._refresh_drr_panel()
        self._apply_inplane_ornament()
        self.viewer.status = (
            f"Switched → {run.arm} C{run.case:02d}  field={self.field}"
        )

    # ---- dock lifecycle ----------------------------------------------
    def _register_dock(
        self,
        *,
        name: str,
        builder=None,
        area: str = "right",
        min_w: int | None = None,
        min_h: int | None = None,
    ):
        """(Re)create a dock for ``name``, rebuilding its content if it died."""
        from qtpy.QtWidgets import QDockWidget, QSizePolicy

        if builder is not None:
            self._panel_specs[name] = {
                "builder": builder,
                "area": area,
                "min_w": min_w,
                "min_h": min_h,
                "widget": None,
            }
        spec = self._panel_specs.get(name)
        if spec is None:
            return None
        min_w, min_h = spec.get("min_w"), spec.get("min_h")

        widget, rebuilt = self._panel_widget(name)
        if widget is None:
            return None

        # Normalize magicgui → QWidget for docking
        dock_widget = widget
        native = None
        try:
            native = getattr(widget, "native", None)
        except RuntimeError:
            return None
        if native is not None and self._qt_alive(native):
            dock_widget = native

        if min_w:
            try:
                dock_widget.setMinimumWidth(min_w)
            except Exception:
                pass
        if min_h:
            try:
                dock_widget.setMinimumHeight(min_h)
            except Exception:
                pass
        try:
            dock_widget.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
            )
        except Exception:
            pass

        self._drop_dock(name)
        dock = self.viewer.window.add_dock_widget(
            dock_widget,
            name=name,
            area=spec["area"],
            allowed_areas=["left", "right", "bottom", "top"],
            tabify=False,
        )
        # Napari's custom title bar always draws an X, whatever the dock
        # features are, and it calls destroyOnClose() → remove_dock_widget()
        # → deleteLater(). Redirect it to a plain hide so the panel survives;
        # the recreate path above covers destroys we don't control.
        try:
            dock.destroyOnClose = lambda *_a, _n=name: self.hide_panel(_n)
        except Exception:
            pass
        try:
            # Keep Closable: napari's title-bar "hide" button calls close(),
            # which Qt silently ignores on a non-closable QDockWidget.
            dock.setFeatures(
                QDockWidget.DockWidgetFeature.DockWidgetMovable
                | QDockWidget.DockWidgetFeature.DockWidgetFloatable
                | QDockWidget.DockWidgetFeature.DockWidgetClosable
            )
        except Exception:
            pass
        if min_w:
            try:
                dock.setMinimumWidth(min_w)
            except Exception:
                pass
        if min_h:
            try:
                dock.setMinimumHeight(min_h)
            except Exception:
                pass
        # remove_dock_widget() re-parents the content to None, which marks it
        # explicitly hidden; without this the re-docked panel comes back blank.
        try:
            dock_widget.show()
        except Exception:
            pass
        self._docks[name] = dock
        # Keep Python refs so GC (and thus the C++ objects) can't drop them
        spec["dock_widget"] = dock_widget
        if rebuilt:
            self._repopulate_panel(name)
        return dock

    def _panel_widget(self, name: str):
        """Live content widget for a panel; rebuild via its builder if dead.

        Returns ``(widget, rebuilt)``; ``widget`` is None if the build failed.
        """
        spec = self._panel_specs[name]
        widget = spec.get("widget")
        if self._qt_alive(widget):
            return widget, False
        self._release_widget(widget)
        widget = spec["builder"]()
        spec["widget"] = widget
        return widget, widget is not None

    def _release_widget(self, widget) -> None:
        """Unhook napari's layer-event callbacks from a dead panel widget."""
        if widget is None:
            return
        try:
            reset = getattr(widget, "reset_choices", None)
        except Exception:  # noqa: BLE001 — dead wrapper
            return
        if reset is None:
            return
        events = self.viewer.layers.events
        for evt in ("inserted", "removed", "reordered", "renamed"):
            try:
                getattr(events, evt).disconnect(reset)
            except Exception:
                pass

    def _drop_dock(self, name: str) -> None:
        """Forget (and if still alive, remove) the dock registered as ``name``."""
        dock = self._docks.pop(name, None)
        if self._qt_alive(dock):
            try:
                self.viewer.window.remove_dock_widget(dock)
            except Exception:
                pass
        self._purge_stale_napari_docks()

    def _purge_stale_napari_docks(self) -> None:
        """Drop deleted QtViewerDockWidget refs napari still keeps keyed by name."""
        registry = getattr(self.viewer.window, "_wrapped_dock_widgets", None)
        if registry is None:
            return
        for key, dock in list(registry.items()):
            if not self._qt_alive(dock):
                registry.pop(key, None)

    def _repopulate_panel(self, name: str) -> None:
        """Refill a freshly rebuilt panel from current state."""
        if not hasattr(self, "pl"):
            return
        if name == "TRE":
            self._refresh_landmarks()
            self._append_warp_summary()
        elif name == "DRR":
            self._refresh_drr_panel()

    def show_panel(self, name: str) -> None:
        if name not in self._panel_specs:
            self.viewer.status = f"No panel named {name!r}"
            return
        dock = self._docks.get(name)
        if not self._dock_usable(dock, name):
            dock = self._register_dock(name=name)
        if not self._qt_alive(dock):
            self.viewer.status = f"Failed to recreate panel '{name}'"
            return
        try:
            dock.setVisible(True)
            dock.show()
            dock.raise_()
        except RuntimeError:
            # Died between the liveness check and here — rebuild once.
            dock = self._register_dock(name=name)
            if not self._qt_alive(dock):
                self.viewer.status = f"Failed to recreate panel '{name}'"
                return
            dock.show()
            dock.raise_()
        self.viewer.status = (
            f"Panel '{name}' shown — P = show all panels, "
            f"Window→TRE Panels to hide/show, drag title/edges to move & resize"
        )

    def _dock_usable(self, dock, name: str) -> bool:
        """A dock is usable only if it *and* its content widget are alive."""
        if not self._qt_alive(dock):
            return False
        if not self._qt_alive(self._panel_specs[name].get("widget")):
            return False
        try:
            inner = dock.widget()
        except RuntimeError:
            return False
        return self._qt_alive(inner)

    def show_all_panels(self) -> None:
        for name in list(self._panel_specs.keys()):
            try:
                self.show_panel(name)
            except Exception as exc:  # noqa: BLE001 — never let P kill the GUI
                self.viewer.status = f"Could not show {name}: {exc}"

    def hide_panel(self, name: str) -> None:
        dock = self._docks.get(name)
        if self._qt_alive(dock):
            try:
                dock.hide()
            except RuntimeError:
                self._docks.pop(name, None)
        self.viewer.status = f"Panel '{name}' hidden — press P to bring panels back"

    def toggle_fullscreen(self) -> None:
        """Enter/leave true OS fullscreen (fills the monitor, chrome aside).

        Napari's built-in View→Toggle Full Screen is Ctrl+F11; we own plain F11.
        """
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QApplication

        qt = self.viewer.window._qt_window
        if qt.isFullScreen():
            # Restore to maximized so the canvas stays wide, not a floating box.
            qt.setWindowState(Qt.WindowState.WindowMaximized)
            qt.showMaximized()
            self.viewer.status = "Maximized (F11 = fullscreen)"
            return

        # Some WMs ignore showFullScreen() while still maximized; clear first.
        qt.setWindowState(Qt.WindowState.WindowNoState)
        qt.showFullScreen()
        # Stubborn compositors: pin geometry to the current screen explicitly.
        screen = qt.screen() or QApplication.primaryScreen()
        if screen is not None and not qt.isFullScreen():
            qt.setGeometry(screen.geometry())
            qt.showFullScreen()
        qt.raise_()
        qt.activateWindow()
        self.viewer.status = (
            "Fullscreen (F11 = exit to maximized)  │  "
            f"{qt.width()}×{qt.height()}  isFullScreen={qt.isFullScreen()}"
        )

    def _force_wide_window(self) -> None:
        """Maximize (or at least span the available screen) after show()."""
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QApplication

        qt = self.viewer.window._qt_window
        if qt.isFullScreen():
            return
        screen = qt.screen() or QApplication.primaryScreen()
        if screen is not None:
            # Prefer maximize; if the WM leaves us windowed, fall back to
            # filling availableGeometry so the canvas is actually wide.
            avail = screen.availableGeometry()
            qt.setGeometry(avail)
        qt.setWindowState(Qt.WindowState.WindowMaximized)
        qt.showMaximized()
        qt.raise_()

    def _setup_window_chrome(self, *, show: bool) -> None:
        """Maximize, F11 fullscreen, Window→TRE Panels to reopen closed docks."""
        from qtpy.QtCore import Qt, QTimer
        from qtpy.QtGui import QAction, QKeySequence

        qt = self.viewer.window._qt_window
        # Maximize AFTER the event loop starts — calling showMaximized() during
        # Viewer.__init__ is often ignored (napari then restores a saved small
        # window_size from settings).
        if show:
            self._force_wide_window()
            QTimer.singleShot(0, self._force_wide_window)
            QTimer.singleShot(200, self._force_wide_window)

        try:
            win_menu = self.viewer.window.window_menu
            panels = win_menu.addMenu("TRE Panels")
            names = list(self._panel_specs.keys())
            act_all = QAction("Show all panels", qt)
            act_all.setShortcut(QKeySequence("P"))
            # ApplicationShortcut so P/F11 work even when a dock has focus.
            act_all.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            act_all.triggered.connect(lambda *_a: self.show_all_panels())
            panels.addAction(act_all)
            panels.addSeparator()
            for name in names:
                act = QAction(f"Show {name}", qt)

                def _make(n: str = name):
                    return lambda *_a: self.show_panel(n)

                act.triggered.connect(_make())
                panels.addAction(act)
            panels.addSeparator()
            for name in names:
                act_h = QAction(f"Hide {name}", qt)

                def _make_hide(n: str = name):
                    return lambda *_a: self.hide_panel(n)

                act_h.triggered.connect(_make_hide())
                panels.addAction(act_h)
            self._panels_menu = panels
            act_fs = QAction("Toggle Fullscreen", qt)
            act_fs.setShortcut(QKeySequence("F11"))
            act_fs.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            act_fs.triggered.connect(lambda *_a: self.toggle_fullscreen())
            win_menu.addAction(act_fs)
        except Exception as exc:  # noqa: BLE001
            self.viewer.status = f"menu setup: {exc}"

        # Do NOT also bind_key("F11") / bind_key("P"): QAction shortcuts already
        # own those keys. A second handler toggles fullscreen twice → no-op.

    def _apply_inplane_ornament(self) -> None:
        """Apply 0/90/180/270° in-plane rotation for *any* orientation.

        Camera *roll* only looks correct on axial; on coronal/sagittal it
        collapses to ~two poses. Negative ``layer.scale`` also fails as a flip:
        it mirrors around data index 0 (not the FOV centre), so 180°/270° land
        off-camera and look like 0°/90°.

        Instead:
          • even/odd ``k`` → swap the two displayed axes (transpose = ±90°)
          • flips → napari ``camera.orientation2d`` (proven view flip API)
          • spacing stays strictly positive
        """
        base = list(_ORIENT[getattr(self, "_orient_name", "axial")])
        slider, a, b = int(base[0]), int(base[1]), int(base[2])
        k = int(self._rot90_k) % 4
        # k=0: (a,b); k=1: (b,a)+flip row; k=2: (a,b)+flip both; k=3: (b,a)+flip col
        if k % 2 == 0:
            row_ax, col_ax = a, b
        else:
            row_ax, col_ax = b, a
        flip_row = k in (1, 2)  # 90° and 180°
        flip_col = k in (2, 3)  # 180° and 270°
        if self._flip_ud:
            flip_row = not flip_row
        if self._flip_lr:
            flip_col = not flip_col

        self.viewer.dims.order = (slider, row_ax, col_ax)

        # Always positive spacing — flips are a camera concern, not a scale sign.
        sp = np.abs(np.asarray(self.vol_dst.scale, dtype=float))
        scale = (float(sp[0]), float(sp[1]), float(sp[2]))
        for lyr in self.viewer.layers:
            try:
                lyr.scale = scale
            except Exception:
                pass

        # Default napari 2D orientation is (down, right). Toggle axes to flip.
        vert = "up" if flip_row else "down"
        horiz = "left" if flip_col else "right"
        try:
            cam = getattr(getattr(self.viewer, "scene", None), "camera", None)
            cam = cam or self.viewer.camera
            if hasattr(cam, "orientation2d"):
                cam.orientation2d = (vert, horiz)
            else:
                # Very old napari: fall back to roll (axial-only approximation)
                cam.angles = (0.0, 0.0, float((k * 90) % 360))
        except Exception:
            pass

        # Re-frame: order/orientation changes leave the old camera centre behind.
        try:
            self.viewer.reset_view()
        except Exception:
            pass

        deg = (k * 90) % 360
        self.viewer.status = (
            f"{self._orient_name}  in-plane {deg}°  "
            f"flipLR={self._flip_lr} flipUD={self._flip_ud}  "
            f"view=({vert},{horiz})  "
            f"| R=+90° Shift+R=+180° F/Shift+F=flip 0=reset  "
            f"P=panels F11=fullscreen"
        )

    def rotate_inplane(self, degrees: int) -> None:
        steps = int(round(degrees / 90.0)) % 4
        self._rot90_k = (self._rot90_k + steps) % 4
        self._apply_inplane_ornament()

    def flip_inplane(self, *, lr: bool = False, ud: bool = False) -> None:
        if lr:
            self._flip_lr = not self._flip_lr
        if ud:
            self._flip_ud = not self._flip_ud
        self._apply_inplane_ornament()

    def reset_inplane(self) -> None:
        self._rot90_k = 0
        self._flip_lr = False
        self._flip_ud = False
        self._apply_inplane_ornament()

    def _build_drr_panel(self):
        """Matplotlib DRR scrub + RTK-projected truth/pred landmarks."""
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QLabel, QSlider, QVBoxLayout, QWidget

        from tre_viewer.data import model_training_dir
        from tre_viewer.drr import resolve_geometry

        # Keep the current view/mode across a rebuild
        self._drr_view = int(getattr(self, "_drr_view", 0))
        self._drr_mode = getattr(self, "_drr_mode", "target")  # target|source|diff
        try:
            self._geom = resolve_geometry(self.run)
            self._mt = model_training_dir(self.run.run_root, self.run.scan_id)
        except Exception as exc:  # noqa: BLE001
            self._geom = None
            self._mt = None
            self.viewer.status = f"DRR panel unavailable: {exc}"
            return None

        fig = Figure(figsize=(5.5, 5.0), tight_layout=True)
        self._drr_ax = fig.add_subplot(111)
        self._drr_canvas = FigureCanvasQTAgg(fig)
        self._drr_canvas.setMinimumHeight(320)
        self._drr_im = None
        self._drr_sc_truth = None
        self._drr_sc_pred = None

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(max(0, self._geom.n_views - 1))
        slider.setValue(min(self._drr_view, max(0, self._geom.n_views - 1)))
        slider.valueChanged.connect(self._on_drr_slider)
        self._drr_slider = slider
        self._drr_meta = QLabel("")
        self._drr_meta.setWordWrap(True)
        self._drr_meta.setStyleSheet("font-family: monospace; font-size: 11px;")

        wrap = QWidget()
        wrap.setMinimumHeight(380)
        lay = QVBoxLayout(wrap)
        lay.addWidget(
            QLabel("DRR + RTK landmarks  │  D=cycle tgt/src/diff  │  drag splitter to resize")
        )
        lay.addWidget(self._drr_canvas, stretch=1)
        lay.addWidget(slider)
        lay.addWidget(self._drr_meta)
        # Bottom pane = LEARN-style multipane (CT above, DRR below)
        self._drr_panel = wrap
        return wrap

    def _build_drr_dock(self) -> None:
        """Register the bottom DRR pane + its keybinding (skipped if no geometry)."""
        dock = self._register_dock(
            name="DRR",
            builder=self._build_drr_panel,
            area="bottom",
            min_h=380,
            min_w=420,
        )
        if dock is None:
            self._panel_specs.pop("DRR", None)
            return

        @self.viewer.bind_key("d")
        def _cycle_drr(viewer):  # noqa: ARG001
            modes = ["target", "source", "diff"]
            i = modes.index(self._drr_mode)
            self._drr_mode = modes[(i + 1) % len(modes)]
            self._refresh_drr_panel()

    def _on_drr_slider(self, value: int) -> None:
        self._drr_view = int(value)
        self._refresh_drr_panel()

    def _refresh_drr_panel(self) -> None:
        if getattr(self, "_geom", None) is None or getattr(self, "_mt", None) is None:
            return
        if not hasattr(self, "pl"):
            return
        # Panel widgets die with the dock; skip until show_panel('DRR') rebuilds
        if not self._qt_alive(getattr(self, "_drr_canvas", None)):
            return
        from tre_viewer.drr import (
            load_proj_128,
            proj_path,
            project_landmarks_to_128,
            r3_ct_physical_landmarks,
        )

        vi = int(self._drr_view)
        view_1 = vi + 1
        src_ph, dst_ph = pair_phases(self.pair)
        try:
            tgt = load_proj_128(proj_path(self._mt, dst_ph, view_1, source=False))
            src = load_proj_128(proj_path(self._mt, dst_ph, view_1, source=True))
        except Exception as exc:  # noqa: BLE001
            self._drr_meta.setText(f"proj load failed: {exc}")
            return

        if self._drr_mode == "source":
            img = src
            title = f"source T50  view {view_1}"
        elif self._drr_mode == "diff":
            img = tgt - src
            title = f"tgt−src  view {view_1}"
        else:
            img = tgt
            title = f"target {dst_ph}  view {view_1}"

        # Project truth (dst) and pred in pack → R3 physical → RTK → 128
        truth_mm, _ = r3_ct_physical_landmarks(self.run, self.pl.truth_pack)
        pred_mm, _ = r3_ct_physical_landmarks(self.run, self.pl.pred_pack)
        M = self._geom.matrices[vi]
        truth_uv = project_landmarks_to_128(truth_mm, M)
        pred_uv = project_landmarks_to_128(pred_mm, M)
        # 2D residual on detector (pixels)
        d2 = np.linalg.norm(pred_uv - truth_uv, axis=1)
        angle = float(self._geom.gantry_deg[vi])

        ax = self._drr_ax
        ax.clear()
        if self._drr_mode == "diff":
            lim = float(np.percentile(np.abs(img), 99)) or 1.0
            ax.imshow(img, cmap="coolwarm", vmin=-lim, vmax=lim, origin="upper")
        else:
            ax.imshow(img, cmap="gray", origin="upper")
        ax.scatter(
            truth_uv[:, 0],
            truth_uv[:, 1],
            s=18,
            c="lime",
            marker="o",
            label="truth",
            linewidths=0.3,
            edgecolors="k",
        )
        ax.scatter(
            pred_uv[:, 0],
            pred_uv[:, 1],
            s=18,
            c="magenta",
            marker="x",
            label="pred",
        )
        # error ticks
        for t, p in zip(truth_uv, pred_uv):
            ax.plot([t[0], p[0]], [t[1], p[1]], color="yellow", lw=0.4, alpha=0.7)
        ax.set_xlim(0, 127)
        ax.set_ylim(127, 0)
        ax.set_title(title, fontsize=9)
        ax.legend(loc="upper right", fontsize=7)
        self._drr_canvas.draw_idle()

        inside = (
            (truth_uv[:, 0] >= 0)
            & (truth_uv[:, 0] < 128)
            & (truth_uv[:, 1] >= 0)
            & (truth_uv[:, 1] < 128)
        ).mean()
        self._drr_meta.setText(
            f"angle {angle:.2f}°  geom OffsetY={self._geom.offset_y:g}  "
            f"2D TRE mean {float(d2.mean()):.2f} px  "
            f"max {float(d2.max()):.2f}  insideFOV {inside:.0%}\n"
            f"{self._geom.path.name}"
        )

    def _apply_thick_slices(self) -> None:
        """±N mm margin on every axis so nearby landmarks stay visible."""
        # ~3 pack slices along SI (dz≈2.5) and a few mm in-plane
        scale = np.asarray(self.vol_dst.scale, dtype=float)
        margin = np.maximum(3.0 * scale, 8.0)  # mm
        self.viewer.dims.margin_left = tuple(float(m) for m in margin)
        self.viewer.dims.margin_right = tuple(float(m) for m in margin)

    def _set_step_data(self, axis: int, index: float) -> None:
        """Move slider using *data* indices (not world mm)."""
        self.viewer.dims.set_current_step(axis, int(round(index)))

    def _reload_volumes_if_needed(self) -> None:
        src_ph, dst_ph = pair_phases(self.pair)
        if self.vol_dst.phase != dst_ph:
            self.vol_dst = load_pack_volume(self.run, dst_ph)
            self.layer_ct.data = self.vol_dst.data
            self.layer_ct.name = f"CT {dst_ph} (target)"
        if self.vol_src.phase != src_ph:
            self.vol_src = load_pack_volume(self.run, src_ph)
            self.layer_src.data = self.vol_src.data
            self.layer_src.name = f"CT {src_ph} (source)"

    def _set_orient(self, name: str) -> None:
        self._orient_name = name
        self._apply_thick_slices()
        # Re-apply in-plane 0/90/180/270 for this plane (all four work on coronal too)
        self._apply_inplane_ornament()
        labels = {
            "axial": "axial │ slider=z (SI)",
            "coronal": "coronal │ slider=y (AP)",
            "sagittal": "sagittal │ slider=x (LR)",
        }
        deg = (int(self._rot90_k) * 90) % 360
        self.viewer.status = (
            f"{labels.get(name, name)}  in-plane {deg}°  │  "
            f"{self.run.arm} C{self.run.case:02d} frame={self.run.frame}  │  "
            f"R=+90° on this plane"
        )

    def _add_handedness_overlay(self) -> None:
        # Lightweight text via status; full canvas burn-in deferred to Phase 4
        pass

    def _refresh_landmarks(self) -> None:
        pl = per_landmark(
            self.run,
            self.field,
            which=self.which,  # type: ignore[arg-type]
            pair=self.pair,
        )
        self.pl = pl
        (nx, ny, nz), spacing = CASE_INFO[self.run.case]

        truth_zyx = xyz_to_zyx(pl.truth_pack)
        pred_zyx = xyz_to_zyx(pl.pred_pack)
        src_zyx = xyz_to_zyx(pl.src_pack)
        colors = tre_to_rgba(pl.tre_mm)

        self.layer_truth.data = truth_zyx
        self.layer_truth.face_color = colors
        self.layer_truth.size = 12.0
        self.layer_truth.features = {"tre_mm": pl.tre_mm, "id": np.arange(len(pl.tre_mm))}

        self.layer_pred.data = pred_zyx
        self.layer_pred.face_color = "magenta"
        self.layer_pred.size = 10.0

        self.layer_src_pts.data = src_zyx
        self.layer_src_pts.size = 8.0

        self.layer_rings.data = truth_zyx
        # Rings in data voxels ≈ TRE_mm / mean in-plane spacing, floored for visibility
        mean_inplane = 0.5 * (spacing[0] + spacing[1])
        ring_vox = np.maximum(pl.tre_mm / max(mean_inplane, 1e-6) * 1.5, 10.0)
        self.layer_rings.size = ring_vox
        self.layer_rings.border_color = colors
        self.layer_rings.face_color = np.zeros((len(pl.tre_mm), 4), dtype=float)

        vecs = error_vectors_zyx(pl.truth_pack, pl.pred_pack, spacing)
        self.layer_err.data = vecs
        self.layer_err.edge_color = colors
        self.layer_err.edge_width = 1.5

        self._set_label(
            "summary_label",
            summary_line(pl)
            + f"\nfield={self.field}  oob={int(pl.oob_mask.sum())}  "
            f"display=pack-mm  DVF_frame={pl.frame}\n"
            f"lime=truth  magenta=pred  rings∝TRE  |  W=worst  Space=blink",
        )
        self.viewer.title = (
            f"TRE Viewer │ {self.run.arm} C{self.run.case:02d} │ "
            f"{self.field} │ mean {pl.registered_stats['mean']:.2f} mm"
        )

        self._worst_rows = worst_table(pl, k=15)
        if self._qt_alive(getattr(self, "worst_list", None)):
            self.worst_list.clear()
            for row in self._worst_rows:
                self.worst_list.addItem(
                    f"#{row['id']:02d}  TRE {row['tre_mm']:.2f}  "
                    f"idnt {row['ident_mm']:.2f}  z={row['slice_z']}"
                )

    def _set_label(self, attr: str, text: str) -> None:
        """Set a panel label's text, tolerating a panel that was destroyed."""
        label = getattr(self, attr, None)
        if self._qt_alive(label):
            label.setText(text)

    def _on_worst_clicked(self, item) -> None:
        row = self.worst_list.row(item)
        if 0 <= row < len(self._worst_rows):
            self.jump_to_landmark(self._worst_rows[row]["id"])

    def jump_worst(self, *, by_identity: bool = False) -> None:
        if not hasattr(self, "pl"):
            return
        key = self.pl.identity_mm if by_identity else self.pl.tre_mm
        order = np.argsort(-key)
        idx = int(order[self._worst_cycle % len(order)])
        self._worst_cycle += 1
        self.jump_to_landmark(idx)

    def _refresh_dvf_overlays(self) -> None:
        """Recompute warp/DVF pack overlays for the current field/pair."""
        try:
            bundle = field_warp_bundle(self.run, self.field, self.pair)
        except Exception as exc:  # noqa: BLE001 — show in UI, don't crash viewer
            self.viewer.status = f"DVF overlay unavailable: {exc}"
            self._bundle = None
            return
        self._bundle = bundle
        pk = bundle["pack"]
        self.layer_warped.data = pk["warped"]
        self.layer_diff.data = pk["diff"]
        self.layer_ident_diff.data = pk["ident_diff"]
        self.layer_dvf_mag.data = pk["mag"]

        # Symmetric HU window for diffs
        for lyr in (self.layer_diff, self.layer_ident_diff):
            lim = float(np.percentile(np.abs(lyr.data), 99)) if lyr.data.size else 200.0
            lim = max(lim, 50.0)
            lyr.contrast_limits = (-lim, lim)

        self.layer_dvf_mag.contrast_limits = (0.0, max(float(pk["mag"].max()), 1.0))

        comp = getattr(self, "_dvf_component", "off")
        if comp == "off":
            self.layer_dvf_mag.visible = False
            self.layer_dvf_comp.visible = False
        elif comp == "mag":
            self.layer_dvf_mag.visible = True
            self.layer_dvf_comp.visible = False
        else:
            self.layer_dvf_mag.visible = False
            self.layer_dvf_comp.visible = True
            data = pk[comp]
            self.layer_dvf_comp.data = data
            lim = float(np.percentile(np.abs(data), 99)) if data.size else 5.0
            lim = max(lim, 1.0)
            self.layer_dvf_comp.contrast_limits = (-lim, lim)
            self.layer_dvf_comp.name = f"DVF {comp} mm"

        # Arrows from upsampled DVF on current slice
        dvf_pack = upsample_dvf_to_pack(bundle["dvf"], self.vol_dst.data.shape)
        order = self.viewer.dims.order
        slider_axis = int(order[0])
        slice_index = int(self.viewer.dims.current_step[slider_axis])
        arrows = decimated_arrows_zyx(
            dvf_pack,
            step=getattr(self, "_arrow_step", 6),
            slice_axis=slider_axis,
            slice_index=slice_index,
            gain=getattr(self, "_arrow_gain", 2.0),
        )
        self.layer_dvf_arrows.data = arrows

        self._append_warp_summary()
        mae_w = bundle["mae_warped_lung"]
        mae_i = bundle["mae_ident_lung"]
        self.viewer.status = (
            f"DVF overlays ready │ lung MAE warped {mae_w:.1f} vs ident {mae_i:.1f} HU │ "
            f"Space=blink"
        )

    def _append_warp_summary(self) -> None:
        """Append the warp-vs-identity lung MAE line to the summary panel."""
        bundle = getattr(self, "_bundle", None)
        if bundle is None or not self._qt_alive(getattr(self, "summary_label", None)):
            return
        mae_w = bundle["mae_warped_lung"]
        mae_i = bundle["mae_ident_lung"]
        extra = (
            f"\nwarp lung MAE {mae_w:.1f} HU  identity {mae_i:.1f} HU  "
            f"Δ {mae_i - mae_w:+.1f}  (+disp image warp)"
        )
        self.summary_label.setText(self.summary_label.text().split("\nwarp")[0] + extra)

    def blink_target_warped(self) -> None:
        """Spacebar: alternate target CT ↔ warped source."""
        if self._bundle is None:
            self._refresh_dvf_overlays()
        if self._bundle is None:
            return
        self._blink_warped = not self._blink_warped
        if self._blink_warped:
            self.layer_warped.visible = True
            self.layer_ct.visible = False
            self.viewer.status = "BLINK: warped source (Space again → target)"
        else:
            self.layer_warped.visible = False
            self.layer_ct.visible = True
            self.viewer.status = "BLINK: target CT (Space again → warped)"

    def jump_to_landmark(self, idx: int) -> None:
        pl = self.pl
        z, y, x = xyz_to_zyx(pl.truth_pack[idx : idx + 1])[0]
        # Slider axis is dims.order[0]; step must be *data index*, not mm
        # (set_point uses world mm when scale≠1 — that was hiding the points).
        order = self.viewer.dims.order
        slider_axis = int(order[0])
        coord = (float(z), float(y), float(x))[slider_axis]
        self._set_step_data(slider_axis, coord)
        err = pl.err_vec_mm[idx]
        self._set_label(
            "selected_label",
            f"selected #{idx}  TRE {pl.tre_mm[idx]:.2f} mm  "
            f"err(LR,AP,SI≈x,y,z)_mm=({err[0]:+.2f},{err[1]:+.2f},{err[2]:+.2f})  "
            f"pack_xyz=({pl.truth_pack[idx,0]:.1f},{pl.truth_pack[idx,1]:.1f},"
            f"{pl.truth_pack[idx,2]:.1f})",
        )
        self.layer_truth.selected_data = {idx}
        self.viewer.status = (
            f"landmark #{idx} @ data zyx=({z:.0f},{y:.0f},{x:.0f})  "
            f"TRE={pl.tre_mm[idx]:.2f} mm"
        )


def launch(
    *,
    arm: str = "A1",
    case: int = 1,
    field: str | None = None,
    which: str = "75",
    pair: str = "T00_T50",
    runs_dir: str | Path | None = None,
    show: bool = True,
) -> TreViewerApp:
    run = resolve_run(arm, case, runs_dir=runs_dir)
    app = TreViewerApp(
        run,
        field=field,
        which=which,
        pair=pair,  # type: ignore[arg-type]
        show=show,
    )
    return app


def main_view(
    *,
    arm: str = "A1",
    case: int = 1,
    field: str | None = None,
    which: str = "75",
    pair: str = "T00_T50",
    runs_dir: str | Path | None = None,
) -> int:
    import napari

    launch(
        arm=arm,
        case=case,
        field=field,
        which=which,
        pair=pair,
        runs_dir=runs_dir,
        show=True,
    )
    napari.run()
    return 0


def smoke_test(arm: str = "A1", case: int = 1) -> dict:
    """Build the viewer off-screen and return summary stats (no event loop)."""
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = launch(arm=arm, case=case, show=False)
    pl = app.pl
    b = app._bundle
    out = {
        "case": app.run.case,
        "field": app.field,
        "frame": app.run.frame,
        "mean_tre": float(pl.registered_stats["mean"]),
        "identity": float(pl.identity_stats["mean"]),
        "n_truth": int(len(app.layer_truth.data)),
        "n_vectors": int(len(app.layer_err.data)),
        "ct_shape": tuple(app.layer_ct.data.shape),
        "scale": tuple(float(x) for x in app.layer_ct.scale),
    }
    if b is not None:
        out["mae_warped_lung"] = b["mae_warped_lung"]
        out["mae_ident_lung"] = b["mae_ident_lung"]
        out["warp_improves"] = b["mae_warped_lung"] < b["mae_ident_lung"] - 1.0
        out["dvf_mag_max_mm"] = float(b["comps"]["mag"].max())
    return out


def panel_smoke_test(arm: str = "A1", case: int = 1) -> dict:
    """Offscreen regression test for the dock close → reopen (P) path.

    Reproduces every way a panel can go away — napari's title-bar X
    (``destroyOnClose``), ``remove_dock_widget`` + ``deleteLater`` (which kills
    the QtViewerDockWidget wrapper), and a destroyed *content* widget — then
    asserts ``show_panel`` / ``show_all_panels`` bring it back without a
    RuntimeError and with live, repopulated contents.
    """
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtCore import QCoreApplication, QEvent

    def flush() -> None:
        QCoreApplication.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        QCoreApplication.processEvents()

    app = launch(arm=arm, case=case, show=False)
    flush()
    out: dict = {"panels": list(app._panel_specs)}
    assert "TRE" in app._panel_specs, "TRE panel was never registered"

    # The main window is never shown (an offscreen GL context aborts), so use
    # isHidden(): children of a not-yet-shown parent are "not visible, not
    # hidden", and only an explicit hide()/setParent(None) flips it.
    def alive_and_shown(name: str) -> None:
        dock = app._docks.get(name)
        assert app._qt_alive(dock), f"{name}: dock dead after show_panel"
        assert not dock.isHidden(), f"{name}: dock still hidden after show_panel"
        inner = dock.widget()
        assert app._qt_alive(inner), f"{name}: content dead after show_panel"
        assert not inner.isHidden(), f"{name}: content hidden after show_panel"

    # 1) napari's title-bar X (destroyOnClose) must hide, not destroy. Click the
    #    real button so the parent() → destroyOnClose lookup is exercised too.
    dock = app._docks["TRE"]
    dock.title.close_button.click()
    flush()
    assert app._qt_alive(dock), "title-bar X destroyed the dock"
    assert dock.isHidden(), "title-bar X did not hide the dock"
    app.show_panel("TRE")
    alive_and_shown("TRE")
    out["x_close_hides_and_reopens"] = True

    # 1b) napari's title-bar hide button calls close(), which Qt drops on a
    #     non-closable dock — so DockWidgetClosable must stay on.
    dock.title.hide_button.click()
    flush()
    assert dock.isHidden(), "title-bar hide button did not hide the dock"
    app.show_panel("TRE")
    alive_and_shown("TRE")
    out["hide_button_works"] = True

    # 2) Hard destroy of the dock wrapper (this is what raised RuntimeError).
    dock = app._docks["TRE"]
    widget = app._panel_specs["TRE"]["widget"]
    app.viewer.window.remove_dock_widget(dock)
    flush()
    assert not app._qt_alive(dock), "_qt_alive reports a deleted dock as alive"
    app.show_panel("TRE")
    alive_and_shown("TRE")
    assert app._panel_specs["TRE"]["widget"] is widget, "content needlessly rebuilt"
    out["dead_dock_recreated"] = True

    # 3) Dock *and* content destroyed → must rebuild the panel from scratch.
    dock = app._docks["TRE"]
    widget = app._panel_specs["TRE"]["widget"]
    old_label = app.summary_label
    app.viewer.window.remove_dock_widget(dock)
    widget.setParent(None)
    widget.deleteLater()
    flush()
    assert not app._qt_alive(widget), "content widget survived deleteLater"
    app.show_panel("TRE")
    alive_and_shown("TRE")
    assert app.summary_label is not old_label, "summary label was not rebuilt"
    assert app.summary_label.text(), "rebuilt summary label is empty"
    assert app.worst_list.count() > 0, "rebuilt worst-landmark list is empty"
    out["dead_content_rebuilt"] = True

    # 4) Same for the magicgui Controls panel, and its callback must still run.
    dock = app._docks["Controls"]
    old_controls = app.controls
    app.viewer.window.remove_dock_widget(dock)
    old_controls.native.setParent(None)
    old_controls.native.deleteLater()
    flush()
    assert not app._qt_alive(old_controls), "_qt_alive missed a dead FunctionGui"
    app.show_panel("Controls")
    alive_and_shown("Controls")
    assert app.controls is not old_controls, "Controls panel was not rebuilt"
    app.controls()  # the "Reload" button path, on freshly built widgets
    out["controls_rebuilt_and_callable"] = True

    if "DRR" in app._panel_specs:
        dock = app._docks["DRR"]
        widget = app._panel_specs["DRR"]["widget"]
        app.viewer.window.remove_dock_widget(dock)
        widget.setParent(None)
        widget.deleteLater()
        flush()
        app._refresh_drr_panel()  # must no-op, not raise, while the panel is dead
        app.show_panel("DRR")
        alive_and_shown("DRR")
        assert app._drr_meta.text(), "rebuilt DRR panel was not repopulated"
        out["drr_rebuilt"] = True

    # 5) P / Window→TRE Panels→Show all, from a fully closed state.
    for name in list(app._panel_specs):
        app.viewer.window.remove_dock_widget(app._docks[name])
    flush()
    action = next(
        a for a in app._panels_menu.actions() if a.text() == "Show all panels"
    )
    assert action.shortcut().toString().upper() == "P", "P shortcut not wired"
    action.trigger()  # exactly what pressing P does
    flush()
    for name in app._panel_specs:
        alive_and_shown(name)
    out["p_key_restores_all"] = True

    # Idempotent, and napari must not be left holding deleted dock refs.
    app.show_all_panels()
    app.hide_panel("TRE")
    assert app._docks["TRE"].isHidden(), "hide_panel did not hide"
    app.show_all_panels()
    for name in app._panel_specs:
        alive_and_shown(name)
    registry = getattr(app.viewer.window, "_wrapped_dock_widgets", {})
    stale = [k for k, d in registry.items() if not app._qt_alive(d)]
    assert not stale, f"napari still holds deleted docks: {stale}"
    out["no_stale_refs"] = True
    out["dock_count"] = len(app._docks)
    return out
