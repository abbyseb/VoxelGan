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
    pack_arrows_zyx,
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
        self._phase_record = None
        self._phase_images = []
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
        initial_landmarks = per_landmark(run, self.field, which=self.which, pair=self.pair)

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
            face_color="green",
            border_width=0.15,
            symbol="disc",
            shading="spherical",
            **_pt,
        )
        self.layer_pred = self.viewer.add_points(
            np.zeros((0, 3)),
            name="pred",
            size=12.0,
            face_color="red",
            border_width=0.15,
            symbol="cross",
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
            visible=False,
            **_pt,
        )
        # User marker styles (napari left panel) — survive reload / case switch.
        self._applying_marker_style = False
        self._marker_style_cache: dict[str, dict] = {
            "truth": {
                "size": 12.0,
                "symbol": "disc",
                "border_width": 0.15,
                "opacity": 1.0,
                "blending": "translucent",
                "shading": "spherical",
                "border_color": "green",
                "face_color": "green",
                "color_by_tre": False,
            },
            "pred": {
                "size": 12.0,
                "symbol": "cross",
                "border_width": 0.15,
                "opacity": 1.0,
                "blending": "translucent",
                "shading": "none",
                "border_color": "red",
                "face_color": "red",
                "color_by_tre": False,
            },
            "src": {
                "size": 8.0,
                "symbol": "o",
                "border_width": 0.15,
                "opacity": 0.5,
                "blending": "translucent",
                "shading": "none",
                "border_color": "white",
                "face_color": "cyan",
                "color_by_tre": False,
            },
            "rings": {
                "size": None,  # always ∝ TRE unless user locks uniform size
                "symbol": "o",
                "border_width": 0.35,
                "opacity": 1.0,
                "blending": "translucent",
                "shading": "none",
                "border_color": None,  # TRE colors when color_by_tre
                "face_color": (0.0, 0.0, 0.0, 0.0),
                "color_by_tre": True,
                "size_by_tre": True,
            },
            "err": {
                "edge_width": 5.0,
                "opacity": 1.0,
                "color_by_tre": False,
                "edge_color": "yellow",
            },
        }
        self.layer_err = self.viewer.add_vectors(
            np.zeros((0, 2, 3)),
            name="error vectors",
            scale=scale,
            edge_width=5.0,
            length=1.0,
            vector_style="arrow",
            edge_color="yellow",
            opacity=1.0,
        )
        self._wire_marker_style_persistence()

        # --- Phase 2: DVF / warp overlays (pack-shaped upsamples; off by default) ---
        self._blink_warped = False
        self._bundle = None
        self._bundle_key = None
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
        self._drr_view = 0
        self._drr_mode = "target"
        self._proj_pages: dict[str, object] = {}

        # --- docks: CT primary; right column tabbed (Cases+TRE / Overlays); DRR bottom ---
        self._docks: dict[str, object] = {}
        self._panel_specs: dict[str, dict] = {}
        self._register_dock(
            name="TRE", builder=self._build_tre_panel, area="right"
        )
        self._register_dock(
            name="Controls",
            builder=self._build_controls_panel,
            area="right",
        )

        self._register_dock(
            name="Phase Performance", builder=self._build_phase_panel, area="right",
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
        def _rot_ccw(viewer):  # noqa: ARG001
            self.rotate_inplane(-90)

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
        self._hijack_napari_rotate_button()
        self._finalize_layout()
        self._set_orient("axial")
        self._refresh_landmarks(initial_landmarks)
        self._refresh_dvf_overlays()
        self.viewer.dims.events.current_step.connect(self._refresh_arrows)
        self.viewer.dims.events.order.connect(self._refresh_arrows)
        self._refresh_projection_views()
        # Land on the worst landmark so points are immediately visible.
        if len(self.pl.tre_mm):
            self.jump_to_landmark(int(np.argmax(self.pl.tre_mm)))
        self._add_handedness_overlay()
        self._apply_inplane_ornament()
        self._sync_rot_widgets()

    # ------------------------------------------------------------------
    def _as_scroll(self, widget):
        """Scrollable dock content — never force the window taller than the screen."""
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QScrollArea

        sa = QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QScrollArea.Shape.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sa.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        sa.setWidget(widget)
        return sa

    def _build_phase_panel(self):
        from .phase_panel import PhasePerformancePanel
        self._phase_panel = PhasePerformancePanel(self)
        return self._phase_panel

    def _finalize_layout(self) -> None:
        """Corners, tabify right panels, sensible sizes — override napari defaults."""
        from qtpy.QtCore import Qt

        qw = getattr(self.viewer.window, "_qt_window", None)
        if qw is None:
            return
        try:
            # Keep the right column full-height; DRR only under the canvas.
            qw.setCorner(
                Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea
            )
            qw.setCorner(
                Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.BottomDockWidgetArea
            )
        except Exception:
            pass

        tre = self._docks.get("TRE")
        ctrl = self._docks.get("Controls")
        if self._qt_alive(tre) and self._qt_alive(ctrl):
            try:
                qw.tabifyDockWidget(tre, ctrl)
                tre.raise_()
            except Exception:
                pass
            try:
                # Prefer a usable TRE list over a giant overlays form.
                qw.resizeDocks([tre, ctrl], [3, 2], Qt.Orientation.Vertical)
            except Exception:
                pass

        phase = self._docks.get("Phase Performance")
        if self._qt_alive(tre) and self._qt_alive(phase):
            qw.tabifyDockWidget(tre, phase)

        drr = self._docks.get("DRR")
        if self._qt_alive(drr):
            try:
                qw.resizeDocks([drr], [220], Qt.Orientation.Horizontal)
            except Exception:
                pass
            try:
                drr.setMaximumHeight(320)
            except Exception:
                pass

        # Soft preferred width for the right column (not a hard minimum).
        for name in ("TRE", "Controls"):
            dock = self._docks.get(name)
            if self._qt_alive(dock):
                try:
                    dock.resize(320, dock.height())
                except Exception:
                    pass

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
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QLabel, QListWidget, QSizePolicy, QVBoxLayout, QWidget

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.summary_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; padding: 4px;"
        )
        self.selected_label = QLabel("selected: —")
        self.selected_label.setWordWrap(True)
        self.selected_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.selected_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; padding: 4px;"
        )
        self.worst_list = QListWidget()
        self.worst_list.setUniformItemSizes(True)
        self.worst_list.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.worst_list.itemClicked.connect(self._on_worst_clicked)

        side = QWidget()
        lay = QVBoxLayout(side)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(4)
        hdr = QLabel("TRE")
        hdr.setStyleSheet("font-weight: 600; font-size: 12px;")
        lay.addWidget(hdr)
        lay.addWidget(self.summary_label)
        lay.addWidget(self.selected_label)
        lay.addWidget(QLabel("Worst landmarks  ·  click or W"))
        lay.addWidget(self.worst_list, stretch=1)
        self._tre_panel = side
        # List scrolls itself — do not wrap in QScrollArea (that expands the list).
        return side

    def _build_controls_panel(self):
        """Cases browser + compact overlay controls (pure Qt — no magicgui bloat)."""
        from qtpy.QtWidgets import (
            QCheckBox,
            QComboBox,
            QDoubleSpinBox,
            QFormLayout,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QLineEdit,
            QPushButton,
            QSpinBox,
            QVBoxLayout,
            QWidget,
        )

        root = QWidget()
        lay = QVBoxLayout(root)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        # --- Cases ---
        cases = QGroupBox("Cases")
        cases_lay = QVBoxLayout(cases)
        cases_lay.setSpacing(4)
        cases_lay.addWidget(QLabel("Runs folder"))
        path_row = QHBoxLayout()
        self._runs_dir_edit = QLineEdit(str(self._runs_dir))
        self._runs_dir_edit.setPlaceholderText("…/arms/…/runs")
        browse = QPushButton("Browse…")
        browse.setFixedWidth(72)
        browse.clicked.connect(self._browse_runs_dir)
        path_row.addWidget(self._runs_dir_edit, stretch=1)
        path_row.addWidget(browse)
        cases_lay.addLayout(path_row)

        scan_btn = QPushButton("Scan folder")
        scan_btn.clicked.connect(self._scan_runs_dir_from_edit)
        self._runs_dir_edit.returnPressed.connect(self._scan_runs_dir_from_edit)
        cases_lay.addWidget(scan_btn)

        cases_lay.addWidget(QLabel("Case / patient"))
        self._case_combo = QComboBox()
        self._case_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self._case_combo.setMinimumContentsLength(12)
        self._case_combo.currentIndexChanged.connect(self._on_case_combo_changed)
        cases_lay.addWidget(self._case_combo)
        self._case_tre_label = QLabel("TRE: —")
        self._case_tre_label.setWordWrap(True)
        self._case_tre_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; padding: 2px;"
        )
        cases_lay.addWidget(self._case_tre_label)
        lay.addWidget(cases)
        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #ffb86c;")
        lay.addWidget(self._error_label)

        # --- Data selectors ---
        data = QGroupBox("Data")
        form = QFormLayout(data)
        form.setContentsMargins(8, 8, 8, 8)
        form.setSpacing(4)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)

        fields = list(self.run.fields_available) or ["identity"]
        if self._phase_record is not None:
            fields.append(self.field)
        self._ctrl_field = QComboBox()
        self._ctrl_field.addItems(fields)
        if self.field in fields:
            self._ctrl_field.setCurrentText(self.field)
        form.addRow("Field", self._ctrl_field)
        self._ctrl_field.setToolTip("identity: no registration; elastix: reference DVF; voxelmap: learned DVF; ckpt: infer from checkpoint")

        self._ctrl_pair = QComboBox()
        self._ctrl_pair.addItems(["T00_T50", "T50_T00"])
        self._ctrl_pair.setCurrentText(self.pair)
        form.addRow("Pair", self._ctrl_pair)
        self._ctrl_pair.setToolTip("Source → target. Reverse landmark TRE is supported; reverse image warping needs an inverse field.")

        self._ctrl_which = QComboBox()
        self._ctrl_which.addItems(["75", "300"])
        self._ctrl_which.setCurrentText(self.which)
        self._ctrl_which.setToolTip("75: primary experiment KPI; 300: pack QA/debugging only")
        form.addRow("Landmarks", self._ctrl_which)

        self._ctrl_orient = QComboBox()
        self._ctrl_orient.addItems(list(_ORIENT.keys()))
        self._ctrl_orient.setCurrentText(self._orient_name)
        form.addRow("Slice plane", self._ctrl_orient)

        self._ctrl_dvf = QComboBox()
        self._ctrl_dvf.addItems(["off", "mag", "SI", "AP", "LR"])
        self._ctrl_dvf.setCurrentText(self._dvf_component)
        form.addRow("Displacement", self._ctrl_dvf)
        self._ctrl_dvf.setToolTip("mag: magnitude; SI: superior/inferior; AP: anterior/posterior; LR: left/right, in mm")
        lay.addWidget(data)
        reload_btn = QPushButton("Apply data / refresh")
        reload_btn.setDefault(True)
        reload_btn.clicked.connect(self._apply_overlay_controls)
        lay.addWidget(reload_btn)
        hint = QLabel("Apply data after changing field, pair or landmark set. Display controls update immediately.")
        hint.setWordWrap(True)
        lay.addWidget(hint)

        # --- Visibility grid ---
        vis = QGroupBox("Overlays")
        grid = QGridLayout(vis)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(4)

        def _cb(text: str, checked: bool) -> QCheckBox:
            box = QCheckBox(text)
            box.setChecked(checked)
            return box

        self._cb_src_pts = _cb("Source landmarks", self.layer_src_pts.visible)
        self._cb_rings = _cb("TRE rings", self.layer_rings.visible)
        self._cb_vectors = _cb("Error vectors", self.layer_err.visible)
        self._cb_arrows = _cb("DVF arrows", self.layer_dvf_arrows.visible)
        self._cb_warped = _cb("Warped source", self.layer_warped.visible)
        self._cb_diff = _cb("Target − warped", self.layer_diff.visible)
        self._cb_ident = _cb("Target − source", self.layer_ident_diff.visible)
        self._cb_truth_tre = _cb(
            "Truth color by TRE",
            bool(self._marker_style_cache["truth"].get("color_by_tre", True)),
        )
        self._cb_truth_tre.toggled.connect(self._on_truth_tre_toggled)
        checks = [
            self._cb_src_pts,
            self._cb_rings,
            self._cb_vectors,
            self._cb_arrows,
            self._cb_warped,
            self._cb_diff,
            self._cb_ident,
            self._cb_truth_tre,
        ]
        for i, box in enumerate(checks):
            grid.addWidget(box, i // 2, i % 2)
        lay.addWidget(vis)

        # --- Arrow params ---
        arrows = QGroupBox("Arrow sampling")
        aform = QFormLayout(arrows)
        aform.setContentsMargins(8, 8, 8, 8)
        self._spin_arrow_step = QSpinBox()
        self._spin_arrow_step.setRange(1, 32)
        self._spin_arrow_step.setValue(int(self._arrow_step))
        aform.addRow("Spacing (voxels)", self._spin_arrow_step)
        self._spin_arrow_gain = QDoubleSpinBox()
        self._spin_arrow_gain.setRange(0.1, 20.0)
        self._spin_arrow_gain.setSingleStep(0.5)
        self._spin_arrow_gain.setValue(float(self._arrow_gain))
        aform.addRow("Length multiplier", self._spin_arrow_gain)
        lay.addWidget(arrows)

        # --- In-plane rotation (full 0→90→180→270 clockwise) ---
        rot = QGroupBox("In-plane rotation")
        rot_lay = QVBoxLayout(rot)
        rot_lay.setContentsMargins(8, 8, 8, 8)
        rot_lay.setSpacing(4)
        self._rot_deg_label = QLabel("0°")
        self._rot_deg_label.setStyleSheet(
            "font-family: monospace; font-size: 13px; font-weight: 600;"
        )
        rot_lay.addWidget(self._rot_deg_label)
        row = QHBoxLayout()
        btn_ccw = QPushButton("↺ 90°")
        btn_ccw.setToolTip("Counter-clockwise 90° (Shift+R)")
        btn_ccw.clicked.connect(lambda: self.rotate_inplane(-90))
        btn_cw = QPushButton("↻ 90°")
        btn_cw.setToolTip("Clockwise 90° — full cycle (R / left transpose button)")
        btn_cw.clicked.connect(lambda: self.rotate_inplane(90))
        btn_reset = QPushButton("Reset")
        btn_reset.setToolTip("Reset rotation + flips (0)")
        btn_reset.clicked.connect(self.reset_inplane)
        row.addWidget(btn_ccw)
        row.addWidget(btn_cw)
        row.addWidget(btn_reset)
        rot_lay.addLayout(row)
        hint = QLabel("Left toolbar ⟳ = CW each click (0→90→180→270→0)")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; color: #888;")
        rot_lay.addWidget(hint)
        lay.addWidget(rot)

        lay.addStretch(1)

        # Callable for smoke / programmatic reload (replaces magicgui FunctionGui).
        self.controls = self._apply_overlay_controls
        self._controls_panel = root
        self._refill_case_combo(select_case=self.run.case)
        self._update_case_tre_label()
        self._sync_rot_widgets()
        self._ctrl_orient.currentTextChanged.connect(self._set_orient)
        self._ctrl_dvf.currentTextChanged.connect(self._apply_display_controls)
        for box in checks[:-1]:
            box.toggled.connect(self._apply_display_controls)
        self._spin_arrow_step.valueChanged.connect(self._apply_display_controls)
        self._spin_arrow_gain.valueChanged.connect(self._apply_display_controls)
        return self._as_scroll(root)

    def _report_error(self, message: str) -> None:
        self.viewer.status = message
        self._set_label("_error_label", message)

    def _apply_display_controls(self, *_args) -> None:
        """Apply presentation-only controls without reloading volumes or landmarks."""
        self.layer_src_pts.visible = self._cb_src_pts.isChecked()
        self.layer_rings.visible = self._cb_rings.isChecked()
        self.layer_err.visible = self._cb_vectors.isChecked()
        self._dvf_component = self._ctrl_dvf.currentText()
        self._arrow_step = int(self._spin_arrow_step.value())
        self._arrow_gain = float(self._spin_arrow_gain.value())
        self.layer_dvf_arrows.visible = self._cb_arrows.isChecked()
        self.layer_warped.visible = self._cb_warped.isChecked()
        self.layer_diff.visible = self._cb_diff.isChecked()
        self.layer_ident_diff.visible = self._cb_ident.isChecked()
        if (self._dvf_component != "off" or any(layer.visible for layer in (
            self.layer_dvf_arrows, self.layer_warped, self.layer_diff, self.layer_ident_diff
        ))):
            self._refresh_dvf_overlays()
        else:
            self.layer_dvf_mag.visible = False
            self.layer_dvf_comp.visible = False
        # A manual visibility change ends the blink cycle.
        self._blink_warped = False
        self.layer_ct.visible = True

    def _apply_overlay_controls(self) -> None:
        """Read compact Controls widgets and refresh overlays / volumes."""
        if self._phase_record is not None:
            self._report_error("Use Return to KPI view in Phase Performance to change the primary TRE pair.")
            return
        field_w = getattr(self, "_ctrl_field", None)
        if field_w is not None and self._qt_alive(field_w):
            field = field_w.currentText() or self.field
            pair = self._ctrl_pair.currentText()
            which = self._ctrl_which.currentText()
            try:
                src_ph, dst_ph = pair_phases(pair)
                vol_dst = (self.vol_dst if self.vol_dst.phase == dst_ph
                           else load_pack_volume(self.run, dst_ph))
                vol_src = (self.vol_src if self.vol_src.phase == src_ph
                           else load_pack_volume(self.run, src_ph))
                pl = per_landmark(self.run, field, which=which, pair=pair, use_cache=False)
            except (OSError, ValueError, ImportError, RuntimeError, KeyError) as exc:
                self._sync_field_choices()
                self._report_error(f"Could not apply data: {exc}")
                return
            self.field, self.pair, self.which = field, pair, which
            self.vol_dst, self.vol_src = vol_dst, vol_src
            self.layer_ct.data = vol_dst.data
            self.layer_ct.name = f"CT {dst_ph} (target)"
            self.layer_src.data = vol_src.data
            self.layer_src.name = f"CT {src_ph} (source)"
            self._bundle = None
            self._bundle_key = None
            self._blink_warped = False
            self.layer_ct.visible = True
            self._set_label("_error_label", "")
            self.layer_src_pts.visible = self._cb_src_pts.isChecked()
            self.layer_rings.visible = self._cb_rings.isChecked()
            self.layer_err.visible = self._cb_vectors.isChecked()
            self.layer_dvf_arrows.visible = self._cb_arrows.isChecked()
            self.layer_warped.visible = self._cb_warped.isChecked()
            self.layer_diff.visible = self._cb_diff.isChecked()
            self.layer_ident_diff.visible = self._cb_ident.isChecked()
            if self._qt_alive(getattr(self, "_cb_truth_tre", None)):
                self._marker_style_cache["truth"]["color_by_tre"] = (
                    self._cb_truth_tre.isChecked()
                )
            self._dvf_component = self._ctrl_dvf.currentText()
            self._arrow_step = int(self._spin_arrow_step.value())
            self._arrow_gain = float(self._spin_arrow_gain.value())
            orient = self._ctrl_orient.currentText()
            self._set_orient(orient)
        else:
            pl = None
        self._refresh_landmarks(pl)
        self._update_case_tre_label()
        self._refresh_dvf_overlays()
        self._refresh_drr_panel()
        self._apply_inplane_ornament()

    def _on_truth_tre_toggled(self, checked: bool) -> None:
        self._marker_style_cache.setdefault("truth", {})["color_by_tre"] = bool(
            checked
        )
        if hasattr(self, "pl"):
            colors = tre_to_rgba(self.pl.tre_mm)
            self._apply_marker_style(
                "truth",
                self.layer_truth,
                n=len(self.pl.tre_mm),
                tre_colors=colors,
            )

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
            combo.addItem(self._case_combo_label(run), str(run.run_root))
        # select
        idx = 0
        for i in range(combo.count()):
            if self._catalog[i].run_root == self.run.run_root:
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
            self._report_error(f"Runs folder error: {exc}")
            return
        if not catalog:
            self._report_error(f"No DIR_Cxx runs in {runs_dir}")
            return
        pick = next((r for r in catalog if r.case == self.run.case), catalog[0])
        if not self._switch_to_run(pick):
            self._runs_dir_edit.setText(str(self._runs_dir))
            return
        self._runs_dir = runs_dir
        self._catalog = catalog
        if self._qt_alive(getattr(self, "_runs_dir_edit", None)):
            self._runs_dir_edit.setText(str(runs_dir))
        # Prefer same case number if present, else first
        self._refill_case_combo(select_case=pick.case)
        self.viewer.status = f"Loaded {len(catalog)} case(s) from {runs_dir}"

    def _on_case_combo_changed(self, index: int) -> None:
        if index < 0 or not self._catalog:
            return
        root = self._case_combo.itemData(index)
        if root is None:
            return
        if root == str(self.run.run_root):
            # still update TRE label
            self._update_case_tre_label()
            return
        hit = next((r for r in self._catalog if str(r.run_root) == root), None)
        if hit is None:
            return
        if not self._switch_to_run(hit):
            self._refill_case_combo(select_case=self.run.case)

    def _update_case_tre_label(self) -> None:
        lab = getattr(self, "_case_tre_label", None)
        if lab is None or not self._qt_alive(lab):
            return
        pl = getattr(self, "pl", None)
        if pl is None:
            lab.setText("Loading landmark results…")
            return
        if self._phase_record is not None and self._phase_record.landmarks is None:
            lab.setText(f"{self.run.run_root.name} · {self._phase_record.phase}: TRE unavailable")
            return
        lab.setText(
            f"Loaded C{pl.case:02d} · {pl.which} landmarks · {pl.pair.replace('_', ' → ')}\n"
            f"{pl.field}: {pl.registered_stats['mean']:.2f} mm · "
            f"identity: {pl.identity_stats['mean']:.2f} mm"
        )

    def _sync_field_choices(self) -> None:
        field_w = getattr(self, "_ctrl_field", None)
        if field_w is None or not self._qt_alive(field_w):
            return
        fields = list(self.run.fields_available) or ["identity"]
        if self._phase_record is not None:
            fields.append(self.field)
        for selector in (self._ctrl_field, self._ctrl_pair, self._ctrl_which):
            selector.setEnabled(self._phase_record is None)
        self._ctrl_pair.blockSignals(True)
        self._ctrl_pair.clear()
        self._ctrl_pair.addItems(list(dict.fromkeys(["T00_T50", "T50_T00", self.pair])))
        self._ctrl_pair.blockSignals(False)
        try:
            cur = field_w.currentText()
            field_w.blockSignals(True)
            field_w.clear()
            field_w.addItems(fields)
            if self.field in fields:
                field_w.setCurrentText(self.field)
            elif cur in fields:
                field_w.setCurrentText(cur)
                self.field = cur
            else:
                self.field = default_field(self.run)
                field_w.setCurrentText(self.field)
            field_w.blockSignals(False)
        except Exception:
            pass

        # Keep other selectors in sync with live app state after case switch.
        for attr, value in (
            ("_ctrl_pair", self.pair),
            ("_ctrl_which", self.which),
            ("_ctrl_orient", self._orient_name),
            ("_ctrl_dvf", self._dvf_component),
        ):
            w = getattr(self, attr, None)
            if w is not None and self._qt_alive(w):
                try:
                    w.setCurrentText(str(value))
                except Exception:
                    pass
        if self._qt_alive(getattr(self, "_spin_arrow_step", None)):
            self._spin_arrow_step.setValue(int(self._arrow_step))
        if self._qt_alive(getattr(self, "_spin_arrow_gain", None)):
            self._spin_arrow_gain.setValue(float(self._arrow_gain))

    def _switch_to_run(self, run: RunRef, *, phase_record=None) -> bool:
        """Load everything before changing case/phase so failed CT loads roll back."""
        if phase_record is not None:
            from .phases import empty_landmarks
            field, pair, which = phase_record.stage, f"T50_{phase_record.phase}", "75"
            pl = phase_record.landmarks or empty_landmarks(phase_record)
        else:
            field = self.field if self.field in run.fields_available else default_field(run)
            pair = "T00_T50" if self._phase_record is not None else self.pair
            which = "75" if self._phase_record is not None else self.which
        src_ph, dst_ph = pair_phases(pair)
        try:
            vol_dst = load_pack_volume(run, dst_ph)
            vol_src = load_pack_volume(run, src_ph)
            if phase_record is None:
                pl = per_landmark(run, field, which=which, pair=pair)
        except (OSError, ValueError, ImportError, RuntimeError, KeyError) as exc:
            self._report_error(f"Could not load C{run.case:02d}: {exc}")
            return False
        for layer in self._phase_images:
            if layer in self.viewer.layers:
                self.viewer.layers.remove(layer)
        self._phase_images = []
        self._phase_record = phase_record
        self.run, self.field, self.pair, self.which = run, field, pair, which
        self._bundle = None
        self._bundle_key = None
        self._blink_warped = False
        self.layer_ct.visible = True
        self._set_label("_error_label", "")
        self._worst_cycle = 0

        src_ph, dst_ph = pair_phases(self.pair)
        self.vol_dst = vol_dst
        self.vol_src = vol_src
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
        self._refill_case_combo(select_case=self.run.case)
        self._update_case_tre_label()
        self._apply_thick_slices()
        self._refresh_landmarks(pl)
        self._refresh_dvf_overlays()
        # Re-resolve projection geometry for the new case.
        try:
            from tre_viewer.data import model_training_dir
            from tre_viewer.drr import resolve_geometry

            self._geom = resolve_geometry(self.run)
            self._mt = model_training_dir(self.run.run_root, self.run.scan_id)
            self._drr_view = min(
                int(self._drr_view), max(0, self._geom.n_views - 1)
            )
        except Exception:
            self._geom = None
            self._mt = None
        self._refresh_projection_views()
        self._apply_inplane_ornament()
        self.viewer.status = (
            f"Switched → {run.arm} C{run.case:02d}  field={self.field}"
        )
        return True

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
                # Soft hint only — hard mins clip small screens / narrow columns.
                dock_widget.setMinimumWidth(min(min_w, 160))
            except Exception:
                pass
        if min_h:
            try:
                dock_widget.setMinimumHeight(min(min_h, 120))
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
                dock.setMinimumWidth(min(min_w, 160))
            except Exception:
                pass
        if min_h:
            try:
                dock.setMinimumHeight(min(min_h, 120))
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
        if name in ("TRE", "Controls", "DRR", "Phase Performance"):
            try:
                self._finalize_layout()
            except Exception:
                pass
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
            self._refresh_landmarks(self.pl)
            self._append_warp_summary()
        elif name == "Controls":
            self._refill_case_combo(select_case=self.run.case)
            self._update_case_tre_label()
            self._sync_field_choices()
        elif name == "DRR":
            self._refresh_projection_views(skip_pages=True)

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
        try:
            self._finalize_layout()
        except Exception:
            pass

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
            panels.addSeparator()
            act_drr = QAction("Open DRR full page", qt)
            act_drr.setShortcut(QKeySequence("Shift+D"))
            act_drr.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            act_drr.triggered.connect(lambda *_a: self.open_drr_page())
            panels.addAction(act_drr)
            act_rtk = QAction("Open RTK landmarks full page", qt)
            act_rtk.setShortcut(QKeySequence("Shift+T"))
            act_rtk.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            act_rtk.triggered.connect(lambda *_a: self.open_rtk_page())
            panels.addAction(act_rtk)
            self._panels_menu = panels

            act_fs = QAction("Toggle Fullscreen", qt)
            act_fs.setShortcut(QKeySequence("F11"))
            act_fs.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            act_fs.triggered.connect(lambda *_a: self.toggle_fullscreen())
            win_menu.addAction(act_fs)

            # Export PNGs from the main TRE window
            export_menu = win_menu.addMenu("Export PNG")
            act_canvas = QAction("CT canvas…", qt)
            act_canvas.setShortcut(QKeySequence("Ctrl+E"))
            act_canvas.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            act_canvas.triggered.connect(
                lambda *_a: self.export_main_png(canvas_only=True)
            )
            export_menu.addAction(act_canvas)
            act_win = QAction("Full main window…", qt)
            act_win.setShortcut(QKeySequence("Ctrl+Shift+E"))
            act_win.setShortcutContext(Qt.ShortcutContext.ApplicationShortcut)
            act_win.triggered.connect(
                lambda *_a: self.export_main_png(canvas_only=False)
            )
            export_menu.addAction(act_win)
        except Exception as exc:  # noqa: BLE001
            self.viewer.status = f"menu setup: {exc}"

        # Do NOT also bind_key("F11") / bind_key("P"): QAction shortcuts already
        # own those keys. A second handler toggles fullscreen twice → no-op.

    def export_main_png(self, *, canvas_only: bool = True) -> None:
        """Save the main napari CT canvas or the whole main window as PNG."""
        from pathlib import Path

        from qtpy.QtWidgets import QFileDialog

        from tre_viewer.proj_pages import suggest_export_png

        kind = "canvas" if canvas_only else "window"
        default = str(suggest_export_png(self, kind))
        path, _ = QFileDialog.getSaveFileName(
            self.viewer.window._qt_window,
            "Export CT canvas PNG" if canvas_only else "Export main window PNG",
            default,
            "PNG image (*.png)",
        )
        if not path:
            return
        if not str(path).lower().endswith(".png"):
            path = f"{path}.png"
        try:
            if canvas_only:
                # napari canvas (CT + overlays), no Qt chrome
                self.viewer.screenshot(path, canvas_only=True, flash=False)
            else:
                pix = self.viewer.window._qt_window.grab()
                if not pix.save(str(path), "PNG"):
                    raise RuntimeError("QPixmap.save returned False")
            self.viewer.status = f"Saved {kind} PNG → {path}"
        except Exception as exc:  # noqa: BLE001
            self.viewer.status = f"PNG export failed: {exc}"

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
            f"| R=CW90° Shift+R=CCW90° F/Shift+F=flip 0=reset  "
            f"P=panels F11=fullscreen"
        )
        self._sync_rot_widgets()

    def _sync_rot_widgets(self) -> None:
        lab = getattr(self, "_rot_deg_label", None)
        if lab is None or not self._qt_alive(lab):
            return
        deg = (int(self._rot90_k) * 90) % 360
        lab.setText(f"{deg}°  ({self._orient_name})")

    def _hijack_napari_rotate_button(self) -> None:
        """Left-toolbar transpose ⟳ → full clockwise 90° cycle (not 2-state swap).

        Stock napari: click = transpose (only 2 poses); Alt-click = layer affine
        rotate. That feels broken for CT diagnosis. We rebind to our ornament
        state so each click advances 0→90→180→270→0 on every plane.
        """
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QApplication

        try:
            vb = self.viewer.window._qt_viewer.viewerButtons
            btn = vb.transposeDimsButton
        except Exception:
            return

        # Drop napari's Alt→rotate_layers filter and transpose action binding.
        try:
            btn.removeEventFilter(vb)
        except Exception:
            pass
        try:
            btn.clicked.disconnect()
        except Exception:
            pass

        def _on_click(*_args) -> None:
            mods = QApplication.keyboardModifiers()
            if mods & Qt.KeyboardModifier.AltModifier:
                self.rotate_inplane(-90)
            else:
                self.rotate_inplane(90)

        btn.clicked.connect(_on_click)
        btn.setToolTip(
            "Rotate view 90° clockwise (full cycle: 0° → 90° → 180° → 270° → 0°).\n"
            "Alt/Option-click: 90° counter-clockwise.\n"
            "Keys: R = CW, Shift+R = CCW, 0 = reset"
        )

    def rotate_inplane(self, degrees: int) -> None:
        steps = int(round(degrees / 90.0)) % 4
        # Python % on negative: (-1)%4 == 3 → one step CCW. Good.
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
        """Bottom launcher: open full-page DRR / RTK windows from the main viewer."""
        from qtpy.QtCore import Qt
        from qtpy.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

        from tre_viewer.data import model_training_dir
        from tre_viewer.drr import resolve_geometry

        self._drr_view = int(getattr(self, "_drr_view", 0))
        self._drr_mode = getattr(self, "_drr_mode", "target")
        try:
            self._geom = resolve_geometry(self.run)
            self._mt = model_training_dir(self.run.run_root, self.run.scan_id)
        except Exception as exc:  # noqa: BLE001
            self._geom = None
            self._mt = None
            self.viewer.status = f"DRR panel unavailable: {exc}"
            return None

        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(10, 8, 10, 8)
        lay.setSpacing(8)

        hdr = QLabel("Projection pages")
        hdr.setStyleSheet("font-weight: 600; font-size: 13px;")
        lay.addWidget(hdr)
        sub = QLabel(
            "Open a full-page window for DRR image scrub or RTK landmark overlay. "
            "Shift+D / Shift+T from the main window."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("font-size: 11px;")
        lay.addWidget(sub)

        row = QHBoxLayout()
        btn_drr = QPushButton("DRR — full page")
        btn_drr.setMinimumHeight(44)
        btn_drr.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_drr.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: 600; padding: 8px 16px; }"
        )
        btn_drr.clicked.connect(self.open_drr_page)
        btn_rtk = QPushButton("RTK landmarks — full page")
        btn_rtk.setMinimumHeight(44)
        btn_rtk.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_rtk.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: 600; padding: 8px 16px; }"
        )
        btn_rtk.clicked.connect(self.open_rtk_page)
        row.addWidget(btn_drr, stretch=1)
        row.addWidget(btn_rtk, stretch=1)
        lay.addLayout(row)

        self._drr_meta = QLabel("")
        self._drr_meta.setWordWrap(True)
        self._drr_meta.setStyleSheet("font-family: monospace; font-size: 11px;")
        lay.addWidget(self._drr_meta)
        self._drr_canvas = None
        self._drr_ax = None
        self._drr_slider = None
        self._drr_panel = wrap
        return wrap

    def _build_drr_dock(self) -> None:
        """Register the bottom projection launcher + mode keybinding."""
        dock = self._register_dock(
            name="DRR",
            builder=self._build_drr_panel,
            area="bottom",
            min_h=120,
            min_w=280,
        )
        if dock is None:
            self._panel_specs.pop("DRR", None)
            return
        try:
            dock.setMaximumHeight(200)
        except Exception:
            pass

        @self.viewer.bind_key("d")
        def _cycle_drr(viewer):  # noqa: ARG001
            modes = ["target", "source", "diff"]
            i = modes.index(self._drr_mode)
            self._drr_mode = modes[(i + 1) % len(modes)]
            self._refresh_projection_views()
            self.viewer.status = f"DRR mode → {self._drr_mode}"

    def open_drr_page(self) -> None:
        self._open_proj_page("drr")

    def open_rtk_page(self) -> None:
        self._open_proj_page("rtk")

    def _open_proj_page(self, kind: str) -> None:
        if self._phase_record is not None:
            return None
        if getattr(self, "_geom", None) is None or getattr(self, "_mt", None) is None:
            self.viewer.status = (
                "No geometry / ModelTraining — cannot open projection page"
            )
            return
        existing = self._proj_pages.get(kind)
        if existing is not None and self._qt_alive(getattr(existing, "win", None)):
            existing.raise_window()
            self.viewer.status = f"Focused {kind.upper()} full page"
            return
        from tre_viewer.proj_pages import ProjectionPageWindow

        self._proj_pages[kind] = ProjectionPageWindow(
            self, kind=kind, on_closed=self._on_proj_page_closed
        )
        self.viewer.status = f"Opened {kind.upper()} full page (Esc to close)"

    def _on_proj_page_closed(self, kind: str) -> None:
        self._proj_pages.pop(kind, None)

    def _sync_drr_sliders(self, *, except_slider=None) -> None:
        for page in list(self._proj_pages.values()):
            try:
                if except_slider is not None and page.slider is except_slider:
                    continue
                page.sync_slider()
            except Exception:
                pass

    def _drr_frame_data(self) -> dict | None:
        """Shared projection frame for dock meta + full-page windows."""
        if self._phase_record is not None:
            return None
        if getattr(self, "_geom", None) is None or getattr(self, "_mt", None) is None:
            return None
        if not hasattr(self, "pl"):
            return None
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
            src = load_proj_128(proj_path(self._mt, src_ph, view_1, source=False))
        except Exception as exc:  # noqa: BLE001
            tp = proj_path(self._mt, dst_ph, view_1, source=False)
            sp = proj_path(self._mt, src_ph, view_1, source=False)
            return {
                "error": (
                    f"proj load failed: {exc}\n"
                    f"  dst {dst_ph}: {tp}\n  src {src_ph}: {sp}"
                )
            }

        mode = getattr(self, "_drr_mode", "target")
        if mode == "source":
            img = src
            title = f"source {src_ph}  view {view_1}"
        elif mode == "diff":
            img = tgt - src
            title = f"{dst_ph}−{src_ph}  view {view_1}"
        else:
            img = tgt
            title = f"target {dst_ph}  view {view_1}"

        truth_mm, _ = r3_ct_physical_landmarks(self.run, self.pl.truth_pack)
        pred_mm, _ = r3_ct_physical_landmarks(self.run, self.pl.pred_pack)
        M = self._geom.matrices[vi]
        truth_uv = project_landmarks_to_128(truth_mm, M)
        pred_uv = project_landmarks_to_128(pred_mm, M)
        d2 = np.linalg.norm(pred_uv - truth_uv, axis=1)
        angle = float(self._geom.gantry_deg[vi])
        return {
            "img": img,
            "title": title,
            "mode": mode,
            "truth_uv": truth_uv,
            "pred_uv": pred_uv,
            "d2": d2,
            "angle": angle,
            "view_1": view_1,
            "geom_name": self._geom.path.name,
        }

    def _refresh_drr_panel(self) -> None:
        """Back-compat alias — refresh launcher meta + any open full pages."""
        self._refresh_projection_views()

    def _refresh_projection_views(self, *, skip_pages: bool = False) -> None:
        frame = self._drr_frame_data()
        if self._qt_alive(getattr(self, "_drr_meta", None)):
            if frame is None:
                self._drr_meta.setText("Projection data unavailable")
            elif frame.get("error"):
                self._drr_meta.setText(str(frame["error"]))
            else:
                d2 = frame["d2"]
                self._drr_meta.setText(
                    f"mode={frame['mode']}  view {frame['view_1']:03d}  "
                    f"angle {frame['angle']:.1f}°  "
                    f"2D TRE {float(d2.mean()):.2f}±{float(d2.std()):.2f} px  "
                    f"·  click DRR / RTK above for full page"
                )
        if skip_pages:
            return
        for page in list(self._proj_pages.values()):
            try:
                if self._qt_alive(getattr(page, "win", None)):
                    page.refresh()
            except Exception:
                pass

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
        widget = getattr(self, "_ctrl_orient", None)
        if self._qt_alive(widget) and widget.currentText() != name:
            widget.blockSignals(True)
            widget.setCurrentText(name)
            widget.blockSignals(False)
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

    def _marker_layers(self) -> dict[str, object]:
        return {
            "truth": self.layer_truth,
            "pred": self.layer_pred,
            "src": self.layer_src_pts,
            "rings": self.layer_rings,
            "err": self.layer_err,
        }

    def _wire_marker_style_persistence(self) -> None:
        """Keep napari left-panel face/size/symbol edits across landmark reloads."""

        def _capture(key: str):
            def _handler(*_args, **_kwargs) -> None:
                if self._applying_marker_style:
                    return
                lyr = self._marker_layers().get(key)
                if lyr is None:
                    return
                try:
                    self._marker_style_cache[key] = self._snapshot_marker_style(
                        key, lyr, from_user_edit=True
                    )
                    if key == "truth":
                        cb = getattr(self, "_cb_truth_tre", None)
                        if self._qt_alive(cb):
                            cb.blockSignals(True)
                            cb.setChecked(
                                bool(
                                    self._marker_style_cache["truth"].get(
                                        "color_by_tre", True
                                    )
                                )
                            )
                            cb.blockSignals(False)
                except Exception:
                    pass

            return _handler

        for key, lyr in self._marker_layers().items():
            for evt_name in (
                "size",
                "face_color",
                "border_color",
                "border_width",
                "symbol",
                "opacity",
                "blending",
                "shading",
                "edge_color",
                "edge_width",
            ):
                ev = getattr(getattr(lyr, "events", None), evt_name, None)
                if ev is None:
                    continue
                try:
                    ev.connect(_capture(key))
                except Exception:
                    pass

    @staticmethod
    def _as_rgba_row(color) -> np.ndarray | None:
        try:
            arr = np.asarray(color, dtype=float)
        except Exception:
            return None
        if arr.size == 0:
            return None
        if arr.ndim == 1 and arr.size in (3, 4):
            if arr.size == 3:
                arr = np.concatenate([arr, [1.0]])
            return arr.reshape(4)
        if arr.ndim == 2 and arr.shape[1] in (3, 4):
            row = arr[0]
            if row.size == 3:
                row = np.concatenate([row, [1.0]])
            return np.asarray(row, dtype=float).reshape(4)
        return None

    def _snapshot_marker_style(
        self, key: str, layer, *, from_user_edit: bool = False
    ) -> dict:
        prev = dict(self._marker_style_cache.get(key, {}))
        out = dict(prev)
        if key == "err":
            try:
                out["edge_width"] = float(layer.edge_width)
            except Exception:
                pass
            try:
                out["opacity"] = float(layer.opacity)
            except Exception:
                pass
            ec = getattr(layer, "edge_color", None)
            row = self._as_rgba_row(ec)
            if row is not None:
                arr = np.asarray(ec, dtype=float)
                uniform = arr.ndim == 1 or (
                    arr.ndim == 2 and np.allclose(arr, arr[0:1], atol=1e-3)
                )
                if not uniform:
                    out["color_by_tre"] = True
                elif from_user_edit:
                    out["color_by_tre"] = False
                    out["edge_color"] = tuple(float(x) for x in row)
                else:
                    out["edge_color"] = tuple(float(x) for x in row)
            return out

        for attr in ("opacity", "blending", "symbol", "shading"):
            if hasattr(layer, attr):
                try:
                    out[attr] = getattr(layer, attr)
                except Exception:
                    pass
        try:
            bw = layer.border_width
            out["border_width"] = float(np.mean(np.asarray(bw, dtype=float)))
        except Exception:
            pass
        try:
            sz = np.asarray(layer.size, dtype=float).reshape(-1)
            if sz.size:
                med = float(np.median(sz))
                out["size"] = med
                if key == "rings":
                    if from_user_edit:
                        out["size_by_tre"] = bool(np.std(sz) > 0.5)
                    elif np.std(sz) > 0.5:
                        out["size_by_tre"] = True
        except Exception:
            pass

        fc = getattr(layer, "face_color", None)
        row = self._as_rgba_row(fc)
        if row is not None:
            arr = np.asarray(fc, dtype=float)
            uniform = arr.ndim == 1 or (
                arr.ndim == 2
                and (len(arr) <= 1 or np.allclose(arr, arr[0:1], atol=1e-3))
            )
            if key == "truth":
                if not uniform:
                    out["color_by_tre"] = True
                elif from_user_edit:
                    out["color_by_tre"] = False
                    out["face_color"] = tuple(float(x) for x in row)
                else:
                    # Keep prior TRE/solid choice; still remember solid colour.
                    out["face_color"] = tuple(float(x) for x in row)
            elif key == "rings":
                out["face_color"] = (0.0, 0.0, 0.0, 0.0)
            else:
                if uniform:
                    out["face_color"] = tuple(float(x) for x in row)
                    if from_user_edit:
                        out["color_by_tre"] = False

        bc = getattr(layer, "border_color", None)
        brow = self._as_rgba_row(bc)
        if brow is not None:
            arr = np.asarray(bc, dtype=float)
            uniform = arr.ndim == 1 or (
                arr.ndim == 2
                and (len(arr) <= 1 or np.allclose(arr, arr[0:1], atol=1e-3))
            )
            if key == "rings":
                if not uniform:
                    out["color_by_tre"] = True
                elif from_user_edit:
                    out["color_by_tre"] = False
                    out["border_color"] = tuple(float(x) for x in brow)
                else:
                    out["border_color"] = tuple(float(x) for x in brow)
            elif uniform:
                out["border_color"] = tuple(float(x) for x in brow)
        return out

    def _apply_marker_style(
        self,
        key: str,
        layer,
        *,
        n: int,
        tre_colors: np.ndarray | None = None,
        tre_sizes: np.ndarray | None = None,
    ) -> None:
        if n == 0:
            return
        style = self._marker_style_cache.get(key, {})
        self._applying_marker_style = True
        try:
            if key == "err":
                if style.get("color_by_tre") and tre_colors is not None:
                    layer.edge_color = tre_colors
                elif style.get("edge_color") is not None:
                    layer.edge_color = self._broadcast_color(style["edge_color"], n)
                if style.get("edge_width") is not None:
                    layer.edge_width = float(style["edge_width"])
                if style.get("opacity") is not None:
                    layer.opacity = float(style["opacity"])
                return

            if key == "rings" and style.get("size_by_tre", True) and tre_sizes is not None:
                layer.size = tre_sizes
            elif style.get("size") is not None:
                layer.size = float(style["size"])

            if style.get("color_by_tre") and tre_colors is not None:
                if key == "rings":
                    layer.border_color = tre_colors
                    layer.face_color = np.zeros((n, 4), dtype=float)
                else:
                    layer.face_color = tre_colors
            else:
                if style.get("face_color") is not None:
                    layer.face_color = self._broadcast_color(style["face_color"], n)
                if style.get("border_color") is not None:
                    layer.border_color = self._broadcast_color(
                        style["border_color"], n
                    )
                elif key == "rings" and tre_colors is not None:
                    layer.border_color = tre_colors

            for attr in ("border_width", "opacity", "blending", "symbol", "shading"):
                if attr in style and style[attr] is not None and hasattr(layer, attr):
                    try:
                        setattr(layer, attr, style[attr])
                    except Exception:
                        pass
        finally:
            self._applying_marker_style = False

    @staticmethod
    def _broadcast_color(color, n: int):
        """Napari Points wants a color name, or an (N,4) array — not a bare RGBA tuple."""
        if isinstance(color, str):
            return color
        row = np.asarray(color, dtype=float).reshape(-1)
        if row.size == 3:
            row = np.concatenate([row, [1.0]])
        if row.size != 4:
            return color
        if n <= 0:
            return row
        return np.repeat(row.reshape(1, 4), n, axis=0)

    def _refresh_landmarks(self, pl=None) -> None:
        pl = pl if pl is not None else per_landmark(
            self.run,
            self.field,
            which=self.which,  # type: ignore[arg-type]
            pair=self.pair,
        )
        self.pl = pl
        self._worst_cycle = 0
        self._set_label("selected_label", "Click a landmark or press W to inspect its error.")
        (nx, ny, nz), spacing = CASE_INFO[self.run.case]

        # Capture any in-panel edits before data assignment resets napari state.
        for key, lyr in self._marker_layers().items():
            try:
                if key == "err":
                    if len(getattr(lyr, "data", [])) > 0:
                        self._marker_style_cache[key] = self._snapshot_marker_style(
                            key, lyr
                        )
                elif len(lyr.data) > 0:
                    self._marker_style_cache[key] = self._snapshot_marker_style(
                        key, lyr
                    )
            except Exception:
                pass

        truth_zyx = xyz_to_zyx(pl.truth_pack)
        pred_zyx = xyz_to_zyx(pl.pred_pack)
        src_zyx = xyz_to_zyx(pl.src_pack)
        colors = tre_to_rgba(pl.tre_mm)
        n = len(pl.tre_mm)

        mean_inplane = 0.5 * (spacing[0] + spacing[1])
        ring_vox = np.maximum(pl.tre_mm / max(mean_inplane, 1e-6) * 1.5, 10.0)

        self.layer_truth.data = truth_zyx
        self.layer_truth.features = {
            "tre_mm": pl.tre_mm,
            "id": np.arange(n),
        }
        self._apply_marker_style(
            "truth", self.layer_truth, n=n, tre_colors=colors
        )

        self.layer_pred.data = pred_zyx
        self._apply_marker_style("pred", self.layer_pred, n=n)

        self.layer_src_pts.data = src_zyx
        self._apply_marker_style("src", self.layer_src_pts, n=n)

        self.layer_rings.data = truth_zyx
        self._apply_marker_style(
            "rings",
            self.layer_rings,
            n=n,
            tre_colors=colors,
            tre_sizes=ring_vox,
        )

        vecs = error_vectors_zyx(pl.truth_pack, pl.pred_pack, spacing)
        self.layer_err.data = vecs
        self._apply_marker_style("err", self.layer_err, n=n, tre_colors=colors)

        self._set_label(
            "summary_label",
            summary_line(pl)
            + f"\nfield={self.field}  oob={int(pl.oob_mask.sum())}  "
            f"display=pack-mm  DVF_frame={pl.frame}\n"
            f"truth=green disc · pred=red cross  |  "
            f"W=worst  Space=blink",
        )
        self.viewer.title = (
            f"TRE Viewer │ {self.run.arm} C{self.run.case:02d} │ "
            f"{self.field} │ mean {pl.registered_stats['mean']:.2f} mm"
        )

        self._update_case_tre_label()
        self._worst_rows = worst_table(pl, k=15)
        if self._qt_alive(getattr(self, "worst_list", None)):
            self.worst_list.clear()
            for row in self._worst_rows:
                self.worst_list.addItem(
                    f"#{row['id']:02d}  TRE {row['tre_mm']:.2f}  "
                    f"idnt {row['ident_mm']:.2f}  z={row['slice_z']}"
                )

        if self._phase_record is not None and self._phase_record.landmarks is None:
            message = f"{self.run.run_root.name} · {self._phase_record.phase}: TRE unavailable\n{self._phase_record.reason}"
            self._set_label("summary_label", message)
            self._set_label("_case_tre_label", message)
            self._set_label("selected_label", "No scored landmarks for this phase.")
            self.viewer.title = f"TRE Viewer │ {self.run.run_root.name} │ {self._phase_record.phase} │ TRE unavailable"

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
        if not hasattr(self, "pl") or not len(self.pl.tre_mm):
            return
        key = self.pl.identity_mm if by_identity else self.pl.tre_mm
        order = np.argsort(-key)
        idx = int(order[self._worst_cycle % len(order)])
        self._worst_cycle += 1
        self.jump_to_landmark(idx)

    def _refresh_dvf_overlays(self) -> None:
        """Recompute warp/DVF pack overlays for the current field/pair."""
        key = (self.run.run_root, self.field, self.pair)
        try:
            if self._phase_record is not None:
                raise ValueError("Phase inspection shows landmarks and real CT; use Check synth CT for image comparison.")
            bundle = (self._bundle if self._bundle is not None and self._bundle_key == key
                      else field_warp_bundle(self.run, self.field, self.pair))
        except Exception as exc:  # noqa: BLE001 — show in UI, don't crash viewer
            self._bundle = None
            self._bundle_key = None
            self._blink_warped = False
            self.layer_ct.visible = True
            empty = np.zeros_like(self.vol_dst.data, dtype=np.float32)
            for layer in (self.layer_warped, self.layer_diff, self.layer_ident_diff,
                          self.layer_dvf_mag, self.layer_dvf_comp):
                layer.data = empty
                layer.visible = False
            self.layer_dvf_arrows.data = np.zeros((0, 2, 3))
            self.layer_dvf_arrows.visible = False
            for name in ("_cb_warped", "_cb_diff", "_cb_ident", "_cb_arrows"):
                box = getattr(self, name, None)
                if self._qt_alive(box):
                    box.blockSignals(True)
                    box.setChecked(False)
                    box.blockSignals(False)
            if self._phase_record is None:
                self._report_error(f"DVF overlay unavailable: {exc}")
            else:
                self._set_label("_error_label", "")
            label = getattr(self, "summary_label", None)
            if self._qt_alive(label):
                label.setText(label.text().split("\nwarp")[0])
            return
        self._bundle = bundle
        self._bundle_key = key
        self._set_label("_error_label", "")
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

        self._refresh_arrows()

        self._append_warp_summary()
        mae_w = bundle["mae_warped_lung"]
        mae_i = bundle["mae_ident_lung"]
        self.viewer.status = (
            f"DVF overlays ready │ {bundle['mae_region']} MAE warped {mae_w:.1f} vs ident {mae_i:.1f} HU │ "
            f"Space=blink"
        )

    def _refresh_arrows(self, _event=None) -> None:
        if (self._bundle is None or not self.layer_dvf_arrows.visible
                or self.viewer.dims.ndim != 3
                or self.layer_dvf_arrows not in self.viewer.layers
                or getattr(self, "_updating_arrows", False)):
            return
        axis = int(self.viewer.dims.order[0])
        self._updating_arrows = True
        try:
            self.layer_dvf_arrows.data = pack_arrows_zyx(
                self._bundle["dvf"], self.vol_dst.data.shape, frame=self.run.frame,
                step=self._arrow_step, slice_axis=axis,
                slice_index=int(self.viewer.dims.current_step[axis]), gain=self._arrow_gain,
            )
        finally:
            self._updating_arrows = False

    def _append_warp_summary(self) -> None:
        """Append the warp-vs-identity lung MAE line to the summary panel."""
        bundle = getattr(self, "_bundle", None)
        if bundle is None or not self._qt_alive(getattr(self, "summary_label", None)):
            return
        mae_w = bundle["mae_warped_lung"]
        mae_i = bundle["mae_ident_lung"]
        extra = (
            f"\nwarp {bundle['mae_region']} MAE {mae_w:.1f} HU  identity {mae_i:.1f} HU  "
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
        if not 0 <= idx < len(pl.tre_mm):
            return
        panel = getattr(self, "_phase_panel", None)
        if self._phase_record is not None and self._qt_alive(panel):
            if panel.landmark.value() != idx:
                panel.landmark.blockSignals(True)
                panel.landmark.setValue(idx)
                panel.landmark.blockSignals(False)
                panel.redraw()
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

    # 4) Same for the Controls panel; Reload must still run after rebuild.
    dock = app._docks["Controls"]
    old_controls = app.controls
    old_widget = app._panel_specs["Controls"]["widget"]
    app.viewer.window.remove_dock_widget(dock)
    old_widget.setParent(None)
    old_widget.deleteLater()
    flush()
    assert not app._qt_alive(old_widget), "Controls content survived deleteLater"
    app.show_panel("Controls")
    alive_and_shown("Controls")
    assert app.controls is not old_controls, "Controls callback was not rebound"
    assert app._qt_alive(app._case_combo), "case combo missing after Controls rebuild"
    app.controls()  # Reload overlays path on freshly built widgets
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
