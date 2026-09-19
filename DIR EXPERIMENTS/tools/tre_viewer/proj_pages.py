"""Full-page DRR / RTK windows opened from the main TRE viewer."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np

if TYPE_CHECKING:
    from tre_viewer.app import TreViewerApp

PageKind = Literal["drr", "rtk"]


def suggest_export_png(app: "TreViewerApp", kind: str) -> Path:
    """Default ``…/tre/exports/<arm>_Cxx_<field>_<kind>_vNNN_<mode>.png`` path."""
    run = app.run
    view = int(getattr(app, "_drr_view", 0)) + 1
    mode = getattr(app, "_drr_mode", "target")
    field = str(getattr(app, "field", "field")).replace("/", "-")
    out_dir = Path(run.run_root) / "tre" / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / (
        f"{run.arm}_C{run.case:02d}_{field}_{kind}_v{view:03d}_{mode}.png"
    )


class ProjectionPageWindow:
    """Maximized Qt window for DRR image scrub or RTK landmark overlay."""

    def __init__(
        self,
        app: "TreViewerApp",
        *,
        kind: PageKind,
        on_closed: Callable[[PageKind], None] | None = None,
    ):
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure
        from qtpy.QtCore import Qt
        from qtpy.QtGui import QKeySequence, QShortcut
        from qtpy.QtWidgets import (
            QComboBox,
            QHBoxLayout,
            QLabel,
            QListWidget,
            QMainWindow,
            QPushButton,
            QSlider,
            QSplitter,
            QVBoxLayout,
            QWidget,
        )

        self.app = app
        self.kind = kind
        self._on_closed = on_closed
        self.fig = None
        self._selected_lm: int | None = None
        self._worst_cycle = 0
        self._worst_rows: list[dict] = []
        self._sort_by = "tre3d"  # tre3d | tre2d
        self._last_d2: np.ndarray | None = None

        title = (
            "DRR — full page"
            if kind == "drr"
            else "RTK landmarks — full page"
        )
        self.win = QMainWindow()
        self.win.setWindowTitle(
            f"{title} │ {app.run.arm} C{app.run.case:02d} │ {app.field}"
        )
        self.win.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.win.destroyed.connect(self._handle_destroyed)

        root = QWidget()
        outer = QVBoxLayout(root)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(6)

        hdr = QLabel(
            "DRR projection  ·  worst-landmark list on the right  ·  "
            "click a row to highlight (W = cycle)"
            if kind == "drr"
            else "RTK landmarks  ·  worst-landmark list on the right  ·  "
            "click a row to highlight (W = cycle)"
        )
        hdr.setWordWrap(True)
        hdr.setStyleSheet("font-size: 13px;")
        outer.addWidget(hdr)

        split = QSplitter(Qt.Orientation.Horizontal)

        # --- left: canvas ---
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(0, 0, 0, 0)
        left_lay.setSpacing(4)

        self.fig = Figure(figsize=(9.0, 8.5), tight_layout=True)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.canvas.setMinimumSize(520, 520)
        left_lay.addWidget(self.canvas, stretch=1)

        tools = QHBoxLayout()
        self.btn_tgt = QPushButton("Target")
        self.btn_src = QPushButton("Source")
        self.btn_diff = QPushButton("Diff")
        self.btn_tgt.clicked.connect(lambda: self._set_mode("target"))
        self.btn_src.clicked.connect(lambda: self._set_mode("source"))
        self.btn_diff.clicked.connect(lambda: self._set_mode("diff"))
        for b in (self.btn_tgt, self.btn_src, self.btn_diff):
            tools.addWidget(b)
        tools.addStretch(1)
        export_btn = QPushButton("Export PNG…")
        export_btn.setToolTip("Save this view as PNG (Ctrl+S)")
        export_btn.clicked.connect(self.export_png)
        tools.addWidget(export_btn)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.win.close)
        tools.addWidget(close_btn)
        left_lay.addLayout(tools)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        n = max(0, int(getattr(app, "_geom", None).n_views if app._geom else 1) - 1)
        self.slider.setMinimum(0)
        self.slider.setMaximum(n)
        self.slider.setValue(int(getattr(app, "_drr_view", 0)))
        self.slider.valueChanged.connect(self._on_slider)
        left_lay.addWidget(self.slider)

        self.meta = QLabel("")
        self.meta.setWordWrap(True)
        self.meta.setStyleSheet("font-family: monospace; font-size: 12px;")
        self.meta.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        left_lay.addWidget(self.meta)
        split.addWidget(left)

        # --- right: worst landmarks ---
        right = QWidget()
        right.setMinimumWidth(260)
        right.setMaximumWidth(420)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(4, 0, 0, 0)
        right_lay.setSpacing(4)

        side_hdr = QLabel("Worst landmarks")
        side_hdr.setStyleSheet("font-weight: 600; font-size: 13px;")
        right_lay.addWidget(side_hdr)

        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItem("3D TRE (mm)", "tre3d")
        self._sort_combo.addItem("2D residual (px)", "tre2d")
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        sort_row.addWidget(self._sort_combo, stretch=1)
        right_lay.addLayout(sort_row)

        jump_row = QHBoxLayout()
        btn_w = QPushButton("W · next worst")
        btn_w.setToolTip("Cycle worst by current sort (W)")
        btn_w.clicked.connect(lambda: self.jump_worst(by_identity=False))
        btn_sw = QPushButton("Shift+W · identity")
        btn_sw.setToolTip("Cycle worst by 3D identity TRE")
        btn_sw.clicked.connect(lambda: self.jump_worst(by_identity=True))
        jump_row.addWidget(btn_w)
        jump_row.addWidget(btn_sw)
        right_lay.addLayout(jump_row)

        self.worst_list = QListWidget()
        self.worst_list.setUniformItemSizes(True)
        self.worst_list.itemClicked.connect(self._on_worst_clicked)
        right_lay.addWidget(self.worst_list, stretch=1)

        self._sel_label = QLabel("selected: —")
        self._sel_label.setWordWrap(True)
        self._sel_label.setStyleSheet(
            "font-family: monospace; font-size: 11px; padding: 2px;"
        )
        right_lay.addWidget(self._sel_label)

        hint = QLabel(
            "Click a landmark → highlight on detector + jump CT slice.\n"
            "List shows 3D TRE and 2D residual at the current view."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; color: #888;")
        right_lay.addWidget(hint)
        split.addWidget(right)
        split.setStretchFactor(0, 5)
        split.setStretchFactor(1, 2)
        outer.addWidget(split, stretch=1)

        self.win.setCentralWidget(root)

        QShortcut(QKeySequence("Esc"), self.win, activated=self.win.close)
        QShortcut(QKeySequence("D"), self.win, activated=self._cycle_mode)
        QShortcut(QKeySequence("Ctrl+S"), self.win, activated=self.export_png)
        QShortcut(QKeySequence("W"), self.win, activated=lambda: self.jump_worst())
        QShortcut(
            QKeySequence("Shift+W"),
            self.win,
            activated=lambda: self.jump_worst(by_identity=True),
        )

        self.refresh()
        self.win.showMaximized()
        self.win.raise_()
        self.win.activateWindow()

    def _handle_destroyed(self, *_args) -> None:
        if self._on_closed is not None:
            try:
                self._on_closed(self.kind)
            except Exception:
                pass

    def _set_mode(self, mode: str) -> None:
        self.app._drr_mode = mode
        self.app._refresh_projection_views()

    def _cycle_mode(self) -> None:
        modes = ["target", "source", "diff"]
        i = modes.index(getattr(self.app, "_drr_mode", "target"))
        self.app._drr_mode = modes[(i + 1) % len(modes)]
        self.app._refresh_projection_views()

    def _on_slider(self, value: int) -> None:
        self.app._drr_view = int(value)
        self.app._sync_drr_sliders(except_slider=self.slider)
        self.app._refresh_projection_views(skip_pages=False)

    def _on_sort_changed(self, *_args) -> None:
        data = self._sort_combo.currentData()
        if data:
            self._sort_by = str(data)
        self._refill_worst_list()

    def sync_slider(self) -> None:
        if not self.app._qt_alive(self.slider):
            return
        geometry = self.app._geom
        nmax = max(0, geometry.n_views - 1) if geometry else 0
        v = min(max(0, int(self.app._drr_view)), nmax)
        self.slider.blockSignals(True)
        self.slider.setMaximum(nmax)
        self.slider.setValue(v)
        self.slider.setEnabled(geometry is not None and nmax > 0)
        self.slider.blockSignals(False)

    def _refill_worst_list(self) -> None:
        from tre_viewer.encodings import worst_table

        pl = getattr(self.app, "pl", None)
        if pl is None or not self.app._qt_alive(self.worst_list):
            return
        d2 = self._last_d2
        if d2 is None or len(d2) != len(pl.tre_mm):
            d2 = np.zeros(len(pl.tre_mm), dtype=float)

        if self._sort_by == "tre2d":
            order = np.argsort(-d2)
            rows = []
            for rank, idx in enumerate(order[:15]):
                rows.append(
                    {
                        "rank": rank + 1,
                        "id": int(idx),
                        "tre_mm": float(pl.tre_mm[idx]),
                        "ident_mm": float(pl.identity_mm[idx]),
                        "d2_px": float(d2[idx]),
                    }
                )
            self._worst_rows = rows
        else:
            self._worst_rows = worst_table(pl, k=15)
            for row in self._worst_rows:
                i = row["id"]
                row["d2_px"] = float(d2[i]) if i < len(d2) else float("nan")

        self.worst_list.blockSignals(True)
        self.worst_list.clear()
        for row in self._worst_rows:
            self.worst_list.addItem(
                f"#{row['id']:02d}  TRE {row['tre_mm']:.2f} mm  "
                f"2D {row.get('d2_px', float('nan')):.1f} px  "
                f"idnt {row['ident_mm']:.2f}"
            )
        # Reselect if still in list
        if self._selected_lm is not None:
            for i, row in enumerate(self._worst_rows):
                if row["id"] == self._selected_lm:
                    self.worst_list.setCurrentRow(i)
                    break
        self.worst_list.blockSignals(False)

    def _on_worst_clicked(self, item) -> None:
        row = self.worst_list.row(item)
        if 0 <= row < len(self._worst_rows):
            self.select_landmark(self._worst_rows[row]["id"])

    def jump_worst(self, *, by_identity: bool = False) -> None:
        pl = getattr(self.app, "pl", None)
        if pl is None or len(pl.tre_mm) == 0:
            return
        if by_identity:
            key = pl.identity_mm
        elif self._sort_by == "tre2d" and self._last_d2 is not None:
            key = self._last_d2
        else:
            key = pl.tre_mm
        order = np.argsort(-np.asarray(key))
        idx = int(order[self._worst_cycle % len(order)])
        self._worst_cycle += 1
        self.select_landmark(idx)

    def select_landmark(self, idx: int) -> None:
        """Highlight on this page and sync the main CT viewer."""
        self._selected_lm = int(idx)
        pl = getattr(self.app, "pl", None)
        d2 = self._last_d2
        if pl is not None and 0 <= idx < len(pl.tre_mm):
            d2v = float(d2[idx]) if d2 is not None and idx < len(d2) else float("nan")
            self._sel_label.setText(
                f"selected #{idx}  TRE {pl.tre_mm[idx]:.2f} mm  "
                f"2D {d2v:.2f} px  idnt {pl.identity_mm[idx]:.2f}"
            )
            for i, row in enumerate(self._worst_rows):
                if row["id"] == idx:
                    self.worst_list.setCurrentRow(i)
                    break
        try:
            self.app.jump_to_landmark(idx)
        except Exception:
            pass
        self.refresh()

    def export_png(self) -> None:
        """Save the current matplotlib figure as PNG."""
        from qtpy.QtWidgets import QFileDialog

        if self.fig is None:
            return
        try:
            self.canvas.draw()
        except Exception:
            pass
        default = str(suggest_export_png(self.app, self.kind))
        path, _ = QFileDialog.getSaveFileName(
            self.win,
            f"Export {self.kind.upper()} PNG",
            default,
            "PNG image (*.png)",
        )
        if not path:
            return
        if not str(path).lower().endswith(".png"):
            path = f"{path}.png"
        try:
            self.fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
            self.app.viewer.status = f"Saved {self.kind.upper()} PNG → {path}"
            base = self.meta.text().split("\nSaved")[0]
            self.meta.setText(base + f"\nSaved → {path}")
        except Exception as exc:  # noqa: BLE001
            self.app.viewer.status = f"PNG export failed: {exc}"

    def refresh(self) -> None:
        self.sync_slider()
        self.win.setWindowTitle(
            f"{self.kind.upper()} │ {self.app.run.arm} C{self.app.run.case:02d} │ {self.app.field}"
        )
        frame = self.app._drr_frame_data()
        if frame is None or frame.get("error"):
            message = str(frame["error"]) if frame else "Projection data unavailable"
            self.meta.setText(message)
            self.ax.clear()
            self.ax.text(.5, .5, message, ha="center", va="center", wrap=True,
                         transform=self.ax.transAxes)
            self.canvas.draw_idle()
            self.worst_list.clear()
            self._worst_rows = []
            self._last_d2 = None
            self._selected_lm = None
            self._sel_label.setText("selected: —")
            return

        self.sync_slider()
        img = frame["img"]
        title = frame["title"]
        truth_uv = frame["truth_uv"]
        pred_uv = frame["pred_uv"]
        d2 = frame["d2"]
        angle = frame["angle"]
        mode = frame["mode"]
        self._last_d2 = np.asarray(d2, dtype=float)
        self._refill_worst_list()

        ax = self.ax
        ax.clear()
        if mode == "diff":
            lim = float(np.percentile(np.abs(img), 99)) or 1.0
            ax.imshow(img, cmap="coolwarm", vmin=-lim, vmax=lim, origin="upper")
        else:
            ax.imshow(img, cmap="gray", origin="upper")

        # Always show landmarks on DRR now (needed for worst-list workflow);
        # keep them lighter than the dedicated RTK page.
        if self.kind == "drr":
            ax.scatter(
                truth_uv[:, 0],
                truth_uv[:, 1],
                s=14,
                c="lime",
                marker="o",
                alpha=0.55,
                linewidths=0.2,
                edgecolors="k",
                label="truth",
            )
            ax.scatter(
                pred_uv[:, 0],
                pred_uv[:, 1],
                s=14,
                c="magenta",
                marker="x",
                alpha=0.55,
                label="pred",
            )
            ax.set_title(f"DRR  ·  {title}", fontsize=12)
            ax.legend(loc="upper right", fontsize=8)
        else:
            ax.scatter(
                truth_uv[:, 0],
                truth_uv[:, 1],
                s=36,
                c="lime",
                marker="o",
                label="truth (RTK)",
                linewidths=0.4,
                edgecolors="k",
            )
            ax.scatter(
                pred_uv[:, 0],
                pred_uv[:, 1],
                s=36,
                c="magenta",
                marker="x",
                label="pred (RTK)",
            )
            for t, p in zip(truth_uv, pred_uv):
                ax.plot(
                    [t[0], p[0]],
                    [t[1], p[1]],
                    color="yellow",
                    lw=0.8,
                    alpha=0.85,
                )
            ax.legend(loc="upper right", fontsize=9)
            ax.set_title(f"RTK landmarks  ·  {title}", fontsize=12)

        # Highlight selected / worst pick
        sel = self._selected_lm
        if sel is not None and 0 <= sel < len(truth_uv):
            t = truth_uv[sel]
            p = pred_uv[sel]
            ax.scatter(
                [t[0]],
                [t[1]],
                s=120,
                facecolors="none",
                edgecolors="cyan",
                linewidths=2.0,
                zorder=5,
            )
            ax.scatter(
                [p[0]],
                [p[1]],
                s=80,
                c="cyan",
                marker="x",
                linewidths=2.0,
                zorder=5,
            )
            ax.plot(
                [t[0], p[0]],
                [t[1], p[1]],
                color="cyan",
                lw=1.6,
                alpha=0.95,
                zorder=4,
            )
            ax.annotate(
                f"#{sel}",
                (t[0], t[1]),
                textcoords="offset points",
                xytext=(6, 6),
                color="cyan",
                fontsize=10,
                fontweight="bold",
            )

        ax.set_xlim(0, 127)
        ax.set_ylim(127, 0)
        self.canvas.draw_idle()

        inside = float(
            (
                (truth_uv[:, 0] >= 0)
                & (truth_uv[:, 0] < 128)
                & (truth_uv[:, 1] >= 0)
                & (truth_uv[:, 1] < 128)
            ).mean()
        )
        worst = int(np.argmax(d2)) if len(d2) else -1
        self.meta.setText(
            f"view {frame['view_1']:03d}  angle {angle:.2f}°  mode={mode}  "
            f"2D TRE mean {float(d2.mean()):.2f} px  max {float(d2.max()):.2f} px "
            f"(#{worst})  insideFOV {inside:.0%}\n"
            f"{frame['geom_name']}  ·  Esc=close  D=mode  W=worst  Ctrl+S=PNG"
        )

    def raise_window(self) -> None:
        self.win.showMaximized()
        self.win.raise_()
        self.win.activateWindow()
        self.refresh()
