"""Phase Performance dock: phase curves, clickable coverage map and trajectories."""
from __future__ import annotations

from pathlib import Path
from threading import Event

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib import colormaps
from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QSpinBox,
    QTabWidget, QFileDialog,
)
from napari.qt.threading import create_worker

from .phases import (
    PHASES, STAGES, REFERENCE, evaluate_cohort, export_results, synth_image_check,
)


class PhasePerformancePanel(QWidget):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.results = {}
        self.runs = []
        self.cancelled = Event()
        self.worker = None
        self.evaluation_error = None
        self.image_worker = None
        layout = QVBoxLayout(self)
        intro = QLabel(
            "75-point Sampled4D • T50 → each phase\n"
            "Direct synth and downstream VoxelMap are evaluated separately. "
            "Gray/blank cells mean TRE unavailable; T50 is the fixed reference."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        row = QHBoxLayout()
        self.stage = QComboBox()
        for key, label in STAGES.items():
            self.stage.addItem(label, key)
        self.stage.currentIndexChanged.connect(self._stage_changed)
        row.addWidget(self.stage)
        self.refresh_button = QPushButton("Evaluate runs")
        self.refresh_button.clicked.connect(self.evaluate)
        row.addWidget(self.refresh_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancelled.set)
        row.addWidget(self.cancel_button)
        layout.addLayout(row)
        row = QHBoxLayout()
        self.patient = QComboBox()
        self.patient.currentIndexChanged.connect(self.redraw)
        row.addWidget(self.patient)
        self.phase = QComboBox()
        self.phase.addItems(PHASES)
        self.phase.currentIndexChanged.connect(self._describe)
        row.addWidget(self.phase)
        open_button = QPushButton("Open CT")
        open_button.clicked.connect(self.open_selected)
        row.addWidget(open_button)
        layout.addLayout(row)
        row = QHBoxLayout()
        row.addWidget(QLabel("Heatmap"))
        self.metric = QComboBox()
        self.metric.addItem("Mean TRE (mm)", "mean")
        self.metric.addItem("p95 TRE (mm)", "p95")
        self.metric.addItem("Improvement (mm)", "improvement")
        self.metric.currentIndexChanged.connect(self.redraw)
        row.addWidget(self.metric)
        row.addWidget(QLabel("Landmark ID"))
        self.landmark = QSpinBox()
        self.landmark.setRange(0, 74)
        self.landmark.setToolTip("Zero-based ID, consistent with the TRE viewer")
        self.landmark.valueChanged.connect(self._landmark_changed)
        row.addWidget(self.landmark)
        layout.addLayout(row)
        self.tabs = QTabWidget()
        self.figures, self.canvases = [], []
        for label in ("Phase curves", "Patient × phase", "Landmark trajectory"):
            figure = Figure(figsize=(6, 4), layout="constrained")
            canvas = FigureCanvasQTAgg(figure)
            canvas.mpl_connect("button_press_event", self._plot_clicked)
            self.figures.append(figure)
            self.canvases.append(canvas)
            self.tabs.addTab(canvas, label)
        layout.addWidget(self.tabs, 1)
        self.status = QLabel("Choose a stage, then Evaluate runs. Uses fields on disk; no GPU inference in the GUI.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        layout.addWidget(self.detail)
        row = QHBoxLayout()
        self.image_button = QPushButton("Check synth CT")
        self.image_button.setToolTip("Compare saved synth CT against real CT; whole-volume HU MAE, independent of TRE")
        self.image_button.clicked.connect(self.image_check)
        row.addWidget(self.image_button)
        export_button = QPushButton("Export JSON")
        export_button.clicked.connect(self.export)
        row.addWidget(export_button)
        primary_button = QPushButton("Return to KPI view")
        primary_button.clicked.connect(lambda: app._switch_to_run(app.run))
        row.addWidget(primary_button)
        layout.addLayout(row)
        cancel = self.cancelled
        self.destroyed.connect(lambda *_: cancel.set())

    def _stage_changed(self):
        self.results = {}
        self.image_button.setEnabled(self.stage.currentData() == "synth")
        self.status.setText("Stage changed. Click Evaluate runs to load this stage.")
        self.redraw()

    def evaluate(self):
        if self.worker is not None:
            return
        self.cancelled.clear()
        self.evaluation_error = None
        self.runs = list(self.app._catalog)
        self.results = {}
        self.patient.blockSignals(True)
        self.patient.clear()
        for run in self.runs:
            self.patient.addItem(run.run_root.name)
        current = next((i for i, r in enumerate(self.runs) if r.run_root == self.app.run.run_root), 0)
        self.patient.setCurrentIndex(current)
        self.patient.blockSignals(False)
        self.stage.setEnabled(False)
        self.refresh_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status.setText("Evaluating phase fields…")
        self.redraw()
        self.worker = create_worker(
            evaluate_cohort, self.runs, self.stage.currentData(), cancelled=self.cancelled.is_set,
            _start_thread=False,
            _connect={"yielded": self._receive, "finished": self._finished, "errored": self._failed},
        )
        self.worker.start()

    def _receive(self, result):
        self.results[(str(result.run.run_root), result.phase)] = result
        self.status.setText(f"Evaluated {len(self.results)}/{len(self.runs) * len(PHASES)} cells")

    def _failed(self, error):
        self.evaluation_error = str(error)
        self.status.setText(f"Evaluation failed: {error}")

    def _finished(self):
        self.worker = None
        self.stage.setEnabled(True)
        self.refresh_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        counts = {key: sum(r.status == key for r in self.results.values())
                  for key in ("ok", "reference", "unavailable", "error")}
        self.status.setText(
            (f"Evaluation failed: {self.evaluation_error}. " if self.evaluation_error else "")
            + ("Cancelled; partial results. " if self.cancelled.is_set() else "")
            + f"{counts['ok']} scored • {counts['reference']} reference • "
            + f"{counts['unavailable']} unavailable • {counts['error']} invalid. "
            "Click a plotted phase or heatmap cell to inspect its CT."
        )
        self.redraw()

    def selected(self):
        idx = self.patient.currentIndex()
        if not 0 <= idx < len(self.runs):
            return None
        return self.results.get((str(self.runs[idx].run_root), self.phase.currentText()))

    def _patient_results(self):
        idx = self.patient.currentIndex()
        if not 0 <= idx < len(self.runs):
            return [None] * len(PHASES)
        return [self.results.get((str(self.runs[idx].run_root), p)) for p in PHASES]

    def redraw(self, *_):
        records = self._patient_results()
        xs = np.arange(10)
        figure = self.figures[0]
        figure.clear()
        ax, delta = figure.subplots(2, 1, sharex=True)
        for key, label in (("mean", "Mean TRE"), ("p95", "p95 TRE")):
            ax.plot(xs, [r.metric(key) if r else np.nan for r in records], "o-", label=label)
        ax.plot(xs, [r.landmarks.identity_stats['mean'] if r and r.landmarks else np.nan
                     for r in records], "--", color="gray", label="Identity mean")
        ax.set_ylabel("TRE (mm)")
        ax.set_title(f"{self.stage.currentText()} · {self.patient.currentText()}")
        ax.legend(fontsize=8)
        values = [r.metric("improvement") if r else np.nan for r in records]
        delta.plot(xs, values, "o-", color="teal")
        delta.axhline(0, color="gray", linewidth=1)
        delta.set_ylabel("Identity − model\n(mm; higher better)")
        delta.set_xticks(xs, PHASES, rotation=45)
        for a in (ax, delta):
            a.grid(alpha=.2)
            a.axvline(5, color="gray", alpha=.3)
        figure = self.figures[1]
        figure.clear()
        ax = figure.subplots()
        key = self.metric.currentData()
        matrix = np.full((max(1, len(self.runs)), 10), np.nan)
        for i, run in enumerate(self.runs):
            for j, phase in enumerate(PHASES):
                r = self.results.get((str(run.run_root), phase))
                if r and r.status == "ok":
                    matrix[i, j] = r.metric(key)
        cmap = colormaps["RdBu" if key == "improvement" else "magma"].with_extremes(bad="#b5b5b5")
        finite = matrix[np.isfinite(matrix)]
        limit = max(1., float(np.abs(finite).max())) if finite.size else 1.
        im = ax.imshow(np.ma.masked_invalid(matrix), cmap=cmap, aspect="auto",
                       vmin=-limit if key == "improvement" else 0, vmax=limit)
        figure.colorbar(im, ax=ax, label=self.metric.currentText())
        ax.set_xticks(xs, PHASES, rotation=45)
        ax.set_yticks(np.arange(len(self.runs)), [r.run_root.name for r in self.runs])
        ax.set_title("Gray = unavailable · ref = excluded from ranking")
        for i in range(len(self.runs)):
            for j in range(10):
                value = matrix[i, j]
                label = "ref" if j == 5 else (f"{value:.1f}" if np.isfinite(value) else "—")
                rgba = cmap(im.norm(value)) if np.isfinite(value) else (1, 1, 1, 1)
                luminance = np.dot(rgba[:3], [.299, .587, .114])
                ax.text(j, i, label, ha="center", va="center", fontsize=8,
                        color="black" if luminance > .55 else "white")
        figure = self.figures[2]
        figure.clear()
        axes = figure.subplots(3, 1, sharex=True)
        landmark = self.landmark.value()
        for axis, component, label in zip(axes, (2, 1, 0), ("SI ≈ z", "AP ≈ y", "LR ≈ x")):
            for attribute, style, legend in (("observed_mm", "o-", "Observed"), ("predicted_mm", "s--", "Predicted")):
                ys = [getattr(r, attribute)[landmark, component]
                      if r and getattr(r, attribute) is not None else np.nan for r in records]
                axis.plot(xs, ys, style, label=legend)
            axis.set_ylabel(f"{label}\nmm")
            axis.axhline(0, color="gray", linewidth=.6)
            axis.grid(alpha=.2)
        axes[0].set_title(f"Landmark #{landmark} · signed displacement from T50")
        axes[0].legend(fontsize=8)
        axes[-1].set_xticks(xs, PHASES, rotation=45)
        for canvas in self.canvases:
            canvas.draw_idle()
        self._describe()

    def _describe(self, *_):
        r = self.selected()
        if r is None:
            self.detail.setText("No evaluation for the selected cell.")
            return
        if r.status in ("ok", "reference"):
            text = (f"{r.run.run_root.name} · {r.phase}: mean {r.metric('mean'):.2f} mm, "
                    f"p95 {r.metric('p95'):.2f} mm, improvement {r.metric('improvement'):+.2f} mm. ")
            if r.metric("improvement") < 0:
                text += "Worse than identity. "
            valid = [x for x in self._patient_results() if x and x.status == 'ok']
            if valid:
                worst = max(valid, key=lambda x: x.metric('mean'))
                text += f"Highest mean among {len(valid)} scored phases: {worst.phase}. "
            self.detail.setText(text + r.reason)
        else:
            self.detail.setText(f"{r.run.run_root.name} · {r.phase}: TRE unavailable. {r.reason}")

    def _plot_clicked(self, event):
        if event.inaxes is None or event.xdata is None:
            return
        figure = event.canvas.figure
        if figure == self.figures[1]:
            if event.inaxes is not figure.axes[0] or event.ydata is None:
                return
            row = int(round(event.ydata))
            if not 0 <= row < len(self.runs):
                return
            self.patient.setCurrentIndex(row)
        phase = int(round(event.xdata))
        if 0 <= phase < 10:
            self.phase.setCurrentIndex(phase)
            self.open_selected()

    def open_selected(self):
        record = self.selected()
        if record is not None and self.app._switch_to_run(record.run, phase_record=record):
            if record.landmarks is not None:
                self.app.jump_to_landmark(self.landmark.value())
            self._describe()

    def _landmark_changed(self, *_):
        self.redraw()
        record = self.selected()
        if record is getattr(self.app, "_phase_record", None) and record and record.landmarks:
            self.app.jump_to_landmark(self.landmark.value())

    def image_check(self):
        record = self.selected()
        if record is None or self.image_worker is not None:
            return
        self.image_button.setEnabled(False)
        self.detail.setText("Loading real and synthesized CTs…")
        self.image_record = record
        self.image_worker = create_worker(
            synth_image_check, record.run, record.phase,
            _start_thread=False,
            _connect={"returned": self._image_ready,
                      "errored": self._image_failed,
                      "finished": self._image_finished},
        )
        self.image_worker.start()

    def _image_finished(self):
        self.image_worker = None
        self.image_button.setEnabled(self.stage.currentData() == "synth")

    def _image_failed(self, error):
        self.detail.setText(f"Image check unavailable: {error}")

    def _image_ready(self, result):
        record = self.image_record
        if self.selected() is not record:
            return  # selection changed during I/O
        if self.app._switch_to_run(record.run, phase_record=record):
            truth, pred, diff, mae = result
            self.app._phase_images = [
                self.app.viewer.add_image(pred, name=f"Synth CT {record.phase}", scale=truth.scale,
                                          colormap="gray", contrast_limits=(-1000, 500)),
                self.app.viewer.add_image(diff, name="Synth − real (HU)", scale=truth.scale,
                                          colormap="coolwarm", visible=False),
            ]
            for index, layer in enumerate(self.app._phase_images, start=2):
                self.app.viewer.layers.move(self.app.viewer.layers.index(layer), index)
            self.detail.setText(f"{record.phase} whole-volume MAE: {mae:.1f} HU (includes background). "
                                "Toggle Synth CT / Synth − real layers to inspect. Image similarity is separate from TRE.")

    def export(self):
        if not self.results:
            self.detail.setText("Evaluate runs before exporting.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export phase performance", "phase_performance.json", "JSON (*.json)")
        if path:
            try:
                export_results(Path(path), self.results.values(), expected_cells=len(self.runs) * len(PHASES))
                self.detail.setText(f"Exported {len(self.results)} cells to {path}")
            except OSError as error:
                self.detail.setText(f"Export failed: {error}")
