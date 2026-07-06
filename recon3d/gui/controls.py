"""Right-hand render-control panel for the viewer.

A thin widget of sliders/checkboxes that drives :class:`GLViewer` render
options and shows live cloud stats.  Kept separate from the main window so the
window file stays about layout and wiring, not widget plumbing.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QGroupBox, QSlider, QCheckBox,
    QLabel, QComboBox, QHBoxLayout, QPushButton,
)


class _Slider(QWidget):
    """Labelled float slider (integer track, float value)."""
    valueChanged = Signal(float)

    def __init__(self, lo, hi, val, step=0.1, fmt="{:.1f}", parent=None):
        super().__init__(parent)
        self._lo, self._hi, self._step, self._fmt = lo, hi, step, fmt
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._s = QSlider(Qt.Horizontal)
        self._s.setMinimum(0)
        self._s.setMaximum(int(round((hi - lo) / step)))
        self._s.setValue(int(round((val - lo) / step)))
        self._s.valueChanged.connect(self._emit)
        self._lbl = QLabel()
        self._lbl.setMinimumWidth(42)
        self._lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._lbl.setObjectName("Muted")
        lay.addWidget(self._s, 1)
        lay.addWidget(self._lbl)
        self._update_label(val)

    def _emit(self, ticks):
        v = self._lo + ticks * self._step
        self._update_label(v)
        self.valueChanged.emit(v)

    def _update_label(self, v):
        self._lbl.setText(self._fmt.format(v))

    def value(self):
        return self._lo + self._s.value() * self._step


class ControlPanel(QWidget):
    themeSelected = Signal(str)
    editThemeRequested = Signal()

    def __init__(self, viewer, parent=None):
        super().__init__(parent)
        self._viewer = viewer
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ---- 外观 / 主题 ----
        theme_box = QGroupBox("外观")
        tl = QVBoxLayout(theme_box)
        self.theme_combo = QComboBox()
        self.theme_combo.currentTextChanged.connect(self.themeSelected)
        tl.addWidget(self.theme_combo)
        row = QHBoxLayout()
        edit_btn = QPushButton("自定义…")
        edit_btn.clicked.connect(self.editThemeRequested)
        row.addWidget(edit_btn)
        tl.addLayout(row)
        root.addWidget(theme_box)

        # ---- 渲染选项 ----
        rbox = QGroupBox("渲染")
        form = QFormLayout(rbox)
        form.setLabelAlignment(Qt.AlignLeft)

        self.point_size = _Slider(0.5, 12.0, viewer.point_size, 0.5, "{:.1f}")
        self.point_size.valueChanged.connect(viewer.set_point_size)
        form.addRow("点大小", self.point_size)

        self.atten = _Slider(0.0, 1.0, viewer.attenuation, 0.05, "{:.2f}")
        self.atten.valueChanged.connect(viewer.set_attenuation)
        form.addRow("透视衰减", self.atten)

        self.bright = _Slider(0.3, 2.0, viewer.brightness, 0.05, "{:.2f}")
        self.bright.valueChanged.connect(viewer.set_brightness)
        form.addRow("亮度", self.bright)

        self.shaded = QCheckBox("球体光照")
        self.shaded.setChecked(viewer.shaded)
        self.shaded.toggled.connect(viewer.set_shaded)
        form.addRow(self.shaded)

        self.fast = QCheckBox("性能模式（扁平点，更流畅）")
        self.fast.setChecked(viewer.fast_mode)
        self.fast.toggled.connect(self._on_fast_toggled)
        form.addRow(self.fast)
        # the viewer reports the detected renderer once GL is up; reflect it
        viewer.rendererReady.connect(self._on_renderer_ready)

        self.grid = QCheckBox("显示网格")
        self.grid.setChecked(viewer.show_grid)
        self.grid.toggled.connect(viewer.set_show_grid)
        form.addRow(self.grid)

        self.axes = QCheckBox("显示坐标轴")
        self.axes.setChecked(viewer.show_axes)
        self.axes.toggled.connect(viewer.set_show_axes)
        form.addRow(self.axes)

        root.addWidget(rbox)

        # ---- 点云信息 ----
        sbox = QGroupBox("点云")
        sl = QVBoxLayout(sbox)
        self.stats = QLabel("尚未加载点云")
        self.stats.setObjectName("Muted")
        self.stats.setWordWrap(True)
        sl.addWidget(self.stats)
        reset = QPushButton("重置视角 (F)")
        reset.clicked.connect(viewer.reset_view)
        sl.addWidget(reset)
        root.addWidget(sbox)

        root.addStretch(1)

    def _on_fast_toggled(self, on: bool):
        self._viewer.set_fast_mode(on)

    def _on_renderer_ready(self, is_software: bool):
        # The viewer auto-enables fast mode on software renderers; reflect its
        # actual state in the checkbox without re-triggering the toggle.
        self.fast.blockSignals(True)
        self.fast.setChecked(self._viewer.fast_mode)
        self.fast.blockSignals(False)

    def set_theme_list(self, names, current):
        self.theme_combo.blockSignals(True)
        self.theme_combo.clear()
        self.theme_combo.addItems(list(names))
        if current in names:
            self.theme_combo.setCurrentText(current)
        self.theme_combo.blockSignals(False)

    def set_stats(self, text: str):
        self.stats.setText(text)
