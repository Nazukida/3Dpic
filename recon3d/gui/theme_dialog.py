"""Theme customization dialog.

Lets the user start from any existing theme, tweak every colour with a live
preview swatch, and save it as a named custom theme.  Emits nothing itself;
the caller reads :attr:`result_theme` after ``exec()`` returns ``Accepted``.
"""

from __future__ import annotations

import copy
from dataclasses import fields

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QComboBox, QColorDialog, QScrollArea, QWidget, QFrame,
    QDialogButtonBox, QMessageBox,
)

from . import themes as T


# human labels for each colour field, grouped
_CHROME_FIELDS = [
    ("bg", "窗口背景"),
    ("surface", "面板表面"),
    ("surface_alt", "输入框 / 列表行"),
    ("border", "边框"),
    ("text", "文字"),
    ("text_muted", "次要文字"),
    ("accent", "强调色"),
    ("accent_text", "强调色上的文字"),
    ("danger", "警示色"),
]
_VIEWER_FIELDS = [
    ("view_bg_top", "视图顶部"),
    ("view_bg_bottom", "视图底部"),
    ("view_axis_x", "X 轴"),
    ("view_axis_y", "Y 轴"),
    ("view_axis_z", "Z 轴"),
    ("view_grid", "网格"),
    ("view_point_tint", "点云着色"),
    ("view_text", "叠加文字"),
]


class _Swatch(QPushButton):
    """A clickable colour chip that opens a colour picker."""
    changed = Signal(str)

    def __init__(self, hex_color: str, parent=None):
        super().__init__(parent)
        self._hex = hex_color
        self.setFixedSize(120, 26)
        self.setCursor(Qt.PointingHandCursor)
        self.clicked.connect(self._pick)
        self._refresh()

    def _refresh(self):
        # pick readable text colour for the chip label
        r, g, b = T.hex_to_rgb(self._hex)
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        fg = "#000000" if lum > 140 else "#ffffff"
        self.setStyleSheet(
            f"QPushButton{{background:{self._hex};color:{fg};"
            f"border:1px solid rgba(128,128,128,0.5);border-radius:5px;"
            f"font-family:monospace;font-size:12px;}}"
        )
        self.setText(self._hex)

    def _pick(self):
        col = QColorDialog.getColor(QColor(self._hex), self, "选择颜色",
                                    QColorDialog.ShowAlphaChannel)
        if col.isValid():
            self._hex = col.name()
            self._refresh()
            self.changed.emit(self._hex)

    def hex(self) -> str:
        return self._hex


class ThemeDialog(QDialog):
    def __init__(self, current: T.Theme, parent=None):
        super().__init__(parent)
        self.setWindowTitle("自定义主题")
        self.setMinimumWidth(560)
        self.result_theme: T.Theme | None = None
        self._working = copy.deepcopy(current)
        self._working.builtin = False
        self._swatches: dict[str, _Swatch] = {}

        root = QVBoxLayout(self)

        # ---- base + name row ----
        top = QHBoxLayout()
        top.addWidget(QLabel("起始主题："))
        self._base = QComboBox()
        for name in T.all_themes():
            self._base.addItem(name)
        self._base.setCurrentText(current.name if current.name in T.all_themes() else T.DEFAULT_THEME)
        self._base.currentTextChanged.connect(self._load_base)
        top.addWidget(self._base, 1)
        top.addSpacing(12)
        top.addWidget(QLabel("另存为："))
        self._name = QLineEdit(self._suggest_name(current.name))
        top.addWidget(self._name, 1)
        root.addLayout(top)

        # ---- scrollable colour grid ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QWidget()
        grid = QGridLayout(inner)
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        row = 0
        row = self._add_section(grid, row, "界面")
        row = self._add_fields(grid, row, _CHROME_FIELDS)
        row = self._add_section(grid, row, "3D 视图")
        row = self._add_fields(grid, row, _VIEWER_FIELDS)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        # ---- live preview strip ----
        self._preview = QFrame()
        self._preview.setFixedHeight(56)
        self._preview.setObjectName("Card")
        root.addWidget(self._preview)
        self._update_preview()

        # ---- buttons ----
        btns = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._on_save)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    # ------------------------------------------------------------------ #
    def _suggest_name(self, base: str) -> str:
        existing = set(T.all_themes())
        cand = f"{base} 自定义"
        i = 2
        while cand in existing and existing:
            cand = f"{base} 自定义 {i}"
            i += 1
        return cand

    def _add_section(self, grid, row, title):
        lbl = QLabel(title)
        lbl.setObjectName("Title")
        grid.addWidget(lbl, row, 0, 1, 4)
        return row + 1

    def _add_fields(self, grid, row, spec):
        # two columns of (label, swatch) per grid row
        col = 0
        for field_name, label in spec:
            hexv = getattr(self._working, field_name)
            sw = _Swatch(hexv)
            sw.changed.connect(lambda v, f=field_name: self._on_color(f, v))
            self._swatches[field_name] = sw
            grid.addWidget(QLabel(label), row, col * 2)
            grid.addWidget(sw, row, col * 2 + 1)
            col += 1
            if col == 2:
                col = 0
                row += 1
        if col != 0:
            row += 1
        return row

    def _on_color(self, field_name, hexv):
        setattr(self._working, field_name, hexv)
        self._update_preview()

    def _load_base(self, name):
        base = T.all_themes().get(name)
        if not base:
            return
        keep_name = self._name.text()
        self._working = copy.deepcopy(base)
        self._working.builtin = False
        for f, sw in self._swatches.items():
            sw._hex = getattr(self._working, f)
            sw._refresh()
        self._name.setText(keep_name or self._suggest_name(name))
        self._update_preview()

    def _update_preview(self):
        w = self._working
        self._preview.setStyleSheet(
            f"#Card{{background:{w.surface};border:1px solid {w.border};border-radius:8px;}}"
        )
        # clear & rebuild preview content
        lay = self._preview.layout()
        if lay is None:
            lay = QHBoxLayout(self._preview)
            lay.setContentsMargins(12, 8, 12, 8)
        else:
            while lay.count():
                item = lay.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
        title = QLabel("Aa 预览")
        title.setStyleSheet(f"color:{w.text};font-weight:700;font-size:14px;")
        muted = QLabel("次要文字")
        muted.setStyleSheet(f"color:{w.text_muted};")
        btn = QLabel("  强调色  ")
        btn.setStyleSheet(
            f"background:{w.accent};color:{w.accent_text};border-radius:6px;padding:4px 8px;font-weight:600;")
        vbar = QLabel()
        vbar.setFixedWidth(90)
        vbar.setStyleSheet(
            f"border-radius:6px;background:qlineargradient(x1:0,y1:0,x2:0,y2:1,"
            f"stop:0 {w.view_bg_top},stop:1 {w.view_bg_bottom});"
            f"border:1px solid {w.border};")
        lay.addWidget(title)
        lay.addWidget(muted)
        lay.addStretch(1)
        lay.addWidget(btn)
        lay.addWidget(vbar)

    def _on_save(self):
        name = self._name.text().strip()
        if not name:
            QMessageBox.warning(self, "需要名称", "请输入主题名称。")
            return
        if name in T.PRESETS:
            QMessageBox.warning(self, "名称被占用",
                                f"“{name}”是内置预设主题，请换一个名称。")
            return
        self._working.name = name
        self._working.builtin = False
        self.result_theme = self._working
        self.accept()
