"""Reconstruction control panel (left dock).

Drives a reconstruction run and streams its progress into the UI.  The actual
pipeline runs as a **subprocess** (``python -m recon3d.run``) rather than in a
thread: the pipeline itself spawns a process pool for features/matching, and
nesting that inside the GUI process is fragile on Windows (spawn re-imports the
main module).  A subprocess is clean, killable, and mirrors the CLI exactly.

Signals:
* ``cloudReady(str)`` - absolute path to the primary output PLY when done.
* ``status(str)``     - short status line for the main window status bar.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox, QLineEdit,
    QPushButton, QComboBox, QCheckBox, QSpinBox, QFileDialog, QPlainTextEdit,
    QProgressBar, QLabel,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
QUALITY_PRESETS = ["fast", "balanced", "high", "ultra"]

# stage name -> (fraction-of-total start, label) for a coarse overall progress
_STAGE_WEIGHTS = [
    ("features", 0.00, "提取特征"),
    ("match", 0.20, "特征匹配"),
    ("tracks", 0.40, "构建轨迹"),
    ("sfm", 0.45, "稀疏重建"),
    ("dense", 0.70, "稠密重建"),
    ("clean", 0.92, "点云清理"),
    ("pipeline] Done", 1.00, "完成"),
]


class _RunnerThread(QThread):
    line = Signal(str)
    finished_ok = Signal(int)   # return code

    def __init__(self, cmd, cwd, parent=None):
        super().__init__(parent)
        self._cmd = cmd
        self._cwd = cwd
        self._proc: subprocess.Popen | None = None
        self._stop = False

    def run(self):
        try:
            self._proc = subprocess.Popen(
                self._cmd, cwd=self._cwd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        except Exception as e:
            self.line.emit(f"[error] failed to start: {e}")
            self.finished_ok.emit(-1)
            return
        assert self._proc.stdout is not None
        for raw in self._proc.stdout:
            if self._stop:
                break
            self.line.emit(raw.rstrip("\n"))
        code = self._proc.wait()
        self.finished_ok.emit(code)

    def stop(self):
        self._stop = True
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
            except Exception:
                pass


class ReconPanel(QWidget):
    cloudReady = Signal(str)
    status = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._runner: _RunnerThread | None = None
        self._out_dir = ""

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        # ---- 输入 ----
        box = QGroupBox("输入")
        form = QFormLayout(box)

        img_row = QHBoxLayout()
        self.img_edit = QLineEdit()
        self.img_edit.setPlaceholderText("包含重叠照片的文件夹…")
        browse_img = QPushButton("浏览")
        browse_img.clicked.connect(self._browse_images)
        img_row.addWidget(self.img_edit, 1)
        img_row.addWidget(browse_img)
        form.addRow("图片", self._wrap(img_row))

        out_row = QHBoxLayout()
        self.out_edit = QLineEdit(os.path.join(PROJECT_ROOT, "output"))
        browse_out = QPushButton("浏览")
        browse_out.clicked.connect(self._browse_out)
        out_row.addWidget(self.out_edit, 1)
        out_row.addWidget(browse_out)
        form.addRow("输出", self._wrap(out_row))

        self.img_count = QLabel("—")
        self.img_count.setObjectName("Muted")
        form.addRow("已检测", self.img_count)
        self.img_edit.textChanged.connect(self._update_count)

        root.addWidget(box)

        # ---- 选项 ----
        obox = QGroupBox("选项")
        of = QFormLayout(obox)
        self.quality = QComboBox()
        # 中文显示名，itemData 保留传给命令行的预设值
        _QUALITY_LABELS = {"fast": "快速", "balanced": "均衡",
                           "high": "高质量", "ultra": "极致"}
        for preset in QUALITY_PRESETS:
            self.quality.addItem(_QUALITY_LABELS.get(preset, preset), preset)
        self.quality.setCurrentIndex(QUALITY_PRESETS.index("balanced"))
        of.addRow("质量", self.quality)

        self.dense = QCheckBox("稠密重建（MVS）")
        self.dense.setChecked(True)
        of.addRow(self.dense)

        self.workers = QSpinBox()
        self.workers.setRange(0, (os.cpu_count() or 4))
        self.workers.setValue(0)
        self.workers.setSpecialValueText("自动")
        of.addRow("并行进程", self.workers)
        root.addWidget(obox)

        # ---- 运行控制 ----
        run_row = QHBoxLayout()
        self.start_btn = QPushButton("开始重建")
        self.start_btn.setObjectName("Primary")
        self.start_btn.clicked.connect(self._start)
        self.stop_btn = QPushButton("停止")
        self.stop_btn.setObjectName("Danger")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop)
        run_row.addWidget(self.start_btn, 1)
        run_row.addWidget(self.stop_btn)
        root.addLayout(run_row)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root.addWidget(self.progress)

        self.stage_label = QLabel("空闲")
        self.stage_label.setObjectName("Muted")
        root.addWidget(self.stage_label)

        # ---- 日志 ----
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setPlaceholderText("重建日志将显示在这里…")
        root.addWidget(self.log, 1)

        self._update_count()

    # ------------------------------------------------------------------ #
    def set_theme(self, theme):
        # monospace log tuned to theme surface
        self.log.setStyleSheet(
            f"QPlainTextEdit{{font-family:'Consolas','Menlo',monospace;"
            f"font-size:12px;background:{theme.surface_alt};color:{theme.text};}}")

    def _wrap(self, layout) -> QWidget:
        w = QWidget()
        layout.setContentsMargins(0, 0, 0, 0)
        w.setLayout(layout)
        return w

    def _browse_images(self):
        d = QFileDialog.getExistingDirectory(self, "选择图片文件夹",
                                             self.img_edit.text() or PROJECT_ROOT)
        if d:
            self.img_edit.setText(d)

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出文件夹",
                                             self.out_edit.text() or PROJECT_ROOT)
        if d:
            self.out_edit.setText(d)

    def _update_count(self):
        d = self.img_edit.text().strip()
        if d and os.path.isdir(d):
            try:
                n = sum(1 for f in os.listdir(d) if f.lower().endswith(IMAGE_EXTS))
            except OSError:
                n = 0
            self.img_count.setText(f"{n} 张图片" + ("" if n >= 2 else "  (至少需要 2 张)"))
            self.start_btn.setEnabled(n >= 2)
        else:
            self.img_count.setText("—")
            self.start_btn.setEnabled(False)

    # ------------------------------------------------------------------ #
    def _start(self):
        image_dir = self.img_edit.text().strip()
        if not image_dir or not os.path.isdir(image_dir):
            self.status.emit("请先选择有效的图片文件夹")
            return
        self._out_dir = self.out_edit.text().strip() or os.path.join(PROJECT_ROOT, "output")

        # currentData() 是英文预设值（fast/balanced/…），命令行需要它
        quality = self.quality.currentData() or "balanced"
        cmd = [sys.executable, "-u", "-m", "recon3d.run",
               "--images", image_dir, "--out", self._out_dir,
               "--quality", quality]
        if not self.dense.isChecked():
            cmd.append("--no-dense")
        if self.workers.value() > 0:
            cmd += ["--workers", str(self.workers.value())]

        self.log.clear()
        self.log.appendPlainText("$ " + " ".join(cmd))
        self.progress.setValue(0)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.stage_label.setText("正在启动…")
        self.status.emit("重建已开始")

        self._runner = _RunnerThread(cmd, PROJECT_ROOT, self)
        self._runner.line.connect(self._on_line)
        self._runner.finished_ok.connect(self._on_finished)
        self._runner.start()

    def _stop(self):
        if self._runner:
            self._runner.stop()
            self.stage_label.setText("正在停止…")
            self.status.emit("正在停止重建")

    def _on_line(self, line: str):
        self.log.appendPlainText(line)
        self._update_progress(line)

    _re_frac = re.compile(r"(\d+)\s*/\s*(\d+)")

    def _update_progress(self, line: str):
        base = None
        span = 0.2
        label = None
        for i, (key, start, lbl) in enumerate(_STAGE_WEIGHTS):
            if key in line:
                base = start
                label = lbl
                nxt = _STAGE_WEIGHTS[i + 1][1] if i + 1 < len(_STAGE_WEIGHTS) else 1.0
                span = max(nxt - start, 0.0)
                break
        if base is None:
            return
        frac_in_stage = 0.0
        m = self._re_frac.search(line)
        if m:
            cur, tot = int(m.group(1)), int(m.group(2))
            if tot:
                frac_in_stage = min(cur / tot, 1.0)
        overall = base + span * frac_in_stage
        self.progress.setValue(int(round(overall * 100)))
        if label:
            self.stage_label.setText(label)

    def _on_finished(self, code: int):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self._update_count()
        if code == 0:
            self.progress.setValue(100)
            self.stage_label.setText("完成")
            primary = self._find_primary_ply()
            if primary:
                self.status.emit(f"重建完成 → {os.path.basename(primary)}")
                self.cloudReady.emit(primary)
            else:
                self.status.emit("重建结束，但未找到 PLY 文件")
        else:
            self.stage_label.setText(f"失败（退出码 {code}）")
            self.status.emit(f"重建失败（退出码 {code}）")

    def _find_primary_ply(self) -> str | None:
        for name in ("dense.ply", "sparse.ply"):
            p = os.path.join(self._out_dir, name)
            if os.path.isfile(p) and os.path.getsize(p) > 0:
                return p
        return None

    def closeEvent(self, e):
        if self._runner and self._runner.isRunning():
            self._runner.stop()
            self._runner.wait(2000)
        super().closeEvent(e)
