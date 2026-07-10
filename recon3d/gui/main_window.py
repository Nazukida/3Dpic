"""Main application window.

Layout: a central :class:`GLViewer`, a dockable :class:`ControlPanel` on the
right, and a dockable reconstruction panel on the left (added lazily so the
viewer works even if the pipeline import is unavailable).  Menus/toolbar drive
file open, view reset, theme switching, and reconstruction.
"""

from __future__ import annotations

import os

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QFileDialog, QDockWidget, QMessageBox, QLabel,
    QApplication, QProgressBar, QWidget, QVBoxLayout,
)

from . import themes as T
from .gl_viewer import GLViewer
from .controls import ControlPanel
from .loader import PlyLoader
from .theme_dialog import ThemeDialog


class MainWindow(QMainWindow):
    def __init__(self, app: QApplication):
        super().__init__()
        self._app = app
        self.setWindowTitle("recon3d — 三维重建与点云查看器")
        self.resize(1280, 820)
        self.setAcceptDrops(True)

        self._settings = T.load_settings()
        self._theme_name = self._settings.get("theme", T.DEFAULT_THEME)
        self._last_dir = self._settings.get("last_dir", os.path.expanduser("~"))
        self._loader: PlyLoader | None = None
        self._recon_panel = None

        # central viewer
        self.viewer = GLViewer(self, theme=self._current_theme())
        self.viewer.status.connect(self._on_viewer_status)
        self.setCentralWidget(self.viewer)

        # right dock: controls
        self.controls = ControlPanel(self.viewer)
        self.controls.themeSelected.connect(self.apply_theme)
        self.controls.editThemeRequested.connect(self.open_theme_editor)
        dock = QDockWidget("控制面板", self)
        dock.setObjectName("ControlsDock")
        dock.setWidget(self.controls)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self._controls_dock = dock

        self._build_menus()
        self._build_statusbar()
        self._refresh_theme_list()
        self.apply_theme(self._theme_name, persist=False)

        # try to attach the reconstruction panel (optional dependency)
        self._try_add_recon_panel()

    # ------------------------------------------------------------------ #
    def _current_theme(self) -> T.Theme:
        return T.all_themes().get(self._theme_name, T.PRESETS[T.DEFAULT_THEME])

    def _build_menus(self):
        mb = self.menuBar()

        # 文件
        m_file = mb.addMenu("文件(&F)")
        act_open = QAction("打开 PLY…(&O)", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self.open_ply_dialog)
        m_file.addAction(act_open)
        m_file.addSeparator()
        act_quit = QAction("退出(&X)", self)
        act_quit.setShortcut(QKeySequence.Quit)
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)

        # 视图
        m_view = mb.addMenu("视图(&V)")
        act_reset = QAction("重置视角(&R)", self)
        act_reset.setShortcut("F")
        act_reset.triggered.connect(self.viewer.reset_view)
        m_view.addAction(act_reset)
        act_controls = QAction("切换控制面板(&C)", self)
        act_controls.setShortcut("C")
        act_controls.triggered.connect(
            lambda: self._controls_dock.setVisible(not self._controls_dock.isVisible()))
        m_view.addAction(act_controls)

        # 主题
        self.m_theme = mb.addMenu("主题(&T)")
        self._rebuild_theme_menu()

        # 帮助
        m_help = mb.addMenu("帮助(&H)")
        act_about = QAction("关于(&A)", self)
        act_about.triggered.connect(self._about)
        m_help.addAction(act_about)

    def _rebuild_theme_menu(self):
        self.m_theme.clear()
        for name in T.all_themes():
            act = QAction(name, self, checkable=True)
            act.setChecked(name == self._theme_name)
            act.triggered.connect(lambda _=False, n=name: self.apply_theme(n))
            self.m_theme.addAction(act)
        self.m_theme.addSeparator()
        act_custom = QAction("自定义…", self)
        act_custom.triggered.connect(self.open_theme_editor)
        self.m_theme.addAction(act_custom)
        act_del = QAction("删除自定义主题…", self)
        act_del.triggered.connect(self._delete_custom_theme)
        self.m_theme.addAction(act_del)

    def _build_statusbar(self):
        sb = self.statusBar()
        self._status_label = QLabel("就绪 — 打开一个 PLY 文件或运行三维重建")
        sb.addWidget(self._status_label, 1)
        self._progress = QProgressBar()
        self._progress.setMaximumWidth(180)
        self._progress.setVisible(False)
        sb.addPermanentWidget(self._progress)

    def _refresh_theme_list(self):
        self.controls.set_theme_list(list(T.all_themes().keys()), self._theme_name)

    # ------------------------------------------------------------------ #
    # Theme
    # ------------------------------------------------------------------ #
    def apply_theme(self, name: str, persist: bool = True):
        theme = T.all_themes().get(name)
        if theme is None:
            return
        self._theme_name = name
        self._app.setStyleSheet(T.build_qss(theme))
        self.viewer.set_theme(theme)
        self._refresh_theme_list()
        self._rebuild_theme_menu()
        if self._recon_panel is not None and hasattr(self._recon_panel, "set_theme"):
            self._recon_panel.set_theme(theme)
        if persist:
            self._settings["theme"] = name
            T.save_settings(self._settings)

    def open_theme_editor(self):
        dlg = ThemeDialog(self._current_theme(), self)
        if dlg.exec() == ThemeDialog.Accepted and dlg.result_theme:
            T.save_user_theme(dlg.result_theme)
            self.apply_theme(dlg.result_theme.name)

    def _delete_custom_theme(self):
        from PySide6.QtWidgets import QInputDialog
        user = T.load_user_themes()
        if not user:
            QMessageBox.information(self, "无自定义主题", "当前没有可删除的自定义主题。")
            return
        name, ok = QInputDialog.getItem(self, "删除自定义主题",
                                        "主题：", list(user.keys()), 0, False)
        if ok and name:
            T.delete_user_theme(name)
            if self._theme_name == name:
                self.apply_theme(T.DEFAULT_THEME)
            else:
                self._refresh_theme_list()
                self._rebuild_theme_menu()

    # ------------------------------------------------------------------ #
    # PLY loading
    # ------------------------------------------------------------------ #
    def open_ply_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "打开 PLY 点云", self._last_dir,
            "PLY 点云 (*.ply);;所有文件 (*)")
        if path:
            self.load_ply(path)

    def load_ply(self, path: str):
        if not os.path.isfile(path):
            QMessageBox.warning(self, "文件未找到", f"找不到文件：\n{path}")
            return
        self._last_dir = os.path.dirname(path)
        self._settings["last_dir"] = self._last_dir
        T.save_settings(self._settings)

        if self._loader is not None and self._loader.isRunning():
            self._status_label.setText("仍在加载上一个文件…")
            return

        self._progress.setRange(0, 0)   # busy indicator
        self._progress.setVisible(True)
        self._status_label.setText(f"正在加载 {os.path.basename(path)}…")

        self._loader = PlyLoader(path, parent=self)
        self._loader.loaded.connect(self._on_ply_loaded)
        self._loader.failed.connect(self._on_ply_failed)
        self._loader.start()

    def _on_ply_loaded(self, xyz, rgb, path, seconds, opacity, scale, is_gaussian):
        self._progress.setVisible(False)
        self.viewer.set_cloud(xyz, rgb, opacity=opacity, scale=scale)
        n = len(xyz)
        has_c = rgb is not None
        self.setWindowTitle(f"recon3d — {os.path.basename(path)}")
        fmt_type = "3D Gaussian Splatting" if is_gaussian else ("RGB 彩色" if has_c else "高度着色")
        self.controls.set_stats(
            f"{os.path.basename(path)}\n{n:,} 个点\n"
            f"{fmt_type}\n"
            f"加载耗时 {seconds:.2f} 秒")
        self._status_label.setText(
            f"已从 {os.path.basename(path)} 加载 {n:,} 个点，耗时 {seconds:.2f} 秒"
            + (" [3DGS]" if is_gaussian else ""))

    def _on_ply_failed(self, path, msg):
        self._progress.setVisible(False)
        self._status_label.setText("加载失败")
        QMessageBox.critical(self, "PLY 加载失败",
                             f"无法读取：\n{path}\n\n{msg}")

    def _on_viewer_status(self, text):
        self._status_label.setText(text)

    # ------------------------------------------------------------------ #
    # Reconstruction panel (optional)
    # ------------------------------------------------------------------ #
    def _try_add_recon_panel(self):
        try:
            from .recon_panel import ReconPanel
        except Exception as e:  # pipeline not available yet — viewer still works
            self._status_label.setText(f"查看器就绪（三维重建不可用：{e}）")
            return
        panel = ReconPanel(self)
        panel.cloudReady.connect(self.load_ply)
        panel.status.connect(self._status_label.setText)
        dock = QDockWidget("三维重建", self)
        dock.setObjectName("ReconDock")
        dock.setWidget(panel)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        self._recon_panel = panel
        panel.set_theme(self._current_theme())

    # ------------------------------------------------------------------ #
    # Drag & drop
    # ------------------------------------------------------------------ #
    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            for u in e.mimeData().urls():
                if u.toLocalFile().lower().endswith(".ply"):
                    e.acceptProposedAction()
                    return
        e.ignore()

    def dropEvent(self, e):
        for u in e.mimeData().urls():
            p = u.toLocalFile()
            if p.lower().endswith(".ply"):
                self.load_ply(p)
                break

    def _about(self):
        QMessageBox.about(
            self, "关于 recon3d",
            "<h3>recon3d</h3>"
            "<p>纯 CPU 三维重建流程，配备高性能 OpenGL 点云查看器。</p>"
            "<p>旋转：左键拖动 &nbsp;•&nbsp; 平移：右键拖动 &nbsp;•&nbsp; "
            "缩放：滚轮 &nbsp;•&nbsp; 重置视角：F</p>")

    def closeEvent(self, e):
        try:
            if self._loader is not None and self._loader.isRunning():
                self._loader.wait(2000)
        except Exception:
            pass
        super().closeEvent(e)
