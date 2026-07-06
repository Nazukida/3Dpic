"""GUI application entry point.

Sets a sane OpenGL surface format *before* the QApplication is created (Qt
requires this), constructs the main window, and optionally opens a PLY passed
on the command line.
"""

from __future__ import annotations

import os
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    # High-DPI: crisp on 4K laptops. Must be set before QApplication.
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication
    from .gl_viewer import default_surface_format
    from .main_window import MainWindow

    # Global default format so every QOpenGLWidget gets a 3.3 core + MSAA ctx.
    QSurfaceFormat.setDefaultFormat(default_surface_format())

    app = QApplication(argv)
    app.setApplicationName("recon3d")
    app.setOrganizationName("recon3d")

    win = MainWindow(app)
    win.show()

    # optional: a .ply given on the command line opens immediately
    for arg in argv[1:]:
        if arg.lower().endswith(".ply") and os.path.isfile(arg):
            win.load_ply(arg)
            break

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
