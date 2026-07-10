"""GUI application entry point.

Sets a sane OpenGL surface format *before* the QApplication is created (Qt
requires this), constructs the main window, and optionally opens a PLY passed
on the command line.
"""

from __future__ import annotations

import os
import sys


def _select_opengl_backend() -> None:
    """Choose the best available OpenGL backend on Windows.

    Qt6 on Windows supports three backends (set via QT_OPENGL env var):
      - "desktop"  — native OpenGL (fastest, needs a real GPU driver)
      - "angle"    — translates OpenGL ES to DirectX (works on RDP / weak GPUs)
      - "software" — Mesa llvmpipe CPU fallback (always works, slow)

    If the user hasn't forced a backend already, we try desktop first by
    probing for a valid pixel format.  If that fails we fall back to ANGLE.
    """
    if sys.platform != "win32":
        return
    if "QT_OPENGL" in os.environ:
        return  # user override — respect it

    # Probe: try to create an actual OpenGL rendering context. This catches
    # cases where ChoosePixelFormat succeeds but wglCreateContext fails (e.g.
    # RDP sessions, VMs without GPU passthrough, broken ICD drivers).
    import ctypes
    from ctypes import wintypes

    gdi32 = ctypes.windll.gdi32
    user32 = ctypes.windll.user32
    try:
        opengl32 = ctypes.windll.opengl32
    except OSError:
        os.environ["QT_OPENGL"] = "angle"
        return

    class PIXELFORMATDESCRIPTOR(ctypes.Structure):
        _fields_ = [
            ("nSize", wintypes.WORD),
            ("nVersion", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("iPixelType", wintypes.BYTE),
            ("cColorBits", wintypes.BYTE),
            ("cRedBits", wintypes.BYTE),
            ("cRedShift", wintypes.BYTE),
            ("cGreenBits", wintypes.BYTE),
            ("cGreenShift", wintypes.BYTE),
            ("cBlueBits", wintypes.BYTE),
            ("cBlueShift", wintypes.BYTE),
            ("cAlphaBits", wintypes.BYTE),
            ("cAlphaShift", wintypes.BYTE),
            ("cAccumBits", wintypes.BYTE),
            ("cAccumRedBits", wintypes.BYTE),
            ("cAccumGreenBits", wintypes.BYTE),
            ("cAccumBlueBits", wintypes.BYTE),
            ("cAccumAlphaBits", wintypes.BYTE),
            ("cDepthBits", wintypes.BYTE),
            ("cStencilBits", wintypes.BYTE),
            ("cAuxBuffers", wintypes.BYTE),
            ("iLayerType", wintypes.BYTE),
            ("bReserved", wintypes.BYTE),
            ("dwLayerMask", wintypes.DWORD),
            ("dwVisibleMask", wintypes.DWORD),
            ("dwDamageMask", wintypes.DWORD),
        ]

    PFD_DRAW_TO_WINDOW = 0x00000004
    PFD_SUPPORT_OPENGL = 0x00000020
    PFD_DOUBLEBUFFER = 0x00000001

    # Create a temporary hidden window for the context probe.
    HWND_MESSAGE = ctypes.c_void_p(-3)  # message-only window (invisible)
    hwnd = user32.CreateWindowExW(
        0, "STATIC", "", 0, 0, 0, 1, 1, HWND_MESSAGE, None, None, None
    )
    if not hwnd:
        os.environ["QT_OPENGL"] = "angle"
        return

    hdc = user32.GetDC(hwnd)
    if not hdc:
        user32.DestroyWindow(hwnd)
        os.environ["QT_OPENGL"] = "angle"
        return

    pfd = PIXELFORMATDESCRIPTOR()
    pfd.nSize = ctypes.sizeof(PIXELFORMATDESCRIPTOR)
    pfd.nVersion = 1
    pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER
    pfd.iPixelType = 0  # PFD_TYPE_RGBA
    pfd.cColorBits = 32
    pfd.cDepthBits = 24

    can_gl = False
    pf = gdi32.ChoosePixelFormat(hdc, ctypes.byref(pfd))
    if pf != 0 and gdi32.SetPixelFormat(hdc, pf, ctypes.byref(pfd)):
        hglrc = opengl32.wglCreateContext(hdc)
        if hglrc:
            can_gl = True
            opengl32.wglDeleteContext(hglrc)

    user32.ReleaseDC(hwnd, hdc)
    user32.DestroyWindow(hwnd)

    if not can_gl:
        os.environ["QT_OPENGL"] = "angle"


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)

    # High-DPI: crisp on 4K laptops. Must be set before QApplication.
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    # Select OpenGL backend before anything Qt-related is imported/created.
    _select_opengl_backend()

    from PySide6.QtGui import QSurfaceFormat
    from PySide6.QtWidgets import QApplication
    from .gl_viewer import default_surface_format
    from .main_window import MainWindow

    # When using ANGLE, request OpenGL ES 3.0 instead of desktop 3.3 Core.
    if os.environ.get("QT_OPENGL") == "angle":
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGLES)
        fmt.setVersion(3, 0)
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setSamples(4)
        fmt.setSwapInterval(1)
        QSurfaceFormat.setDefaultFormat(fmt)
    else:
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
