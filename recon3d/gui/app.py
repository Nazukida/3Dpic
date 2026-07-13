"""GUI application entry point.

Sets a sane OpenGL surface format *before* the QApplication is created (Qt
requires this), constructs the main window, and optionally opens a PLY passed
on the command line.
"""

from __future__ import annotations

import os
import sys


def _probe_has_hardware_gl() -> bool:
    """Return True if the primary display exposes an OpenGL pixel format
    backed by a real hardware installable-client-driver (ICD).

    Every Windows machine always exposes the "Microsoft GDI Generic" software
    OpenGL 1.1 renderer; its pixel formats all carry ``PFD_GENERIC_FORMAT``.
    Only a real GPU driver additionally registers *non-generic* (ICD) formats.
    We need a non-generic, double-buffered, depth-capable format to obtain the
    3.x core context our shaders require.

    This enumerates formats directly (no GL context created), which is the
    reliable signal. Probing via a dummy context is *not* reliable:
    ``ChoosePixelFormat`` happily hands back a GDI-Generic 1.1 context even on
    machines that have no ICD at all (e.g. WDDM-only GPUs), and
    ``wglGetProcAddress`` returns NULL for the WGL ARB entry points from such a
    context — both produce false "native GL works" results. A real top-level
    window is used (not a message-only window): message-only windows only see
    the GDI Generic formats even when an ICD is installed.
    """
    import ctypes
    from ctypes import wintypes

    gdi32 = ctypes.windll.gdi32
    user32 = ctypes.windll.user32

    class PIXELFORMATDESCRIPTOR(ctypes.Structure):
        _fields_ = [
            ("nSize", wintypes.WORD), ("nVersion", wintypes.WORD),
            ("dwFlags", wintypes.DWORD), ("iPixelType", wintypes.BYTE),
            ("cColorBits", wintypes.BYTE), ("cRedBits", wintypes.BYTE),
            ("cRedShift", wintypes.BYTE), ("cGreenBits", wintypes.BYTE),
            ("cGreenShift", wintypes.BYTE), ("cBlueBits", wintypes.BYTE),
            ("cBlueShift", wintypes.BYTE), ("cAlphaBits", wintypes.BYTE),
            ("cAlphaShift", wintypes.BYTE), ("cAccumBits", wintypes.BYTE),
            ("cAccumRedBits", wintypes.BYTE), ("cAccumGreenBits", wintypes.BYTE),
            ("cAccumBlueBits", wintypes.BYTE), ("cAccumAlphaBits", wintypes.BYTE),
            ("cDepthBits", wintypes.BYTE), ("cStencilBits", wintypes.BYTE),
            ("cAuxBuffers", wintypes.BYTE), ("iLayerType", wintypes.BYTE),
            ("bReserved", wintypes.BYTE), ("dwLayerMask", wintypes.DWORD),
            ("dwVisibleMask", wintypes.DWORD), ("dwDamageMask", wintypes.DWORD),
        ]

    PFD_DOUBLEBUFFER = 0x00000001
    PFD_SUPPORT_OPENGL = 0x00000020
    PFD_GENERIC_FORMAT = 0x00000040
    WS_POPUP = 0x80000000

    hwnd = user32.CreateWindowExW(0, "STATIC", "glprobe", WS_POPUP,
                                  0, 0, 2, 2, None, None, None, None)
    if not hwnd:
        return True  # cannot probe; assume hardware (preserve prior default)
    hdc = user32.GetDC(hwnd)
    if not hdc:
        user32.DestroyWindow(hwnd)
        return True
    try:
        max_pf = gdi32.DescribePixelFormat(hdc, 1, 0, None) or 0
        for i in range(1, max_pf + 1):
            pfd = PIXELFORMATDESCRIPTOR()
            pfd.nSize = ctypes.sizeof(PIXELFORMATDESCRIPTOR)
            gdi32.DescribePixelFormat(hdc, i, ctypes.sizeof(PIXELFORMATDESCRIPTOR),
                                      ctypes.byref(pfd))
            if (pfd.dwFlags & PFD_SUPPORT_OPENGL
                    and not (pfd.dwFlags & PFD_GENERIC_FORMAT)
                    and (pfd.dwFlags & PFD_DOUBLEBUFFER)
                    and pfd.cDepthBits >= 16):
                return True
        return False
    finally:
        user32.ReleaseDC(hwnd, hdc)
        user32.DestroyWindow(hwnd)


def _select_opengl_backend() -> None:
    """Pick the Qt OpenGL backend (``QT_OPENGL`` env var) before Qt starts.

    Three backends are possible on Windows:
      - "desktop"  — native OpenGL via opengl32.dll. Needs a real GPU driver
                     (an OpenGL ICD). Fastest.
      - "angle"    — OpenGL ES translated to DirectX. Only works if ANGLE DLLs
                     (libGLESv2.dll / libEGL.dll) are shipped alongside Qt —
                     PySide6 wheels do NOT bundle them.
      - "software" — Mesa llvmpipe via the bundled opengl32sw.dll. Always
                     works, but CPU-bound.

    Qt's default *dynamic* selection is unreliable on drivers without an
    OpenGL ICD (some WDDM-only GPUs, RDP without GPU passthrough): it loads
    opengl32sw.dll but then fails pixel-format negotiation, producing the
    "Attempted to use GDI functions with a non-opengl32.dll library" /
    "Unable find a suitable pixel format" / "Failed to create context" storm.
    So we decide explicitly: a hardware ICD is present -> "desktop", otherwise
    -> "software" (the only path that yields a real modern GL context here).
    "angle" is never auto-selected (no DLLs shipped) but stays available as a
    manual ``QT_OPENGL=angle`` override for environments that provide them.
    """
    if sys.platform != "win32":
        return
    if "QT_OPENGL" in os.environ:
        return  # user override — respect it
    os.environ["QT_OPENGL"] = "desktop" if _probe_has_hardware_gl() else "software"


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

    backend = os.environ.get("QT_OPENGL")
    if backend == "software":
        # Mesa llvmpipe (opengl32sw.dll) ships a desktop GL 3.0 context but is
        # picky: pinning a 3.3-core profile OR requesting MSAA makes it fail
        # pixel-format negotiation. Leave version & samples unset — we get a
        # usable GL 3.0 context, and the viewer detects llvmpipe and switches
        # to its fast software render path on its own.
        fmt = QSurfaceFormat()
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setSwapInterval(0)   # a CPU rasteriser gains nothing from vsync
        QSurfaceFormat.setDefaultFormat(fmt)
    elif backend == "angle":
        # OpenGL ES via ANGLE (only when the user has supplied ANGLE DLLs).
        fmt = QSurfaceFormat()
        fmt.setRenderableType(QSurfaceFormat.OpenGLES)
        fmt.setVersion(3, 0)
        fmt.setDepthBufferSize(24)
        fmt.setStencilBufferSize(8)
        fmt.setSwapInterval(1)
        QSurfaceFormat.setDefaultFormat(fmt)
    else:
        # desktop — real GPU: global default so every QOpenGLWidget gets a
        # 3.3 core + 8x MSAA context.
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
