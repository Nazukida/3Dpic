"""Background workers so the UI never blocks on I/O.

``PlyLoader`` reads a PLY off-thread and hands the arrays back on the GUI
thread via a signal.  Reading a multi-million-point cloud takes long enough
that doing it inline would freeze the window.
"""

from __future__ import annotations

import time
import numpy as np
from PySide6.QtCore import QThread, Signal

from .ply_io import read_ply


class PlyLoader(QThread):
    loaded = Signal(object, object, str, float)   # xyz, rgb, path, seconds
    failed = Signal(str, str)                      # path, error message

    def __init__(self, path: str, max_points: int | None = None, parent=None):
        super().__init__(parent)
        self._path = path
        self._max_points = max_points

    def run(self):
        t0 = time.time()
        try:
            data = read_ply(self._path, max_points=self._max_points)
        except Exception as e:  # surface the failure to the UI, don't crash
            self.failed.emit(self._path, str(e))
            return
        dt = time.time() - t0
        self.loaded.emit(data.xyz, data.rgb, self._path, dt)
