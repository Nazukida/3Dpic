"""Background workers so the UI never blocks on I/O.

``ModelLoader`` (alias ``PlyLoader``) reads any supported 3D-model file off
thread via :func:`recon3d.gui.model_io.read_model` and hands the arrays back on
the GUI thread via a signal.  Reading a multi-million-point cloud takes long
enough that doing it inline would freeze the window.
"""

from __future__ import annotations

import time
import numpy as np
from PySide6.QtCore import QThread, Signal

from .model_io import read_model, ModelReadError


class ModelLoader(QThread):
    # xyz, rgb, path, seconds, opacity, scale, is_gaussian, source(fmt)
    loaded = Signal(object, object, str, float, object, object, bool, str)
    failed = Signal(str, str)                      # path, error message

    def __init__(self, path: str, max_points: int | None = None, parent=None):
        super().__init__(parent)
        self._path = path
        self._max_points = max_points

    def run(self):
        t0 = time.time()
        try:
            data = read_model(self._path, max_points=self._max_points)
        except (ModelReadError, Exception) as e:  # surface failure to the UI
            self.failed.emit(self._path, str(e))
            return
        dt = time.time() - t0
        self.loaded.emit(data.xyz, data.rgb, self._path, dt,
                         data.opacity, data.scale, data.is_gaussian,
                         data.source or "模型")


# Backward-compatible alias (older code / recon_panel refer to PlyLoader).
PlyLoader = ModelLoader
