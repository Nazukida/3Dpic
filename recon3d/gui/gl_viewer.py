"""High-performance OpenGL point-cloud viewer.

Built entirely on Qt's own OpenGL wrappers (``QOpenGLShaderProgram``,
``QOpenGLBuffer``, ``QOpenGLVertexArrayObject`` + ``QOpenGLFunctions``) so it
needs no PyOpenGL dependency.

Performance strategy
--------------------
* Point data lives in a single interleaved VBO (``pos vec3`` + ``color vec3``)
  uploaded once.  Nothing is touched per-frame on the CPU.
* Points are **shuffled on upload** so drawing a contiguous prefix yields a
  uniform random subsample.  During camera interaction we draw only that
  prefix (LOD); when the camera settles a timer triggers a full-detail redraw.
  This keeps dragging smooth even with tens of millions of points.
* Points are drawn as **sphere imposters**: the fragment shader reconstructs a
  spherical normal from ``gl_PointCoord`` and applies cheap Lambert shading, so
  even colour-only clouds read as 3D.  Toggleable to flat round points.

Interaction: LMB orbit, RMB/MMB pan, wheel zoom, ``F`` / double-click re-fits.
"""

from __future__ import annotations

import sys
import time

import numpy as np

from PySide6.QtCore import Qt, QTimer, QPoint, Signal
from PySide6.QtGui import QVector3D, QMatrix4x4, QSurfaceFormat
from PySide6.QtOpenGL import (
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLBuffer,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget

# Platform-specific raw GL access for functions not in QOpenGLFunctions.
# On Windows we use the system opengl32.dll directly — but ONLY when we're on
# the real desktop backend. Under "software" the live context is Mesa's
# opengl32sw.dll and under "angle" it's libGLESv2.dll; calling through the
# system opengl32.dll then is a different library than the one backing the
# context, which produces the noisy "Attempted to use GDI functions with a
# non-opengl32.dll library" warnings (and is the wrong driver anyway). On
# other platforms we skip the calls entirely (graceful degradation — the
# depth-mask tweak is non-critical and only helps semi-transparent splats).
_gl32 = None
if sys.platform == "win32":
    import os as _os
    import ctypes
    if _os.environ.get("QT_OPENGL", "desktop") == "desktop":
        try:
            _gl32 = ctypes.windll.opengl32
        except Exception:
            pass

# GL enums we use that QOpenGLFunctions doesn't expose as named constants.
GL_PROGRAM_POINT_SIZE = 0x8642
GL_DEPTH_TEST = 0x0B71
GL_BLEND = 0x0BE2
GL_SRC_ALPHA = 0x0302
GL_ONE_MINUS_SRC_ALPHA = 0x0303
GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_FLOAT = 0x1406
GL_POINTS = 0x0000
GL_LINES = 0x0001
GL_TRIANGLES = 0x0004
GL_MULTISAMPLE = 0x809D
GL_LINE_SMOOTH = 0x0B20
GL_POINT_SPRITE = 0x8861


# --------------------------------------------------------------------------- #
# Shaders
# --------------------------------------------------------------------------- #
_POINT_VS = """
in vec3 in_pos;
in vec3 in_col;
in float in_opacity;
in float in_scale;

uniform mat4 u_mvp;
uniform mat4 u_mv;
uniform float u_point_size;   // base size in px at unit distance
uniform float u_attenuation;  // 0 = constant size, 1 = perspective size
uniform float u_viewport_h;   // for size scaling
uniform float u_max_point;    // clamp ceiling (small on software renderers)
uniform float u_fov;          // vertical FOV in degrees
uniform vec3  u_tint;
uniform int   u_has_scale;    // 1 = per-point scale available (Gaussian)
uniform int   u_quality;      // 0=perf, 1=standard, 2=high

out vec3 v_col;
out float v_depth;
out float v_alpha;

void main() {
    vec4 mv = u_mv * vec4(in_pos, 1.0);
    v_depth = -mv.z;
    gl_Position = u_mvp * vec4(in_pos, 1.0);

    // Perspective point size with proper projection formula.
    float dist = max(v_depth, 0.0001);
    float half_tan = tan(radians(u_fov * 0.5));
    float proj_scale = (u_viewport_h * 0.5) / (dist * half_tan);
    float atten = mix(1.0, proj_scale, u_attenuation);
    float size = u_point_size;
    if (u_has_scale == 1) {
        size *= in_scale;
    }
    gl_PointSize = clamp(size * atten, 1.0, u_max_point);

    // sRGB -> linear conversion for quality mode (lighting in linear space)
    vec3 tinted = in_col * u_tint;
    if (u_quality == 2) {
        v_col = pow(max(tinted, vec3(0.0)), vec3(2.2));
    } else {
        v_col = tinted;
    }
    v_alpha = in_opacity;
}
"""

_POINT_FS = """
in vec3 v_col;
in float v_depth;
in float v_alpha;

uniform int  u_shaded;   // 1 = sphere imposter shading, 0 = flat round
uniform int  u_fast;     // 1 = flat OPAQUE square (no discard/blend) - cheapest
uniform int  u_quality;  // 0=perf, 1=standard, 2=high
uniform vec3 u_light_dir;
uniform vec3 u_bg_sky;       // theme top color (for hemisphere ambient)
uniform vec3 u_bg_ground;    // theme bottom color (for hemisphere ambient)
uniform float u_scene_radius;

out vec4 frag;

void main() {
    // Fast path: flat opaque square — cheapest for CPU rasterisers.
    if (u_fast == 1) {
        frag = vec4(v_col, v_alpha);
        return;
    }

    vec2 pc = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(pc, pc);

    // Fallback for broken gl_PointCoord (always 0,0).
    if (gl_PointCoord == vec2(0.0)) {
        frag = vec4(v_col, v_alpha);
        return;
    }

    // Anti-aliased disc boundary (replaces hard discard).
    float d = sqrt(r2);
    float fw = fwidth(d);
    if (d > 1.0 + fw) discard;
    float disc_aa = 1.0 - smoothstep(1.0 - fw * 2.0, 1.0, d);

    vec3 col = v_col;

    if (u_quality == 2) {
        // --- HIGH QUALITY: Blinn-Phong + hemisphere ambient + EDL ---
        vec3 n = vec3(pc.x, -pc.y, sqrt(max(0.0, 1.0 - r2)));
        vec3 L = normalize(u_light_dir);

        // Hemisphere ambient: blend sky/ground by normal Y
        float hem = n.y * 0.5 + 0.5;
        vec3 ambient_col = mix(u_bg_ground, u_bg_sky, hem);
        float ambient_str = 0.30;

        // Primary diffuse (Lambert)
        float diff = max(dot(n, L), 0.0);

        // Secondary fill light (softer, from opposite side)
        vec3 L2 = normalize(vec3(-0.3, -0.2, -0.5));
        float diff2 = max(dot(n, L2), 0.0) * 0.30;

        // Blinn-Phong specular (view is along +Z in tangent space of the sprite)
        vec3 V = vec3(0.0, 0.0, 1.0);
        vec3 H = normalize(L + V);
        float spec = pow(max(dot(n, H), 0.0), 32.0) * 0.15;

        // Fresnel rim (Schlick approximation)
        float fresnel = pow(1.0 - max(dot(n, V), 0.0), 3.0) * 0.08;

        // Compose lighting
        col = col * (ambient_str * ambient_col + (1.0 - ambient_str) * (diff + diff2)) + vec3(spec + fresnel);

        // Eye-Dome Lighting approximation: darken edges at depth
        float edl = 1.0 - 0.12 * smoothstep(0.3, 0.9, r2) * clamp(v_depth / max(u_scene_radius, 0.01), 0.0, 1.0);
        col *= edl;

        // Linear -> sRGB output
        col = pow(max(col, vec3(0.0)), vec3(1.0 / 2.2));

    } else if (u_shaded == 1) {
        // --- STANDARD: Lambert sphere imposter (original quality) ---
        vec3 n = vec3(pc.x, -pc.y, sqrt(max(0.0, 1.0 - r2)));
        float diff = max(dot(n, normalize(u_light_dir)), 0.0);
        float ambient = 0.35;
        col = col * (ambient + (1.0 - ambient) * diff);
        col += 0.06 * pow(1.0 - n.z, 2.0);
    }

    // Improved alpha falloff with anti-aliased edges
    float core = exp(-2.5 * r2);
    float edge_soft = 1.0 - smoothstep(0.65, 1.0, d);
    float alpha = v_alpha * mix(edge_soft, core, 0.6) * disc_aa;

    frag = vec4(col, alpha);
}
"""

_LINE_VS = """
in vec3 in_pos;
in vec3 in_col;
uniform mat4 u_mvp;
out vec3 v_col;
void main() { gl_Position = u_mvp * vec4(in_pos, 1.0); v_col = in_col; }
"""

_LINE_FS = """
in vec3 v_col;
uniform float u_alpha;
out vec4 frag;
void main() { frag = vec4(v_col, u_alpha); }
"""

_BG_VS = """
const vec2 verts[3] = vec2[3](vec2(-1.0,-1.0), vec2(3.0,-1.0), vec2(-1.0,3.0));
out vec2 v_uv;
void main() {
    vec2 p = verts[gl_VertexID];
    v_uv = p * 0.5 + 0.5;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

_BG_FS = """
in vec2 v_uv;
uniform vec3 u_top;
uniform vec3 u_bottom;
out vec4 frag;
void main() { frag = vec4(mix(u_bottom, u_top, v_uv.y), 1.0); }
"""


_ES_HEADER = "#version 300 es\nprecision highp float;\nprecision highp int;\n"
_DESKTOP_HEADER = "#version 330 core\n"
_LEGACY_HEADER = "#version 130\n"


def _candidate_headers(is_es: bool) -> list[str]:
    """Version preambles to try, best-first, for the live context.

    Drivers vary wildly: ANGLE exposes GLSL 3.00 es, native desktop gives
    3.30 core, and Mesa/llvmpipe (software fallback) reports a desktop context
    that only accepts up to GLSL 1.30. The shader *bodies* below avoid
    ``layout(location=)`` (attribute locations are bound manually) so the same
    source compiles under all three; we just try headers until one links.
    """
    order = [_ES_HEADER, _DESKTOP_HEADER, _LEGACY_HEADER] if is_es \
        else [_DESKTOP_HEADER, _ES_HEADER, _LEGACY_HEADER]
    return order


def _viridis(t: np.ndarray) -> np.ndarray:
    """Compact viridis approximation (polynomial) -> (N,3) float in 0..1."""
    t = np.clip(t, 0.0, 1.0)[:, None]
    # coefficients of a cubic fit per channel (good enough for a height ramp)
    c0 = np.array([0.2777, 0.0054, 0.3341])
    c1 = np.array([0.1050, 1.4046, 1.3845])
    c2 = np.array([-0.3308, 0.2148, -4.7716])
    c3 = np.array([-4.6342, -5.7991, -6.1689])
    c4 = np.array([6.2282, 14.1799, 56.6905])
    c5 = np.array([4.7763, -13.7451, -65.3529])
    c6 = np.array([-5.4354, 4.6459, 26.3124])
    return np.clip(c0 + t*(c1 + t*(c2 + t*(c3 + t*(c4 + t*(c5 + t*c6))))), 0, 1)


class GLViewer(QOpenGLWidget):
    """A point-cloud canvas. Feed it xyz+rgb via :meth:`set_cloud`."""

    status = Signal(str)          # short status strings (fps / point counts)
    rendererReady = Signal(bool)  # emitted once GL is up: True if software GL

    def __init__(self, parent=None, theme=None):
        super().__init__(parent)
        self.setMinimumSize(320, 240)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMouseTracking(True)

        # --- data (CPU side, pending upload) ---
        self._interleaved: np.ndarray | None = None   # (N,6) float32
        self._n_points = 0
        self._pending_upload = False
        self._bbox_min = np.zeros(3)
        self._bbox_max = np.ones(3)
        self._center = np.zeros(3)
        self._radius = 1.0

        # --- camera ---
        self._yaw = 0.6
        self._pitch = 0.5
        self._dist = 4.0
        self._target = QVector3D(0, 0, 0)
        self._fov = 45.0
        self._last_mouse = QPoint()

        # --- render options ---
        self.point_size = 2.0
        self.attenuation = 0.0          # 0 = constant screen px (slider = px size); >0 adds perspective growth
        self.shaded = True
        self.show_grid = True
        self.show_axes = True
        self.brightness = 1.0
        self._colormap_when_uncolored = True
        self._has_per_point_scale = False
        self.quality_level = 2          # 0=perf, 1=standard, 2=high

        # --- performance / LOD (finalised in initializeGL once we know the
        # renderer; these are safe fallbacks for hardware GPUs) ---
        self.fast_mode = False          # flat opaque square points, no blend
        self._interactive_budget = 2_000_000   # max points drawn while moving
        self._static_budget = 0                # max points at idle (0 = all)
        self._max_point_px = 12.0
        self._true_extent = 1.0                # raw max axis extent (sizes the far plane)
        self._soft = False
        self._renderer_name = ""

        # --- LOD interaction state ---
        self._interacting = False
        self._idle_timer = QTimer(self)
        self._idle_timer.setSingleShot(True)
        self._idle_timer.setInterval(180)
        self._idle_timer.timeout.connect(self._on_idle)

        # --- adaptive frame-rate-feedback LOD ---
        # A single self-tuning controller picks how many of the shuffled points
        # to draw each frame. It targets a *frame time*, not a fixed count:
        #   - while dragging (interacting): ~33 ms  -> ~30 fps, smooth orbit
        #   - once settled (idle):           a heavier one-shot budget so the
        #     crispest affordable view appears without a long freeze.
        # Because points are shuffled on upload, drawing a prefix is always a
        # uniform random subsample, so fewer points still reads as the scene.
        self._target_fps_ms = 33.0          # interactive frame-time target
        self._static_target_ms = 300.0      # idle settle frame-time target
        self._last_paint_ms = 0.0           # measured duration of last paintGL
        # Exponentially-smoothed cost PER POINT (ms). Rendering cost is roughly
        # linear in point count across our operating range (~0.7 µs/pt on
        # llvmpipe), so once we know the per-point cost we can predict the
        # point count that hits any target frame time directly — no feedback
        # oscillation, converges in ~2 frames. Robust to jitter via the EMA.
        self._cpp_ema = 0.0
        self._adaptive_n = 0                # current adaptive draw count
        self._refine_pending = False        # static-refinement pump active

        # --- theme ---
        from . import themes as _t
        self._theme = theme or _t.PRESETS[_t.DEFAULT_THEME]

        # GL objects (created in initializeGL)
        self._prog = None
        self._line_prog = None
        self._bg_prog = None
        self._vbo = None
        self._vao = None
        self._grid_vbo = None
        self._grid_vao = None
        self._grid_count = 0
        self._axis_vbo = None
        self._axis_vao = None
        self._empty_vao = None
        self.gl = None
        self._gl_ready = False

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def set_theme(self, theme):
        self._theme = theme
        if self._gl_ready:
            self._rebuild_grid()
            self.update()

    def set_cloud(self, xyz: np.ndarray, rgb: np.ndarray | None,
                  opacity: np.ndarray | None = None,
                  scale: np.ndarray | None = None):
        """Replace the displayed cloud.

        ``xyz`` (N,3); ``rgb`` (N,3) uint8 or None;
        ``opacity`` (N,) float32 [0,1] or None (defaults to 1.0);
        ``scale`` (N,) float32 or None (defaults to 1.0, used as point size multiplier).
        """
        xyz = np.ascontiguousarray(xyz, dtype=np.float32)
        n = len(xyz)
        if n == 0:
            self._interleaved = None
            self._n_points = 0
            self._has_per_point_scale = False
            self._pending_upload = True
            if self._gl_ready:
                self.update()
            return

        # Bounding geometry for framing / grid / camera target. Use a ROBUST
        # (0.5/99.5-percentile) box instead of the raw min/max: reconstructions
        # routinely carry a few % of stray outlier points far outside the model
        # body (e.g. dense.ply: 2.7% of points sit >2x beyond the body, inflating
        # the raw box ~2.2x and shrinking the framed cloud to a centred blob you
        # have to zoom into). The true extent is kept separately only to size
        # the far clip plane so those outliers are never z-clipped.
        true_min = xyz.min(axis=0)
        true_max = xyz.max(axis=0)
        self._true_extent = float(np.max(true_max - true_min)) or 1.0
        lo = np.percentile(xyz, 0.5, axis=0)
        hi = np.percentile(xyz, 99.5, axis=0)
        if np.any(hi - lo <= 0):           # degenerate (flat / too few points)
            lo, hi = true_min, true_max
        self._bbox_min = lo
        self._bbox_max = hi
        self._center = (lo + hi) * 0.5
        self._radius = float(np.linalg.norm(hi - lo) * 0.5) or 1.0

        # colours
        if rgb is not None and len(rgb) == n:
            col = np.ascontiguousarray(rgb, dtype=np.float32) / 255.0
        elif self._colormap_when_uncolored:
            z = xyz[:, 2].astype(np.float32)
            zmin, zmax = float(z.min()), float(z.max())
            t = (z - zmin) / (zmax - zmin + 1e-9)
            col = _viridis(t).astype(np.float32)
        else:
            col = np.ones((n, 3), np.float32)

        # per-point opacity (default fully opaque)
        if opacity is not None and len(opacity) == n:
            alpha = np.ascontiguousarray(opacity, dtype=np.float32)
        else:
            alpha = np.ones(n, dtype=np.float32)

        # per-point scale (relative size multiplier, default 1.0)
        if scale is not None and len(scale) == n:
            # Normalise scale to a reasonable range for point sizes.
            # Use median as reference so outliers don't crush the majority.
            median_s = float(np.median(scale))
            if median_s > 0:
                rel_scale = (scale / median_s).astype(np.float32)
            else:
                rel_scale = np.ones(n, dtype=np.float32)
            # Clamp to prevent extremes
            rel_scale = np.clip(rel_scale, 0.1, 5.0)
            self._has_per_point_scale = True
        else:
            rel_scale = np.ones(n, dtype=np.float32)
            self._has_per_point_scale = False

        # Filter out very transparent Gaussians (opacity < threshold)
        # to reduce clutter and improve visual clarity
        if opacity is not None:
            visible = alpha > 0.05
            if not np.all(visible):
                keep = np.where(visible)[0]
                xyz = xyz[keep]
                col = col[keep]
                alpha = alpha[keep]
                rel_scale = rel_scale[keep]
                n = len(xyz)

        # shuffle so a prefix is a uniform subsample (LOD)
        rng = np.random.default_rng(12345)
        perm = rng.permutation(n)

        # Interleaved layout: pos(3) + col(3) + opacity(1) + scale(1) = 8 floats
        inter = np.empty((n, 8), dtype=np.float32)
        inter[:, 0:3] = xyz[perm]
        inter[:, 3:6] = col[perm]
        inter[:, 6] = alpha[perm]
        inter[:, 7] = rel_scale[perm]
        self._interleaved = np.ascontiguousarray(inter)
        self._n_points = n
        self._pending_upload = True

        # New cloud: reset the adaptive LOD controller so it re-tunes to this
        # cloud's size and the current renderer's fill-rate.
        self._adaptive_n = 0
        self._last_paint_ms = 0.0
        self._cpp_ema = 0.0
        self._refine_pending = False

        self.reset_view()
        if self._gl_ready:
            self.update()
        self.status.emit(f"{n:,} 个点")

    def reset_view(self):
        self._target = QVector3D(*[float(v) for v in self._center])
        self._yaw = 0.6
        self._pitch = 0.5
        # Frame the (robust) bounding box so it fills the viewport without
        # clipping. _fit_distance projects the 8 box corners onto the camera
        # right/up plane and sizes the distance so the larger projected extent
        # maps to ~92% of the matching half-FOV — so the cloud fills the view
        # regardless of its shape, and stray outlier points (excluded from the
        # robust box in set_cloud) can't shrink the framed model.
        self._dist = self._fit_distance(0.92)
        if self._gl_ready:
            self.update()

    def _fit_distance(self, fill: float = 0.92) -> float:
        """Camera distance so the bounding box fills ``fill`` of the viewport.

        The 8 box corners are projected onto the camera's right/up plane; those
        perpendicular components are independent of distance, so the distance is
        sized directly: ``dist = max(perp_y/(fill*tan(fov/2)),
        perp_x/(fill*tan(fov/2)*aspect))``. Taking the binding of the vertical
        and horizontal fits keeps the whole box on screen (no clipping) at a
        consistent fill fraction for any cloud shape.
        """
        half = (np.asarray(self._bbox_max, dtype=np.float64)
                - np.asarray(self._bbox_min, dtype=np.float64)) * 0.5
        cp, sp = np.cos(self._pitch), np.sin(self._pitch)
        cy, sy = np.cos(self._yaw), np.sin(self._yaw)
        d = np.array([cp * sy, sp, cp * cy], dtype=np.float64)
        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(d, world_up)
        right /= (np.linalg.norm(right) + 1e-12)
        up = np.cross(right, d)
        mx = my = 0.0
        for sx in (-1.0, 1.0):
            for syy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    c = np.array([sx * half[0], syy * half[1], sz * half[2]])
                    mx = max(mx, abs(float(np.dot(c, right))))
                    my = max(my, abs(float(np.dot(c, up))))
        tan_h = np.tan(np.radians(self._fov) * 0.5)
        w, h = self.width(), self.height()
        aspect = (w / h) if (w > 0 and h > 0) else 1.0
        dist_y = my / max(fill * tan_h, 1e-6)
        dist_x = mx / max(fill * tan_h * aspect, 1e-6)
        # floor: a degenerate (near-zero-extent) box must not collapse distance.
        return float(max(dist_y, dist_x, float(self._radius) * 0.25))

    def set_point_size(self, v: float):
        self.point_size = float(v); self.update()

    def set_attenuation(self, v: float):
        self.attenuation = float(np.clip(v, 0.0, 1.0)); self.update()

    def set_shaded(self, on: bool):
        self.shaded = bool(on); self.update()

    def set_fast_mode(self, on: bool):
        """Toggle the flat-opaque performance path (auto-on for software GL)."""
        self.fast_mode = bool(on); self.update()

    def set_brightness(self, v: float):
        self.brightness = float(v); self.update()

    def set_show_grid(self, on: bool):
        self.show_grid = bool(on); self.update()

    def set_show_axes(self, on: bool):
        self.show_axes = bool(on); self.update()

    def set_quality_level(self, level: int):
        """Set render quality: 0=performance, 1=standard, 2=high."""
        level = int(np.clip(level, 0, 2))
        self.quality_level = level
        if level == 0:
            self.fast_mode = True
            self.shaded = False
        elif level == 1:
            self.fast_mode = False
            self.shaded = True
        else:
            self.fast_mode = False
            self.shaded = True
        self.update()

    # ------------------------------------------------------------------ #
    # GL lifecycle
    # ------------------------------------------------------------------ #
    def initializeGL(self):
        from PySide6.QtGui import QOpenGLContext
        ctx = QOpenGLContext.currentContext()
        self.gl = ctx.functions()
        self._is_es = ctx.isOpenGLES()
        self.gl.glClearColor(0.05, 0.06, 0.09, 1.0)

        # Detect a software rasteriser (llvmpipe/softpipe/swrast, VMware SVGA).
        # These have NO GPU: fill-rate is the bottleneck, so we default to a
        # cheap "performance" render path and much tighter level-of-detail
        # budgets. On real hardware we keep the pretty, high-budget defaults.
        try:
            renderer = str(self.gl.glGetString(0x1F01) or "").lower()  # GL_RENDERER
        except Exception:
            renderer = ""
        self._soft = any(s in renderer for s in
                         ("llvmpipe", "softpipe", "swrast", "software", "vmware"))
        self._renderer_name = renderer
        if self._soft:
            self.fast_mode = True
            self.quality_level = 0
            # The adaptive LOD controller (paintGL / _adaptive_draw_n) targets a
            # frame *time*, so these are just hard ceilings. Default point
            # sizing is constant screen-space (attenuation 0, point_size 2 ->
            # 2 px splats): 2 px is ~25x cheaper to fill than the old 10 px, so
            # the controller draws far more points per frame budget (~1 M while
            # dragging, 1.5 M idle) — denser AND clearer, because small splats
            # stop the cloud reading as one overdrawn blob. The time target
            # still self-limits if the user raises the point size.
            self._interactive_budget = 1_000_000
            self._static_budget = 1_500_000
            self._max_point_px = 6.0
        else:
            self._interactive_budget = 8_000_000
            self._static_budget = 0             # 0 = draw everything at idle
            self._max_point_px = 12.0

        try:
            self._prog = self._make_program(_POINT_VS, _POINT_FS)
            self._line_prog = self._make_program(_LINE_VS, _LINE_FS)
            self._bg_prog = self._make_program(_BG_VS, _BG_FS)
        except Exception as e:
            # Qt swallows exceptions raised from initializeGL; make them loud.
            import traceback
            traceback.print_exc()
            self._init_error = str(e)
            return

        # point cloud VAO/VBO
        self._vao = QOpenGLVertexArrayObject(self)
        self._vao.create()
        self._vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        self._vbo.create()
        self._vbo.setUsagePattern(QOpenGLBuffer.StaticDraw)

        # background needs a bound (empty) VAO in core profile
        self._empty_vao = QOpenGLVertexArrayObject(self)
        self._empty_vao.create()

        # grid + axes
        self._grid_vao = QOpenGLVertexArrayObject(self); self._grid_vao.create()
        self._grid_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer); self._grid_vbo.create()
        self._axis_vao = QOpenGLVertexArrayObject(self); self._axis_vao.create()
        self._axis_vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer); self._axis_vbo.create()

        self._gl_ready = True
        self._rebuild_grid()
        if self._interleaved is not None:
            self._pending_upload = True

        # Let the UI sync its "performance mode" checkbox and inform the user
        # when we fell back to a software renderer (so slowness is explained).
        self.rendererReady.emit(self._soft)
        if self._soft:
            self.status.emit("检测到软件渲染（无 GPU），已自动启用性能模式")

    def _make_program(self, vs: str, fs: str) -> QOpenGLShaderProgram:
        # Once one header has compiled, reuse it for every program.
        headers = ([self._shader_header] if getattr(self, "_shader_header", None)
                   else _candidate_headers(getattr(self, "_is_es", False)))
        last_log = ""
        for header in headers:
            prog = QOpenGLShaderProgram(self)
            ok_v = prog.addShaderFromSourceCode(QOpenGLShader.Vertex, header + vs)
            if not ok_v:
                last_log = f"vertex:\n{prog.log()}"
                continue
            ok_f = prog.addShaderFromSourceCode(QOpenGLShader.Fragment, header + fs)
            if not ok_f:
                last_log = f"fragment:\n{prog.log()}"
                continue
            # Bind attribute locations explicitly (shaders omit layout() so the
            # same source compiles on GLSL 1.30). Harmless if a name is unused.
            prog.bindAttributeLocation("in_pos", 0)
            prog.bindAttributeLocation("in_col", 1)
            prog.bindAttributeLocation("in_opacity", 2)
            prog.bindAttributeLocation("in_scale", 3)
            if not prog.link():
                last_log = f"link:\n{prog.log()}"
                continue
            self._shader_header = header   # remember the winner
            return prog
        raise RuntimeError(f"no compatible GLSL version; last error:\n{last_log}")

    def _upload_cloud(self):
        if self._interleaved is None:
            self._n_gpu = 0
            self._pending_upload = False
            return
        data = self._interleaved
        self._vao.bind()
        self._vbo.bind()
        nbytes = data.nbytes
        self._vbo.allocate(data.tobytes(), nbytes)
        stride = 8 * 4  # pos(3) + col(3) + opacity(1) + scale(1) = 8 floats
        # location 0: position (3 floats @ offset 0)
        self._prog.bind()
        self._prog.enableAttributeArray(0)
        self._prog.setAttributeBuffer(0, GL_FLOAT, 0, 3, stride)
        # location 1: colour (3 floats @ offset 12)
        self._prog.enableAttributeArray(1)
        self._prog.setAttributeBuffer(1, GL_FLOAT, 3 * 4, 3, stride)
        # location 2: opacity (1 float @ offset 24)
        self._prog.enableAttributeArray(2)
        self._prog.setAttributeBuffer(2, GL_FLOAT, 6 * 4, 1, stride)
        # location 3: scale (1 float @ offset 28)
        self._prog.enableAttributeArray(3)
        self._prog.setAttributeBuffer(3, GL_FLOAT, 7 * 4, 1, stride)
        self._prog.release()
        self._vbo.release()
        self._vao.release()
        self._n_gpu = len(data)
        self._pending_upload = False

    def _rebuild_grid(self):
        if not self._gl_ready:
            return
        from . import themes as _t
        # Ground grid on the XZ plane, seated at the cloud's floor (bbox min Y)
        # so it reads as a floor beneath the scene instead of slicing through
        # its middle (which hid it entirely for centred clouds).
        r = max(self._radius, 1.0)
        step = _nice_step(r / 5.0)
        n = 10
        gc = _t.rgb_floats(self._theme.view_grid)
        gy = float(self._bbox_min[1])
        cx = float(self._center[0])
        cz = float(self._center[2])
        lines = []
        extent = step * n
        for i in range(-n, n + 1):
            d = i * step
            lines += [(cx - extent, gy, cz + d, *gc),
                      (cx + extent, gy, cz + d, *gc)]
            lines += [(cx + d, gy, cz - extent, *gc),
                      (cx + d, gy, cz + extent, *gc)]
        grid = np.array(lines, dtype=np.float32)
        self._grid_count = len(grid)
        self._grid_vao.bind(); self._grid_vbo.bind()
        self._grid_vbo.allocate(grid.tobytes(), grid.nbytes)
        self._line_prog.bind()
        self._line_prog.enableAttributeArray(0)
        self._line_prog.setAttributeBuffer(0, GL_FLOAT, 0, 3, 6 * 4)
        self._line_prog.enableAttributeArray(1)
        self._line_prog.setAttributeBuffer(1, GL_FLOAT, 3 * 4, 3, 6 * 4)
        self._line_prog.release()
        self._grid_vbo.release(); self._grid_vao.release()

        # axis gizmo (X red, Y green, Z blue), seated at the grid origin on the
        # cloud floor, length ~ two grid steps
        L = step * 2
        ax = _t.rgb_floats(self._theme.view_axis_x)
        ay = _t.rgb_floats(self._theme.view_axis_y)
        az = _t.rgb_floats(self._theme.view_axis_z)
        axis = np.array([
            (cx, gy, cz, *ax), (cx + L, gy, cz, *ax),
            (cx, gy, cz, *ay), (cx, gy + L, cz, *ay),
            (cx, gy, cz, *az), (cx, gy, cz + L, *az),
        ], dtype=np.float32)
        self._axis_vao.bind(); self._axis_vbo.bind()
        self._axis_vbo.allocate(axis.tobytes(), axis.nbytes)
        self._line_prog.bind()
        self._line_prog.enableAttributeArray(0)
        self._line_prog.setAttributeBuffer(0, GL_FLOAT, 0, 3, 6 * 4)
        self._line_prog.enableAttributeArray(1)
        self._line_prog.setAttributeBuffer(1, GL_FLOAT, 3 * 4, 3, 6 * 4)
        self._line_prog.release()
        self._axis_vbo.release(); self._axis_vao.release()

    def resizeGL(self, w, h):
        self.gl.glViewport(0, 0, w, max(1, h))

    # ------------------------------------------------------------------ #
    # Camera math
    # ------------------------------------------------------------------ #
    def _eye(self) -> QVector3D:
        cp = np.cos(self._pitch); sp = np.sin(self._pitch)
        cy = np.cos(self._yaw); sy = np.sin(self._yaw)
        dir_ = QVector3D(cp * sy, sp, cp * cy)
        return self._target + dir_ * self._dist

    def _view_matrix(self) -> QMatrix4x4:
        m = QMatrix4x4()
        m.lookAt(self._eye(), self._target, QVector3D(0, 1, 0))
        return m

    def _proj_matrix(self) -> QMatrix4x4:
        m = QMatrix4x4()
        aspect = self.width() / max(1, self.height())
        near = max(self._dist * 0.001, self._radius * 1e-3, 1e-4)
        # Far plane sized from the TRUE (not robust) extent so outlier points
        # beyond the framed model body are still drawn, never z-clipped.
        far = self._dist + self._true_extent * 2.5 + 10.0
        m.perspective(self._fov, aspect, near, far)
        return m

    # ------------------------------------------------------------------ #
    # Painting
    # ------------------------------------------------------------------ #
    def paintGL(self):
        gl = self.gl
        # Guard: paintGL can fire before initializeGL finishes, or after a
        # shader build failure. Draw a plain clear rather than crashing.
        if gl is None:
            return
        if self._bg_prog is None or self._prog is None or self._line_prog is None:
            gl.glClearColor(0.05, 0.06, 0.09, 1.0)
            gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            return
        if self._pending_upload:
            self._upload_cloud()

        paint_t0 = time.perf_counter()
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # ---- background gradient (no depth) ----
        gl.glDisable(GL_DEPTH_TEST)
        self._bg_prog.bind()
        top = self._theme.gl_bg_top(); bot = self._theme.gl_bg_bottom()
        self._set_vec3(self._bg_prog, "u_top", top)
        self._set_vec3(self._bg_prog, "u_bottom", bot)
        self._empty_vao.bind()
        gl.glDrawArrays(GL_TRIANGLES, 0, 3)
        self._empty_vao.release()
        self._bg_prog.release()

        proj = self._proj_matrix()
        view = self._view_matrix()
        mvp = proj * view

        gl.glEnable(GL_DEPTH_TEST)

        # ---- grid + axes (blended lines) ----
        gl.glEnable(GL_BLEND)
        gl.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        if self.show_grid and self._grid_count:
            self._line_prog.bind()
            self._line_prog.setUniformValue("u_mvp", mvp)
            self._line_prog.setUniformValue1f("u_alpha", 0.5)
            self._grid_vao.bind()
            gl.glDrawArrays(GL_LINES, 0, self._grid_count)
            self._grid_vao.release()
            self._line_prog.release()
        if self.show_axes:
            self._line_prog.bind()
            self._line_prog.setUniformValue("u_mvp", mvp)
            self._line_prog.setUniformValue1f("u_alpha", 1.0)
            self._axis_vao.bind()
            gl.glDrawArrays(GL_LINES, 0, 6)
            self._axis_vao.release()
            self._line_prog.release()

        # ---- point cloud ----
        n_gpu = getattr(self, "_n_gpu", 0)
        if n_gpu > 0:
            # Adaptive level of detail. Points are shuffled on upload, so a
            # prefix is always a uniform random subsample. The controller sizes
            # that prefix to hit a target frame TIME: a tight one (~33 ms) while
            # dragging so orbiting stays smooth, and a looser one (~300 ms) once
            # idle so the crispest affordable view appears — neither a fixed
            # point count. It self-tunes to whatever fill-rate this renderer has.
            draw_n = self._adaptive_draw_n(n_gpu)

            # Fast mode draws flat OPAQUE squares, so blending (enabled for the
            # grid/axis lines above) must be off — it both wastes fill-rate and
            # would let the background bleed through. Quality mode keeps blend
            # for soft anti-aliased round points.
            if self.fast_mode:
                gl.glDisable(GL_BLEND)

            # On desktop GL, gl_PointSize needs PROGRAM_POINT_SIZE enabled; on
            # ES it is always honoured and the enum is invalid.
            # GL_POINT_SPRITE is needed for gl_PointCoord to work on legacy
            # (GLSL 1.30 / compatibility) contexts; without it gl_PointCoord
            # is always (0,0) and the fragment shader discards everything.
            # Core 3.2+ contexts ignore the enum (point sprites are always on)
            # so the enable is harmless there.
            if not getattr(self, "_is_es", False):
                gl.glEnable(GL_PROGRAM_POINT_SIZE)
                gl.glEnable(GL_POINT_SPRITE)
            self._prog.bind()
            self._prog.setUniformValue("u_mvp", mvp)
            self._prog.setUniformValue("u_mv", view)
            self._prog.setUniformValue1f("u_point_size", float(self.point_size))
            self._prog.setUniformValue1f("u_attenuation", float(self.attenuation))
            self._prog.setUniformValue1f("u_viewport_h", float(self.height()))
            self._prog.setUniformValue1f("u_max_point", float(self._max_point_px))
            self._prog.setUniformValue1f("u_fov", float(self._fov))
            self._prog.setUniformValue1f("u_scene_radius", float(self._radius))
            self._prog.setUniformValue1i("u_shaded", 1 if self.shaded else 0)
            self._prog.setUniformValue1i("u_fast", 1 if self.fast_mode else 0)
            self._prog.setUniformValue1i("u_quality", int(self.quality_level))
            self._prog.setUniformValue1i("u_has_scale", 1 if self._has_per_point_scale else 0)
            b = float(self.brightness)
            self._set_vec3(self._prog, "u_tint", tuple(c * b for c in self._theme.gl_point_tint()))
            self._set_vec3(self._prog, "u_light_dir", (0.4, 0.6, 0.7))
            # Hemisphere ambient colors from theme background
            top = self._theme.gl_bg_top(); bot = self._theme.gl_bg_bottom()
            self._set_vec3(self._prog, "u_bg_sky", top)
            self._set_vec3(self._prog, "u_bg_ground", bot)

            # Disable depth writes for semi-transparent Gaussian data so
            # overlapping splats composite correctly regardless of draw order.
            disable_depth_write = (self._has_per_point_scale and _gl32 is not None
                                   and not self.fast_mode)
            if disable_depth_write:
                _gl32.glDepthMask(0)

            self._vao.bind()
            gl.glDrawArrays(GL_POINTS, 0, draw_n)
            self._vao.release()

            if disable_depth_write:
                _gl32.glDepthMask(1)

            self._prog.release()

            # Record this frame's cost and, when settled, keep pumping repaints
            # until the adaptive count converges on the idle target — i.e. until
            # a single frame grows expensive enough. This makes the crisp view
            # appear progressively instead of one long freeze.
            self._last_paint_ms = (time.perf_counter() - paint_t0) * 1000.0
            # Update per-point cost EMA (cost is ~linear in count, so this is
            # stable and lets us predict the budget for any frame-time target).
            if draw_n > 0 and self._last_paint_ms > 0:
                cpp = self._last_paint_ms / draw_n
                if self._cpp_ema <= 0:
                    self._cpp_ema = cpp
                else:
                    self._cpp_ema = 0.7 * self._cpp_ema + 0.3 * cpp
            self._maybe_schedule_refine(n_gpu, draw_n)

        gl.glDisable(GL_BLEND)

    def _set_vec3(self, prog, name, v):
        prog.setUniformValue(name, QVector3D(float(v[0]), float(v[1]), float(v[2])))

    # ------------------------------------------------------------------ #
    # Adaptive LOD
    # ------------------------------------------------------------------ #
    def _adaptive_draw_n(self, n_gpu: int) -> int:
        """Pick how many points to draw this frame to hit a target frame time.

        Rendering cost is ~linear in point count, so we keep an EMA of the cost
        per point and predict the count directly: ``n = target_ms / cpp``.
        While dragging the target is ~33 ms (smooth orbit); once idle it's a
        looser ~300 ms so the densest affordable view settles in one tolerable
        frame. Because points are shuffled on upload, any prefix is a uniform
        subsample, so fewer points still reads as the whole scene.
        """
        if n_gpu <= 0:
            self._adaptive_n = 0
            return 0

        interacting = self._interacting
        target_ms = self._target_fps_ms if interacting else self._static_target_ms
        ceiling = (self._interactive_budget if interacting
                   else (self._static_budget or n_gpu))
        ceiling = max(15000, min(ceiling, n_gpu))

        if self._cpp_ema > 1e-6:
            n = int(target_ms / self._cpp_ema)
        else:
            # pre-calibration: a sensible probe count (~1/8 of the cloud)
            n = min(ceiling, max(30000, n_gpu // 8))

        n = int(max(15000, min(n, ceiling)))
        self._adaptive_n = n
        return n

    def _static_converged(self, n_gpu: int, draw_n: int) -> bool:
        """Has the idle view settled? Stops the refine pump when true.

        Converged once the predicted budget (from the per-point cost) matches
        what we just drew — i.e. another frame wouldn't change the count. Also
        stops at the floor, or when every point is drawn and that's affordable.
        """
        if draw_n <= 15000:
            return True                              # pinned at floor; give up
        if draw_n >= n_gpu:
            # drew everything; done only if it fits the budget
            return self._cpp_ema <= 0 or (n_gpu * self._cpp_ema) < self._static_target_ms * 1.1
        if self._cpp_ema <= 0:
            return False
        predicted = int(self._static_target_ms / self._cpp_ema)
        return abs(predicted - draw_n) < max(15000, draw_n * 0.08)

    def _maybe_schedule_refine(self, n_gpu: int, draw_n: int):
        """After an idle frame, keep refining toward the static target until the
        frame time converges (in either direction). Guards against runaway."""
        if self._interacting:
            self._refine_pending = False
            return
        if self._static_converged(n_gpu, draw_n):
            self._refine_pending = False
            return
        if self._refine_pending:
            return
        self._refine_pending = True
        QTimer.singleShot(0, self._refine_step)

    def _refine_step(self):
        self._refine_pending = False
        if not self._interacting and self._gl_ready:
            self.update()

    # ------------------------------------------------------------------ #
    # Interaction
    # ------------------------------------------------------------------ #
    def _begin_interaction(self):
        self._interacting = True
        self._refine_pending = False
        self._idle_timer.start()

    def _on_idle(self):
        self._interacting = False
        self.update()

    def mousePressEvent(self, e):
        self._last_mouse = e.position().toPoint()
        self._begin_interaction()

    def mouseMoveEvent(self, e):
        pos = e.position().toPoint()
        dx = pos.x() - self._last_mouse.x()
        dy = pos.y() - self._last_mouse.y()
        self._last_mouse = pos
        if e.buttons() & Qt.LeftButton:
            self._yaw -= dx * 0.008
            self._pitch += dy * 0.008
            lim = np.radians(89.0)
            self._pitch = float(np.clip(self._pitch, -lim, lim))
            self._begin_interaction()
            self.update()
        elif e.buttons() & (Qt.RightButton | Qt.MiddleButton):
            # pan in the view plane, scaled so 1px ~ constant screen motion
            scale = self._dist * np.tan(np.radians(self._fov * 0.5)) / max(1, self.height()) * 2.0
            right, up = self._basis()
            self._target -= right * (dx * scale)
            self._target += up * (dy * scale)
            self._begin_interaction()
            self.update()

    def mouseReleaseEvent(self, e):
        self._idle_timer.start()

    def wheelEvent(self, e):
        delta = e.angleDelta().y() / 120.0
        factor = 0.85 ** delta
        self._dist = float(np.clip(self._dist * factor, self._radius * 1e-3, self._radius * 50 + 100))
        self._begin_interaction()
        self.update()

    def mouseDoubleClickEvent(self, e):
        self.reset_view()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_F:
            self.reset_view()
        else:
            super().keyPressEvent(e)

    def _basis(self):
        """Camera right/up vectors in world space."""
        eye = self._eye()
        fwd = (self._target - eye).normalized()
        world_up = QVector3D(0, 1, 0)
        right = QVector3D.crossProduct(fwd, world_up).normalized()
        up = QVector3D.crossProduct(right, fwd).normalized()
        return right, up


def _nice_step(x: float) -> float:
    """Round a length to a 1/2/5 * 10^k 'nice' value for grid spacing."""
    if x <= 0:
        return 1.0
    import math
    exp = math.floor(math.log10(x))
    base = 10 ** exp
    f = x / base
    if f < 1.5:
        nice = 1
    elif f < 3.5:
        nice = 2
    elif f < 7.5:
        nice = 5
    else:
        nice = 10
    return nice * base


def default_surface_format() -> QSurfaceFormat:
    """A 3.3 core format with MSAA — call before the app creates windows."""
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setVersion(3, 3)
    fmt.setDepthBufferSize(32)       # 32-bit depth for tighter z-precision
    fmt.setStencilBufferSize(8)
    fmt.setSamples(8)                # 8x MSAA for smoother point edges
    fmt.setSwapInterval(1)
    return fmt
