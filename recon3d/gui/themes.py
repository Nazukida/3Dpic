"""Theme system for recon3d.

A :class:`Theme` bundles every colour the app needs: the Qt "chrome" (windows,
buttons, panels) *and* the 3D viewer palette (background gradient, point tint,
axis/grid colours).  Themes are plain dataclasses so they serialise to JSON,
which lets the user save their own custom themes next to the built-in presets.

The module exposes three things the rest of the GUI relies on:

* :data:`PRESETS`            - ordered dict of built-in themes.
* :func:`build_qss`          - turn a theme into a Qt style sheet string.
* :func:`load_user_themes` / :func:`save_user_theme` - persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import json
import os
from typing import Iterable


# --------------------------------------------------------------------------- #
# Small colour helpers (kept dependency-free: plain hex string manipulation).
# --------------------------------------------------------------------------- #
def _clamp(v: float) -> int:
    return max(0, min(255, int(round(v))))


def hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{_clamp(r):02x}{_clamp(g):02x}{_clamp(b):02x}"


def mix(a: str, b: str, t: float) -> str:
    """Linear blend of two hex colours; ``t=0`` -> a, ``t=1`` -> b."""
    ar, ag, ab = hex_to_rgb(a)
    br, bg, bb = hex_to_rgb(b)
    return rgb_to_hex(ar + (br - ar) * t, ag + (bg - ag) * t, ab + (bb - ab) * t)


def lighten(h: str, t: float) -> str:
    return mix(h, "#ffffff", t)


def darken(h: str, t: float) -> str:
    return mix(h, "#000000", t)


def rgb_floats(h: str) -> tuple[float, float, float]:
    """Hex -> 0..1 floats, for feeding OpenGL."""
    r, g, b = hex_to_rgb(h)
    return r / 255.0, g / 255.0, b / 255.0


# --------------------------------------------------------------------------- #
# Theme definition.
# --------------------------------------------------------------------------- #
@dataclass
class Theme:
    name: str

    # --- Qt chrome ---
    bg: str            # window background
    surface: str       # panels / docks / cards
    surface_alt: str   # inputs, list rows
    border: str        # hairline separators
    text: str          # primary text
    text_muted: str    # secondary text
    accent: str        # primary interactive colour
    accent_text: str   # text drawn on top of accent
    danger: str = "#e5484d"

    # --- 3D viewer palette ---
    view_bg_top: str = "#1b1e28"      # gradient top of the GL canvas
    view_bg_bottom: str = "#0d0f16"   # gradient bottom
    view_axis_x: str = "#e5484d"
    view_axis_y: str = "#46a758"
    view_axis_z: str = "#3e8ef7"
    view_grid: str = "#2a2f3a"
    view_point_tint: str = "#ffffff"  # multiplied with per-point colour
    view_text: str = "#c9d1e0"        # HUD / overlay text

    # whether this is a light theme (affects a few UI heuristics)
    is_light: bool = False

    # marks user-saved themes so the UI can offer "delete"
    builtin: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Theme":
        # tolerate older/partial dicts by falling back to field defaults
        allowed = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in allowed})

    # convenience for the GL viewer
    def gl_bg_top(self) -> tuple[float, float, float]:
        return rgb_floats(self.view_bg_top)

    def gl_bg_bottom(self) -> tuple[float, float, float]:
        return rgb_floats(self.view_bg_bottom)

    def gl_point_tint(self) -> tuple[float, float, float]:
        return rgb_floats(self.view_point_tint)


# --------------------------------------------------------------------------- #
# Built-in presets.  Each is tuned as a coherent set, not just random colours.
# --------------------------------------------------------------------------- #
def _midnight() -> Theme:
    return Theme(
        name="午夜蓝",
        bg="#0d0f16", surface="#161a24", surface_alt="#1f2430",
        border="#2a2f3d", text="#e6e9f0", text_muted="#8b93a7",
        accent="#5b8def", accent_text="#ffffff",
        view_bg_top="#1a1f2e", view_bg_bottom="#090b12",
        view_grid="#232838", view_point_tint="#ffffff", view_text="#c9d1e0",
        is_light=False,
    )


def _graphite() -> Theme:
    return Theme(
        name="石墨灰",
        bg="#17181a", surface="#202124", surface_alt="#2a2c30",
        border="#34363b", text="#e8eaed", text_muted="#9aa0a6",
        accent="#8ab4f8", accent_text="#0b0d10",
        view_bg_top="#26282c", view_bg_bottom="#141517",
        view_grid="#33363c", view_point_tint="#ffffff", view_text="#c7ccd4",
        is_light=False,
    )


def _nord() -> Theme:
    return Theme(
        name="北欧",
        bg="#2e3440", surface="#3b4252", surface_alt="#434c5e",
        border="#4c566a", text="#eceff4", text_muted="#a7b0c0",
        accent="#88c0d0", accent_text="#2e3440",
        view_bg_top="#3b4252", view_bg_bottom="#242933",
        view_axis_x="#bf616a", view_axis_y="#a3be8c", view_axis_z="#81a1c1",
        view_grid="#434c5e", view_point_tint="#eceff4", view_text="#d8dee9",
        is_light=False,
    )


def _solarized_dark() -> Theme:
    return Theme(
        name="曝晒暗",
        bg="#002b36", surface="#073642", surface_alt="#0a4351",
        border="#0f5162", text="#eee8d5", text_muted="#93a1a1",
        accent="#268bd2", accent_text="#ffffff",
        view_bg_top="#073642", view_bg_bottom="#00212b",
        view_axis_x="#dc322f", view_axis_y="#859900", view_axis_z="#268bd2",
        view_grid="#0d4a5a", view_point_tint="#fdf6e3", view_text="#93a1a1",
        is_light=False,
    )


def _amber() -> Theme:
    # warm, high-contrast "lab" look
    return Theme(
        name="琥珀实验室",
        bg="#14110c", surface="#1f1a12", surface_alt="#2a2318",
        border="#3a3122", text="#f3e9d6", text_muted="#b09a72",
        accent="#f5a623", accent_text="#1a1408",
        view_bg_top="#241d12", view_bg_bottom="#0d0b07",
        view_axis_x="#ff6b4a", view_axis_y="#c8d94a", view_axis_z="#4ac8ff",
        view_grid="#332a1c", view_point_tint="#fff3dc", view_text="#e6d5b0",
        is_light=False,
    )


def _snow() -> Theme:
    return Theme(
        name="雪白",
        bg="#f4f6fb", surface="#ffffff", surface_alt="#eef1f7",
        border="#d6dbe6", text="#1c2330", text_muted="#5d6575",
        accent="#3b6fe0", accent_text="#ffffff",
        view_bg_top="#eef2fa", view_bg_bottom="#d4dcec",
        view_axis_x="#d1453b", view_axis_y="#2f9e44", view_axis_z="#1c6fd6",
        view_grid="#c2cbdc", view_point_tint="#20232b", view_text="#3a4152",
        is_light=True,
    )


def _paper() -> Theme:
    return Theme(
        name="纸张",
        bg="#ece7db", surface="#f6f2e9", surface_alt="#e2dccb",
        border="#cfc7b2", text="#2c2a24", text_muted="#6f6a5c",
        accent="#b06a2c", accent_text="#fff7ec",
        view_bg_top="#efe9db", view_bg_bottom="#d8d0bd",
        view_axis_x="#b5432f", view_axis_y="#5d7d2f", view_axis_z="#2f6d8c",
        view_grid="#ccc3ad", view_point_tint="#2c2a24", view_text="#4a463c",
        is_light=True,
    )


def _neon() -> Theme:
    # vivid synthwave, fun preset for point clouds
    return Theme(
        name="霓虹",
        bg="#0b0720", surface="#150f30", surface_alt="#1e163f",
        border="#2c2159", text="#eae4ff", text_muted="#9b8fd6",
        accent="#ff3caa", accent_text="#ffffff",
        view_bg_top="#1a0f3a", view_bg_bottom="#05030f",
        view_axis_x="#ff3caa", view_axis_y="#3cff9e", view_axis_z="#3c9eff",
        view_grid="#241a52", view_point_tint="#ffffff", view_text="#c9b8ff",
        is_light=False,
    )


# order matters: this is the order shown in the theme picker
_PRESET_FACTORIES = [
    _midnight, _graphite, _nord, _solarized_dark,
    _amber, _neon, _snow, _paper,
]

PRESETS: "dict[str, Theme]" = {t.name: t for t in (f() for f in _PRESET_FACTORIES)}

DEFAULT_THEME = "午夜蓝"


# --------------------------------------------------------------------------- #
# QSS generation.
# --------------------------------------------------------------------------- #
def build_qss(t: Theme) -> str:
    """Render a theme as a Qt style sheet.

    Kept deliberately flat and explicit so it's easy to read and tweak per
    widget.  Radii/spacing are constant across themes for a consistent feel.
    """
    hover = lighten(t.accent, 0.12) if not t.is_light else darken(t.accent, 0.08)
    pressed = darken(t.accent, 0.12)
    sel_bg = t.accent
    sel_fg = t.accent_text
    scroll = mix(t.surface, t.text, 0.18)
    scroll_hi = mix(t.surface, t.text, 0.30)
    input_focus = t.accent
    disabled = t.text_muted

    return f"""
* {{
    font-family: "Segoe UI", "Inter", "Microsoft YaHei UI", "Microsoft YaHei",
                 "PingFang SC", "Noto Sans CJK SC", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    outline: none;
}}
QWidget {{
    background-color: {t.bg};
    color: {t.text};
}}
QMainWindow, QDialog {{ background-color: {t.bg}; }}

QToolTip {{
    background-color: {t.surface_alt};
    color: {t.text};
    border: 1px solid {t.border};
    padding: 4px 6px;
    border-radius: 4px;
}}

/* ---- panels / docks ---- */
QDockWidget {{
    color: {t.text_muted};
    titlebar-close-icon: none;
}}
QDockWidget::title {{
    background-color: {t.surface};
    padding: 7px 10px;
    border: 1px solid {t.border};
    border-bottom: none;
    font-weight: 600;
}}
QGroupBox {{
    background-color: {t.surface};
    border: 1px solid {t.border};
    border-radius: 8px;
    margin-top: 16px;
    padding: 10px 10px 10px 10px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    top: 2px;
    padding: 0 4px;
    color: {t.text_muted};
}}
QFrame#Card, QWidget#Card {{
    background-color: {t.surface};
    border: 1px solid {t.border};
    border-radius: 8px;
}}

/* ---- buttons ---- */
QPushButton {{
    background-color: {t.surface_alt};
    color: {t.text};
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 6px 14px;
}}
QPushButton:hover {{ background-color: {lighten(t.surface_alt, 0.06) if not t.is_light else darken(t.surface_alt, 0.04)}; }}
QPushButton:pressed {{ background-color: {darken(t.surface_alt, 0.06)}; }}
QPushButton:disabled {{ color: {disabled}; background-color: {t.surface}; }}
QPushButton#Primary {{
    background-color: {t.accent};
    color: {sel_fg};
    border: 1px solid {t.accent};
    font-weight: 600;
}}
QPushButton#Primary:hover {{ background-color: {hover}; border-color: {hover}; }}
QPushButton#Primary:pressed {{ background-color: {pressed}; border-color: {pressed}; }}
QPushButton#Danger {{
    background-color: {t.danger};
    color: #ffffff;
    border: 1px solid {t.danger};
    font-weight: 600;
}}

/* ---- inputs ---- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
    background-color: {t.surface_alt};
    color: {t.text};
    border: 1px solid {t.border};
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: {sel_bg};
    selection-color: {sel_fg};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QPlainTextEdit:focus, QTextEdit:focus {{
    border: 1px solid {input_focus};
}}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{
    background-color: {t.surface_alt};
    color: {t.text};
    border: 1px solid {t.border};
    selection-background-color: {sel_bg};
    selection-color: {sel_fg};
    outline: none;
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{ width: 16px; border: none; }}

/* ---- checkbox / radio ---- */
QCheckBox, QRadioButton {{ spacing: 7px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {t.border};
    border-radius: 4px;
    background-color: {t.surface_alt};
}}
QRadioButton::indicator {{ border-radius: 8px; }}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {t.accent};
    border-color: {t.accent};
}}

/* ---- sliders ---- */
QSlider::groove:horizontal {{
    height: 4px; background: {t.border}; border-radius: 2px;
}}
QSlider::handle:horizontal {{
    background: {t.accent};
    width: 14px; height: 14px; margin: -6px 0; border-radius: 7px;
}}
QSlider::sub-page:horizontal {{ background: {mix(t.accent, t.surface, 0.3)}; border-radius: 2px; }}

/* ---- menus / toolbar ---- */
QMenuBar {{ background-color: {t.surface}; color: {t.text}; }}
QMenuBar::item {{ background: transparent; padding: 6px 10px; }}
QMenuBar::item:selected {{ background: {t.surface_alt}; }}
QMenu {{ background-color: {t.surface}; color: {t.text}; border: 1px solid {t.border}; padding: 4px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: {sel_bg}; color: {sel_fg}; }}
QMenu::separator {{ height: 1px; background: {t.border}; margin: 4px 6px; }}
QToolBar {{
    background-color: {t.surface};
    border-bottom: 1px solid {t.border};
    spacing: 4px; padding: 4px;
}}
QToolButton {{ background: transparent; border: none; border-radius: 6px; padding: 6px 8px; color: {t.text}; }}
QToolButton:hover {{ background: {t.surface_alt}; }}
QToolButton:checked {{ background: {mix(t.accent, t.surface, 0.35)}; }}

/* ---- status bar ---- */
QStatusBar {{ background-color: {t.surface}; color: {t.text_muted}; border-top: 1px solid {t.border}; }}
QStatusBar::item {{ border: none; }}

/* ---- lists / tables ---- */
QListWidget, QTreeWidget, QTableWidget {{
    background-color: {t.surface};
    border: 1px solid {t.border};
    border-radius: 6px;
    alternate-background-color: {t.surface_alt};
}}
QListWidget::item, QTreeWidget::item {{ padding: 5px 6px; border-radius: 4px; }}
QListWidget::item:selected, QTreeWidget::item:selected {{ background: {sel_bg}; color: {sel_fg}; }}
QHeaderView::section {{
    background-color: {t.surface_alt}; color: {t.text_muted};
    padding: 5px; border: none; border-right: 1px solid {t.border};
}}

/* ---- progress ---- */
QProgressBar {{
    background-color: {t.surface_alt};
    border: 1px solid {t.border};
    border-radius: 6px;
    text-align: center;
    color: {t.text};
    height: 16px;
}}
QProgressBar::chunk {{
    background-color: {t.accent};
    border-radius: 5px;
}}

/* ---- scrollbars ---- */
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {scroll}; border-radius: 5px; min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: {scroll_hi}; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {scroll}; border-radius: 5px; min-width: 24px; }}
QScrollBar::handle:horizontal:hover {{ background: {scroll_hi}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---- tabs ---- */
QTabWidget::pane {{ border: 1px solid {t.border}; border-radius: 6px; top: -1px; }}
QTabBar::tab {{
    background: {t.surface}; color: {t.text_muted};
    padding: 7px 14px; border: 1px solid {t.border};
    border-top-left-radius: 6px; border-top-right-radius: 6px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{ background: {t.surface_alt}; color: {t.text}; }}

QLabel {{ background: transparent; }}
QLabel#Muted {{ color: {t.text_muted}; }}
QLabel#Title {{ font-size: 15px; font-weight: 700; }}
QSplitter::handle {{ background: {t.border}; }}
"""


# --------------------------------------------------------------------------- #
# Persistence of user themes.
# --------------------------------------------------------------------------- #
def _config_dir() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~/.config")
    d = os.path.join(base, "recon3d")
    os.makedirs(d, exist_ok=True)
    return d


def _user_themes_path() -> str:
    return os.path.join(_config_dir(), "themes.json")


def _settings_path() -> str:
    return os.path.join(_config_dir(), "settings.json")


def load_user_themes() -> "dict[str, Theme]":
    path = _user_themes_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    out: dict[str, Theme] = {}
    for d in data if isinstance(data, list) else []:
        try:
            t = Theme.from_dict(d)
            t.builtin = False
            out[t.name] = t
        except Exception:
            continue
    return out


def save_user_theme(theme: Theme) -> None:
    theme.builtin = False
    existing = load_user_themes()
    existing[theme.name] = theme
    _write_user_themes(existing.values())


def delete_user_theme(name: str) -> None:
    existing = load_user_themes()
    if name in existing:
        del existing[name]
        _write_user_themes(existing.values())


def _write_user_themes(themes: Iterable[Theme]) -> None:
    path = _user_themes_path()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in themes], f, indent=2)
    os.replace(tmp, path)


def all_themes() -> "dict[str, Theme]":
    """Presets first, then user themes (user may override a preset name)."""
    merged: dict[str, Theme] = dict(PRESETS)
    merged.update(load_user_themes())
    return merged


# ---- lightweight app settings (last theme, last dir, render prefs) -------- #
def load_settings() -> dict:
    path = _settings_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_settings(d: dict) -> None:
    path = _settings_path()
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass
