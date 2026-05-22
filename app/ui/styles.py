"""Shared Qt stylesheets."""

APP_BACKGROUND = "#ecf0f1"
PRIMARY = "#3498db"
PRIMARY_HOVER = "#2980b9"
DANGER = "#e74c3c"
DANGER_HOVER = "#c0392b"
TEXT_DARK = "#2c3e50"
TEXT_MUTED = "#7f8c8d"
PANEL_BORDER = "#bdc3c7"
VIDEO_BG = "#34495e"

MAIN_WINDOW_STYLE = f"background-color: {APP_BACKGROUND};"

BTN_PRIMARY = f"""
    QPushButton {{
        background-color: {PRIMARY};
        color: white;
        border: none;
        padding: 15px 40px;
        border-radius: 8px;
        min-width: 220px;
        font-size: 15px;
    }}
    QPushButton:hover {{
        background-color: {PRIMARY_HOVER};
    }}
"""

BTN_DANGER = f"""
    QPushButton {{
        background-color: {DANGER};
        color: white;
        border: none;
        padding: 12px;
        border-radius: 8px;
        min-height: 42px;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background-color: {DANGER_HOVER};
    }}
"""

BTN_SECONDARY = f"""
    QPushButton {{
        background-color: transparent;
        color: {TEXT_DARK};
        border: 1px solid {PANEL_BORDER};
        padding: 12px;
        border-radius: 8px;
        min-height: 42px;
        font-size: 13px;
    }}
    QPushButton:hover {{
        background-color: #dfe6e9;
    }}
"""
