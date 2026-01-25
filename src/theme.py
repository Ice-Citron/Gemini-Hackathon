#!/usr/bin/env python3
"""
SkyHammer Theme Configuration
Evangelion / Cyberpunk Neon Color Palette
"""

# =============================================================================
# THEME CONFIGURATION (EVANGELION / CYBERPUNK NEON)
# =============================================================================
# Color palette inspired by Neon Genesis Evangelion + Cyberpunk aesthetics

C_PRIMARY = "bold yellow"       # Main warnings / headers (NERV UI panels)
C_SECONDARY = "cyan"            # Information / borders / accents
C_ACCENT = "bold orange1"       # Critical alerts / attention items
C_SUCCESS = "bold green"        # Operations success / patched code
C_ERROR = "bold red"            # Failures / Attacks / vulnerabilities
C_DIM = "dim"                   # Background info / timestamps (single word for Rich closing tags)
C_HIGHLIGHT = "black on yellow" # High contrast highlight (Evangelion style)

# NOTE: Rich markup closing tags don't support multi-word styles like [/bold yellow]
# Use [/] to close any tag, or use single-word style names

# Questionary Style (for interactive prompts)
def get_eva_style():
    """Get Evangelion-style questionary theme"""
    try:
        import questionary
        return questionary.Style([
            ('qmark', 'fg:yellow bold'),
            ('question', 'fg:yellow bold'),
            ('answer', 'fg:cyan bold'),
            ('pointer', 'fg:cyan bold'),
            ('highlighted', 'fg:cyan bold'),
            ('selected', 'fg:cyan bold'),
            ('separator', 'fg:black'),
            ('instruction', 'fg:black'),
        ])
    except ImportError:
        return None
