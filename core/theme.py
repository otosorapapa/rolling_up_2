"""Utilities for loading reusable CSS themes for the Streamlit app."""
from functools import lru_cache
from pathlib import Path
from textwrap import dedent

_STYLE_DIR = Path(__file__).parent / "styles"


def _wrap_css(css: str) -> str:
    """Return CSS content wrapped in a <style> block."""
    return f"<style>\n{dedent(css).strip()}\n</style>"


def _load_css(filename: str) -> str:
    path = _STYLE_DIR / filename
    return path.read_text(encoding="utf-8")


@lru_cache(maxsize=None)
def base_theme_css() -> str:
    """CSS for the global McKinsey-inspired light theme."""
    return _wrap_css(_load_css("base_theme.css"))


@lru_cache(maxsize=None)
def elegant_theme_css() -> str:
    """CSS for the optional elegant UI refinements."""
    return _wrap_css(_load_css("elegant_theme.css"))


__all__ = ["base_theme_css", "elegant_theme_css"]
