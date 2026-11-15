"""Matplotlib style sheets bundled with the package."""
from importlib.resources import files

import matplotlib.pyplot as plt


def style_path(name: str) -> str:
  """Absolute path of a bundled style sheet, e.g. ``style_path("poster")``."""
  return str(files(__name__) / f"{name}.mplstyle")


def use_style(name: str) -> None:
  """Apply a bundled style sheet (``"paper"`` or ``"poster"``)."""
  plt.style.use(style_path(name))
