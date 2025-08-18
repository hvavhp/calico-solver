"""Enum package for Calico domain types.

Exports:
- Color: The six quilt colors used for buttons and tile attributes
- Pattern: The six quilt patterns used for cats and tile attributes
"""

from .color import Color
from .design_goal import DesignGoalConfig
from .pattern import Pattern

__all__ = [
    "Color",
    "Pattern",
    "DesignGoalConfig",
]
