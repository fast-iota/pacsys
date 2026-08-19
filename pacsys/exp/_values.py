"""Shared value validation for experiment utilities."""

from typing import SupportsFloat, SupportsIndex

import numpy as np


def numeric_value(value: object) -> float:
    """Return a numeric value as float, rejecting booleans and containers."""
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"Cannot convert {type(value).__name__} to a numeric value")
    if isinstance(value, (str, SupportsFloat, SupportsIndex)):
        return float(value)
    raise TypeError(f"Cannot convert {type(value).__name__} to a numeric value")
