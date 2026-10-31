"""Shared type aliases for Orbit AI."""

from typing import Any, Dict, List, Union

JSONPrimitive = Union[str, int, float, bool, None]
JSONObject = Dict[str, Any]
Vector128 = List[float]
