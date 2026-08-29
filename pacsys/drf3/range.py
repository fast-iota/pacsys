import re
from typing import Literal

RANGE_RE = re.compile("(\\[(\\d*)(?::(\\d*))?\\])|(\\{(\\d*)(?::(\\d*))?\\})")

# Sentinel for an unbounded byte-range length
MAXIMUM = -2147483648
MAX_UPPER_BOUND = 2147483648
MAX_INDEX = 2147483647  # Java Integer.MAX_VALUE (ArrayRange bound)


def parse_range(raw_string: str | None):
    if raw_string is None:
        return None
    match = RANGE_RE.fullmatch(raw_string)
    if match is None:
        raise ValueError(f"Bad range {raw_string}")
    if match.group(1) is not None:
        s1, s2 = match.group(2), match.group(3)
        if s1 is None and s2 is None:
            # []
            return ARRAY_RANGE(mode="full")
        idx1 = int(s1) if s1 != "" and s1 is not None else None
        idx2 = int(s2) if s2 != "" and s2 is not None else None
        if idx1 is None and idx2 is None:
            # [:]
            return ARRAY_RANGE(mode="full")
        if idx2 is None and ":" not in raw_string:
            return ARRAY_RANGE(mode="single", low=idx1, high=idx2)
        return ARRAY_RANGE(mode="std", low=idx1, high=idx2)
    if match.group(4) is not None:
        s1 = match.group(5)
        s2 = match.group(6)
        s1empty = s1 is None
        s2empty = s2 is None
        if s1empty and s2empty:
            return BYTE_RANGE(mode="full")
        idx1 = int(s1) if s1 != "" and s1 is not None else None
        idx2 = int(s2) if s2 != "" and s2 is not None else None
        if idx1 is None and idx2 is None:
            return BYTE_RANGE(mode="full")
        if idx2 is None and ":" not in raw_string:
            return BYTE_RANGE(mode="single", offset=idx1, length=idx2)
        return BYTE_RANGE(mode="std", offset=idx1, length=idx2)
    raise Exception("Unrecognized range specifier")


class ARRAY_RANGE:  # noqa: N801 -- established DRF API
    def __init__(
        self,
        mode: Literal["full", "std", "single"] | None = None,
        low: int | None = None,
        high: int | None = None,
    ):
        if low is not None and not 0 <= low <= MAX_INDEX:
            raise ValueError(f"array range start must be 0..{MAX_INDEX}, got {low}")
        if high is not None and not 0 <= high <= MAX_INDEX:
            raise ValueError(f"array range end must be 0..{MAX_INDEX}, got {high}")
        if low is not None and high is not None and high < low:
            raise ValueError(f"array range end must not precede start, got [{low}:{high}]")
        self.low = low
        self.high = high
        self.mode: Literal["full", "std", "single"] = mode or ("full" if (low is None and high is None) else "std")

    def __eq__(self, other):
        if not isinstance(other, ARRAY_RANGE):
            return NotImplemented
        return self.low == other.low and self.high == other.high and self.mode == other.mode

    def __str__(self):
        if self.mode == "full":
            return "[:]"
        if self.mode == "single":
            return f"[{self.low}]"
        s = "["
        if self.low is not None:
            s += f"{self.low}"
        s += ":"
        if self.high is not None:
            s += f"{self.high}"
        s += "]"
        return s

    def __repr__(self):
        return f"<ARRAY_RANGE: {self!s} ({self.mode} mode)>"


class BYTE_RANGE:  # noqa: N801 -- established DRF API
    def __init__(
        self,
        mode: Literal["full", "std", "single"] | None = None,
        offset: int | None = None,
        length: int | None = None,
    ):
        if offset is not None and offset < 0:
            raise ValueError("offset must be non-negative")
        if length is not None and (length != MAXIMUM and length < 0):
            raise ValueError("length must be non-negative")
        if offset is not None and length is not None:
            if length != MAXIMUM and offset + length > MAX_UPPER_BOUND:
                raise ValueError("offset + length must be less than Integer.MAX_VALUE")
            if offset == 0 and length == MAXIMUM:
                if mode != "full":
                    raise ValueError("mode must be 'full' when offset=0 and length=MAXIMUM")
        self.offset = offset
        self.length = length
        self.mode = mode

    def __eq__(self, other):
        if not isinstance(other, BYTE_RANGE):
            return NotImplemented
        return self.offset == other.offset and self.length == other.length and self.mode == other.mode

    def __str__(self):
        if self.mode == "full":
            return "{:}"
        if self.mode == "single":
            return f"{{{self.offset}}}"
        s = "{"
        if self.offset is not None:
            s += f"{self.offset}"
        s += ":"
        if self.length is not None:
            s += f"{self.length}"
        s += "}"
        return s

    def __repr__(self):
        return f"<BYTE_RANGE: {self!s} ({self.mode} mode)>"
