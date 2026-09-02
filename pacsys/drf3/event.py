import math
import re

# DRF2 time-freq: dec-number [ S | M | U | H | K ]
_TIME_FREQ_RE = re.compile(r"^(\d+)([SMUHK])?$", re.IGNORECASE)


def _java_round(x: float) -> int:
    """Match Java Math.round() -- floor(x + 0.5), rounds .5 up."""
    return math.floor(x + 0.5)


def _parse_time_freq(raw: str) -> int:
    """Parse a DRF2 time-freq value and return milliseconds (matches Java TimeFreq)."""
    m = _TIME_FREQ_RE.match(raw)
    if m is None:
        raise ValueError(f"Bad time-freq value: {raw}")
    num = int(m.group(1))
    if num > 0x7FFFFFFF:  # Java Integer.parseInt range; also keeps the divisions below from overflowing
        raise ValueError(f"Bad time-freq value: {raw}")
    unit = (m.group(2) or "M").upper()
    if num == 0:
        return 0
    if unit == "S":
        return num * 1000
    if unit == "M":
        return num
    if unit == "U":
        return _java_round(num / 1000)
    if unit == "H":
        return _java_round(1000 / num)
    if unit == "K":
        return _java_round(1000 / (num * 1000))
    raise ValueError(f"Bad time-freq unit: {unit}")


def parse_event(parse_str: str | None):
    if parse_str is None:
        return None
    if not parse_str:
        raise ValueError("event string must not be empty")
    normalized = parse_str.upper()
    char = normalized[0]
    if normalized == "U":
        return DefaultEvent()
    if normalized == "I":
        return ImmediateEvent()
    if char in ["P", "Q"]:
        return PeriodicEvent(parse_str, char)
    if char == "E":
        return ClockEvent(parse_str, char)
    if char == "S":
        return StateEvent(parse_str, char)
    if char == "N":
        return NeverEvent(parse_str, char)
    raise ValueError(f"Invalid event: {parse_str}")


class DRF_EVENT:  # noqa: N801 -- established DRF API
    def __init__(self, raw_string: str, mode):
        self.raw_string = raw_string
        self.mode = mode

    def __eq__(self, other):
        if not isinstance(other, DRF_EVENT):
            return NotImplemented
        return self.raw_string == other.raw_string

    def __repr__(self):
        return f"<DRF_EVENT mode {self.mode}: ({self.raw_string})>"


class DefaultEvent(DRF_EVENT):
    def __init__(self, raw_string="U", mode="U"):
        if raw_string != "U" or mode != "U":
            raise ValueError(f"DefaultEvent requires raw_string='U' and mode='U', got {raw_string!r}, {mode!r}")
        super().__init__(raw_string, mode)


class ImmediateEvent(DRF_EVENT):
    def __init__(self, raw_string="I", mode="I"):
        if raw_string != "I" or mode != "I":
            raise ValueError(f"ImmediateEvent requires raw_string='I' and mode='I', got {raw_string!r}, {mode!r}")
        super().__init__(raw_string, mode)


class PeriodicEvent(DRF_EVENT):
    def __init__(self, raw_string, mode):
        super().__init__(raw_string, mode)
        match = re.match("(?i)(P|Q)(?:,(\\w+)(?:,(F|FALSE|T|TRUE))?)?" + "$", raw_string)
        if match is None:
            raise ValueError(f"Bad periodic event {raw_string}")
        imm = True
        freq = 1000
        if match.group(2) is not None:
            freq = _parse_time_freq(match.group(2))
            if match.group(3) is not None:
                imm = match.group(3)[0].upper() == "T"
        self.cont = match.group(1)[0].upper() == "P"
        self.imm = imm
        self.freq = freq


class ClockEvent(DRF_EVENT):
    def __init__(self, raw_string, mode):
        super().__init__(raw_string, mode)
        match = re.match("(?i)E,([0-9A-F]+)(?:,([HSE])(?:,(\\w+))?)?" + "$", raw_string)
        if match is None:
            raise ValueError(f"Bad clock event {raw_string}")
        evt = int(match.group(1), 16)
        delay = 0
        clock_type = "either"
        if match.group(2) is not None:
            clock_type = match.group(2)
            if match.group(3) is not None:
                delay = _parse_time_freq(match.group(3))
        self.evt = evt
        self.delay = delay
        self.clock_type = clock_type


class NeverEvent(DRF_EVENT):
    """No data should ever be sent back. Used for write/setting operations.

    Java reference:
    - isRepetitive() = false, defaultTimeout() = 0
    - FTD code = 2 (the "never" fetch-time-descriptor)
    - Only event type that REMOVES the EVENT flag from ACNET options, meaning "don't trigger data transmission"
    """

    def __init__(self, raw_string="N", mode="N"):
        if raw_string.upper() != "N":
            raise ValueError(f"Invalid never event: {raw_string}")
        super().__init__("N", "N")


class StateEvent(DRF_EVENT):
    def __init__(self, raw_string, mode):
        super().__init__(raw_string, mode)
        match = re.match("(?i)S,(\\S+),(\\d+),(\\w+),(=|!=|\\*|>|<|<=|>=)" + "$", raw_string)
        if match is None:
            raise ValueError(f"Bad state event {raw_string}")
