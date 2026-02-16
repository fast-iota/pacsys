from enum import Enum, auto


class DRF_EXTRA(Enum):
    FTP = auto()
    LIVEDATA = auto()
    LOGGER = auto()
    LOGGERSINGLE = auto()
    LOGGERDURATION = auto()
    SRFILE = auto()
    SDAFILE = auto()
    REDIR = auto()
    MIRROR = auto()


DRF_EXTRA_NAMES = {el.name: el for el in DRF_EXTRA}

# Extras that provide their own data (historical/file) - no event should be injected
HISTORICAL_EXTRAS = frozenset({DRF_EXTRA.LOGGER, DRF_EXTRA.LOGGERSINGLE, DRF_EXTRA.LOGGERDURATION})


def parse_extra(raw_string: str) -> DRF_EXTRA:
    upper = raw_string.upper()
    # Strip colon-delimited parameters (e.g., "LOGGER:123:456" -> "LOGGER")
    name = upper.split(":")[0] if ":" in upper else upper
    # Bracket-form dates also use bare name before "["
    name = name.split("[")[0] if "[" in name else name
    if name not in DRF_EXTRA_NAMES:
        raise ValueError(f"Invalid extra {raw_string}")
    return DRF_EXTRA_NAMES[name]
