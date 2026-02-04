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

def parse_extra(raw_string: str) -> DRF_EXTRA:
    if raw_string not in DRF_EXTRA_NAMES:
        raise ValueError(f"Invalid extra {raw_string}")
    return DRF_EXTRA_NAMES[raw_string]
