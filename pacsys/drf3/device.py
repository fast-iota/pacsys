import re

from .property import DRF_PROPERTY

# ACNET device grammar: qualifier char at index 1, then 1-62 name chars (Java DeviceParser)
ACNET_NAME_GRAMMAR = "[A-Z0#][:?_|&@$~^#!][A-Z0-9_:-]{1,62}"
PATTERN_NAME = re.compile("(?i)" + ACNET_NAME_GRAMMAR)


class Device:
    def __init__(self, raw_string: str, canonical_string: str, is_acnet: bool = True):
        self.raw_string = raw_string
        self.canonical_string = canonical_string
        self.is_acnet = is_acnet

    @property
    def canonical(self):
        return self.canonical_string

    def qualified_name(self, prop: DRF_PROPERTY):
        return get_qualified_device(self.raw_string, prop)


def get_qualified_device(device_str: str, prop: DRF_PROPERTY):
    if PATTERN_NAME.fullmatch(device_str) is None:
        raise ValueError(f"Cannot qualify non-ACNET device {device_str!r}")
    if prop not in DRF_PROPERTY:
        raise ValueError(f"prop must be a DRF_PROPERTY member, got {prop!r}")
    ext = prop.value
    if ext is None:
        raise ValueError(
            f"Property {prop.name} has no qualifier character and cannot be used in qualified device names"
        )
    ld = list(device_str)
    ld[1] = ext
    return "".join(ld)


def parse_device(raw_string, assume_epics: bool = True) -> Device:
    if raw_string is None:
        raise ValueError("raw_string must not be None")
    if PATTERN_NAME.fullmatch(raw_string) is None:
        if assume_epics:
            return Device(raw_string=raw_string, canonical_string=raw_string, is_acnet=False)
        raise ValueError(f"{raw_string} is not a valid device")
    ld = list(raw_string)
    ld[1] = ":"
    return Device(raw_string=raw_string, canonical_string="".join(ld), is_acnet=True)
