import re

from .device import get_qualified_device, parse_device
from .event import DRF_EVENT, DefaultEvent, parse_event
from .extra import DRF_EXTRA, parse_extra
from .field import DEFAULT_FIELD_FOR_PROPERTY, DRF_FIELD, get_default_field, parse_field
from .property import DRF_PROPERTY, DRF_PROPERTY_ALIASES, get_default_property, parse_property
from .range import BYTE_RANGE, ARRAY_RANGE, parse_range

# 1=DEVIVE, 2=PROPERTY OR FIELD, 3=RANGE, 4=FIELD, 5=EVENT
PATTERN_FULL = re.compile(
    "(?i)(.{3,}?)" + "(?:\\.(\\w+))?" + "(\\[[\\d:]*\\]|\\{[\\d:]*\\})?" + "(?:\\.(\\w+))?" + "(?:@(.+))?" + "$"
)


class DataRequest:
    def __init__(
        self,
        raw_string: str,
        device: str,
        property: DRF_PROPERTY,
        range: ARRAY_RANGE,
        field: DRF_FIELD | None,
        event: DRF_EVENT,
        extra: DRF_EXTRA | None = None,
    ):
        assert isinstance(raw_string, str)
        self.raw_string = raw_string
        assert isinstance(device, str)
        self.device = device
        assert property is None or isinstance(property, DRF_PROPERTY)
        self.property = property
        assert range is None or isinstance(range, (ARRAY_RANGE, BYTE_RANGE))
        self.range = range
        assert field is None or isinstance(field, DRF_FIELD)
        self.field = field
        assert event is None or isinstance(event, DRF_EVENT)
        self.event = event
        assert extra is None or isinstance(extra, DRF_EXTRA)
        self.extra = extra
        self.property_explicit = False

    def __eq__(self, other):
        return (
            self.device == other.device
            and self.property == other.property
            and self.range == other.range
            and self.field == other.field
            and self.event == other.event
            and self.extra == other.extra
        )

    def __str__(self):
        return (
            f"DiscreteRequest[{self.raw_string}] = [{self.device=}] [{self.property=}]"
            f" [{self.range=}]"
            f" [{self.field=}]"
            f" [{self.event=}]"
            f" [{self.extra=}]"
        )

    def __repr__(self):
        return self.__str__()

    @property
    def is_reading(self):
        return self.property == DRF_PROPERTY.READING

    @property
    def is_setting(self):
        return self.property == DRF_PROPERTY.SETTING

    @property
    def is_status(self):
        return self.property == DRF_PROPERTY.STATUS

    @property
    def is_control(self):
        return self.property == DRF_PROPERTY.CONTROL

    @property
    def extra_type(self):
        if self.extra == "FTP":
            return DRF_EXTRA.FTP
        else:
            raise Exception(f"Unknown extra {self.extra}")

    @property
    def parts(self):
        return self.device, self.property, self.range, self.field, self.event

    def to_canonical(
        self,
        device: str | None = None,
        property: DRF_PROPERTY | None = None,
        range: ARRAY_RANGE | None = None,
        field: DRF_FIELD | None = None,
        event: DRF_EVENT | None = None,
        extra: DRF_EXTRA | None = None,
    ) -> str:
        out = ""
        out += device or self.device
        p = property or self.property
        if p is not None:
            out += "."
            out += p.name
        r = range or self.range
        if r is not None:
            rs = str(r)
            out += rs
        f = field or self.field
        if f is not None:
            if DEFAULT_FIELD_FOR_PROPERTY[p] == f:
                pass
            else:
                fs = f.name
                out += f".{fs}"
        e = event or self.event
        if e is not None:
            if e.mode != "U":
                out += f"@{e.raw_string}"
        ex = extra or self.extra
        if ex is not None:
            out += f"<-{ex.name}"
        return out

    def to_qualified(
        self,
        device: str | None = None,
        property: DRF_PROPERTY | None = None,
        range: ARRAY_RANGE | None = None,
        field: DRF_FIELD | None = None,
        event: DRF_EVENT | None = None,
        extra: DRF_EXTRA | None = None,
    ) -> str:
        out = ""
        d = device or self.device
        p = property or self.property
        ds = get_qualified_device(d, p)
        out += ds
        r = range or self.range
        if r is not None:
            rs = str(r)
            out += rs
        f = field or self.field
        if f is not None:
            if DEFAULT_FIELD_FOR_PROPERTY[p] == f:
                pass
            else:
                fs = f.name
                out += f".{fs}"
        e = event or self.event
        if e is not None:
            if e.mode != "U":
                out += f"@{e.raw_string}"
        ex = extra or self.extra
        if ex is not None:
            out += f"<-{ex.name}"
        return out

    def name_as(self, property: DRF_PROPERTY):
        return get_qualified_device(self.device, property)

    def pretty_print(self):
        return (
            f"DiscreteRequest[{self.raw_string}]\n [{self.device=}]\n [{self.property=}]\n"
            f" [{self.range=}]\n"
            f" [{self.field=}]\n"
            f" [{self.event=}]\n"
            f" [{self.extra=}]\n"
        )


def parse_request(device_str: str) -> DataRequest:
    assert device_str is not None
    if "<-" in device_str:
        splits = device_str.split("<-")
        assert len(splits) == 2, f"Invalid drf {device_str}"
        device_str, extra = splits
        extra_obj = parse_extra(extra)
    else:
        extra_obj = None
    match = PATTERN_FULL.match(device_str)
    if match is None:
        raise ValueError(f"{device_str} is not a valid DRF2 device")
    dev, prop, rng, field, event = match.groups()
    dev_obj = parse_device(dev)
    dev_name = dev_obj.canonical_string

    prop_explicit = False
    if prop is None:
        prop_obj = get_default_property(device_str)
    elif prop.upper() in DRF_PROPERTY_ALIASES:
        prop_obj = parse_property(prop)
        prop_explicit = True
    else:
        prop_obj = get_default_property(device_str)
        field = prop

    rng = parse_range(rng)

    if field is None:
        field_obj = get_default_field(prop_obj)
    else:
        field_obj = parse_field(field.upper())

    if event is None:
        event_obj = DefaultEvent()
    else:
        event_obj = parse_event(event)

    req = DataRequest(device_str, dev_name, prop_obj, rng, field_obj, event_obj, extra_obj)
    req.property_explicit = prop_explicit
    return req
