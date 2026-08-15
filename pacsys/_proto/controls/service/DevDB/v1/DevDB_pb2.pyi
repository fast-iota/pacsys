from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PlotSelector(_message.Message):
    __slots__ = ("id",)
    ID_FIELD_NUMBER: _ClassVar[int]
    id: int
    def __init__(self, id: _Optional[int] = ...) -> None: ...

class PlotConfigSpecification(_message.Message):
    __slots__ = ("id", "name", "config")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    id: int
    name: str
    config: str
    def __init__(self, id: _Optional[int] = ..., name: _Optional[str] = ..., config: _Optional[str] = ...) -> None: ...

class PlotConfigResults(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedCompositeFieldContainer[PlotConfigSpecification]
    def __init__(self, data: _Optional[_Iterable[_Union[PlotConfigSpecification, _Mapping]]] = ...) -> None: ...

class PlotConfig(_message.Message):
    __slots__ = ("config",)
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    config: str
    def __init__(self, config: _Optional[str] = ...) -> None: ...

class PlotConfigResult(_message.Message):
    __slots__ = ("config", "user_config", "errMsg")
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    USER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    ERRMSG_FIELD_NUMBER: _ClassVar[int]
    config: PlotConfigResults
    user_config: PlotConfig
    errMsg: str
    def __init__(self, config: _Optional[_Union[PlotConfigResults, _Mapping]] = ..., user_config: _Optional[_Union[PlotConfig, _Mapping]] = ..., errMsg: _Optional[str] = ...) -> None: ...

class DeviceList(_message.Message):
    __slots__ = ("device",)
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    device: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, device: _Optional[_Iterable[str]] = ...) -> None: ...

class Property(_message.Message):
    __slots__ = ("primary_units", "common_units", "min_val", "max_val", "p_index", "c_index", "coeff", "is_step_motor", "is_destructive_read", "is_fe_scaling", "is_contr_setting", "is_knobbable")
    PRIMARY_UNITS_FIELD_NUMBER: _ClassVar[int]
    COMMON_UNITS_FIELD_NUMBER: _ClassVar[int]
    MIN_VAL_FIELD_NUMBER: _ClassVar[int]
    MAX_VAL_FIELD_NUMBER: _ClassVar[int]
    P_INDEX_FIELD_NUMBER: _ClassVar[int]
    C_INDEX_FIELD_NUMBER: _ClassVar[int]
    COEFF_FIELD_NUMBER: _ClassVar[int]
    IS_STEP_MOTOR_FIELD_NUMBER: _ClassVar[int]
    IS_DESTRUCTIVE_READ_FIELD_NUMBER: _ClassVar[int]
    IS_FE_SCALING_FIELD_NUMBER: _ClassVar[int]
    IS_CONTR_SETTING_FIELD_NUMBER: _ClassVar[int]
    IS_KNOBBABLE_FIELD_NUMBER: _ClassVar[int]
    primary_units: str
    common_units: str
    min_val: float
    max_val: float
    p_index: int
    c_index: int
    coeff: _containers.RepeatedScalarFieldContainer[float]
    is_step_motor: bool
    is_destructive_read: bool
    is_fe_scaling: bool
    is_contr_setting: bool
    is_knobbable: bool
    def __init__(self, primary_units: _Optional[str] = ..., common_units: _Optional[str] = ..., min_val: _Optional[float] = ..., max_val: _Optional[float] = ..., p_index: _Optional[int] = ..., c_index: _Optional[int] = ..., coeff: _Optional[_Iterable[float]] = ..., is_step_motor: bool = ..., is_destructive_read: bool = ..., is_fe_scaling: bool = ..., is_contr_setting: bool = ..., is_knobbable: bool = ...) -> None: ...

class DigitalStatusItem(_message.Message):
    __slots__ = ("mask_val", "match_val", "invert", "short_name", "long_name", "true_str", "true_color", "true_char", "false_str", "false_color", "false_char")
    MASK_VAL_FIELD_NUMBER: _ClassVar[int]
    MATCH_VAL_FIELD_NUMBER: _ClassVar[int]
    INVERT_FIELD_NUMBER: _ClassVar[int]
    SHORT_NAME_FIELD_NUMBER: _ClassVar[int]
    LONG_NAME_FIELD_NUMBER: _ClassVar[int]
    TRUE_STR_FIELD_NUMBER: _ClassVar[int]
    TRUE_COLOR_FIELD_NUMBER: _ClassVar[int]
    TRUE_CHAR_FIELD_NUMBER: _ClassVar[int]
    FALSE_STR_FIELD_NUMBER: _ClassVar[int]
    FALSE_COLOR_FIELD_NUMBER: _ClassVar[int]
    FALSE_CHAR_FIELD_NUMBER: _ClassVar[int]
    mask_val: int
    match_val: int
    invert: bool
    short_name: str
    long_name: str
    true_str: str
    true_color: int
    true_char: str
    false_str: str
    false_color: int
    false_char: str
    def __init__(self, mask_val: _Optional[int] = ..., match_val: _Optional[int] = ..., invert: bool = ..., short_name: _Optional[str] = ..., long_name: _Optional[str] = ..., true_str: _Optional[str] = ..., true_color: _Optional[int] = ..., true_char: _Optional[str] = ..., false_str: _Optional[str] = ..., false_color: _Optional[int] = ..., false_char: _Optional[str] = ...) -> None: ...

class DigitalExtStatusItem(_message.Message):
    __slots__ = ("bit_no", "color0", "name0", "color1", "name1", "description")
    BIT_NO_FIELD_NUMBER: _ClassVar[int]
    COLOR0_FIELD_NUMBER: _ClassVar[int]
    NAME0_FIELD_NUMBER: _ClassVar[int]
    COLOR1_FIELD_NUMBER: _ClassVar[int]
    NAME1_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    bit_no: int
    color0: int
    name0: str
    color1: int
    name1: str
    description: str
    def __init__(self, bit_no: _Optional[int] = ..., color0: _Optional[int] = ..., name0: _Optional[str] = ..., color1: _Optional[int] = ..., name1: _Optional[str] = ..., description: _Optional[str] = ...) -> None: ...

class DigitalStatus(_message.Message):
    __slots__ = ("bits", "ext_bits")
    BITS_FIELD_NUMBER: _ClassVar[int]
    EXT_BITS_FIELD_NUMBER: _ClassVar[int]
    bits: _containers.RepeatedCompositeFieldContainer[DigitalStatusItem]
    ext_bits: _containers.RepeatedCompositeFieldContainer[DigitalExtStatusItem]
    def __init__(self, bits: _Optional[_Iterable[_Union[DigitalStatusItem, _Mapping]]] = ..., ext_bits: _Optional[_Iterable[_Union[DigitalExtStatusItem, _Mapping]]] = ...) -> None: ...

class DigitalControlItem(_message.Message):
    __slots__ = ("value", "short_name", "long_name")
    VALUE_FIELD_NUMBER: _ClassVar[int]
    SHORT_NAME_FIELD_NUMBER: _ClassVar[int]
    LONG_NAME_FIELD_NUMBER: _ClassVar[int]
    value: int
    short_name: str
    long_name: str
    def __init__(self, value: _Optional[int] = ..., short_name: _Optional[str] = ..., long_name: _Optional[str] = ...) -> None: ...

class DigitalControl(_message.Message):
    __slots__ = ("cmds",)
    CMDS_FIELD_NUMBER: _ClassVar[int]
    cmds: _containers.RepeatedCompositeFieldContainer[DigitalControlItem]
    def __init__(self, cmds: _Optional[_Iterable[_Union[DigitalControlItem, _Mapping]]] = ...) -> None: ...

class DeviceInfo(_message.Message):
    __slots__ = ("device_index", "description", "reading", "setting", "control", "status")
    DEVICE_INDEX_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    READING_FIELD_NUMBER: _ClassVar[int]
    SETTING_FIELD_NUMBER: _ClassVar[int]
    CONTROL_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    device_index: int
    description: str
    reading: Property
    setting: Property
    control: DigitalControl
    status: DigitalStatus
    def __init__(self, device_index: _Optional[int] = ..., description: _Optional[str] = ..., reading: _Optional[_Union[Property, _Mapping]] = ..., setting: _Optional[_Union[Property, _Mapping]] = ..., control: _Optional[_Union[DigitalControl, _Mapping]] = ..., status: _Optional[_Union[DigitalStatus, _Mapping]] = ...) -> None: ...

class InfoEntry(_message.Message):
    __slots__ = ("name", "device", "errMsg")
    NAME_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    ERRMSG_FIELD_NUMBER: _ClassVar[int]
    name: str
    device: DeviceInfo
    errMsg: str
    def __init__(self, name: _Optional[str] = ..., device: _Optional[_Union[DeviceInfo, _Mapping]] = ..., errMsg: _Optional[str] = ...) -> None: ...

class DeviceInfoReply(_message.Message):
    __slots__ = ("set",)
    SET_FIELD_NUMBER: _ClassVar[int]
    set: _containers.RepeatedCompositeFieldContainer[InfoEntry]
    def __init__(self, set: _Optional[_Iterable[_Union[InfoEntry, _Mapping]]] = ...) -> None: ...

class AlarmBlock(_message.Message):
    __slots__ = ("di", "pi", "status", "min_or_nom", "max_or_tol", "tries_needed", "tries_now", "clock_event_no", "subfunction_code", "specific_data", "segment")
    DI_FIELD_NUMBER: _ClassVar[int]
    PI_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    MIN_OR_NOM_FIELD_NUMBER: _ClassVar[int]
    MAX_OR_TOL_FIELD_NUMBER: _ClassVar[int]
    TRIES_NEEDED_FIELD_NUMBER: _ClassVar[int]
    TRIES_NOW_FIELD_NUMBER: _ClassVar[int]
    CLOCK_EVENT_NO_FIELD_NUMBER: _ClassVar[int]
    SUBFUNCTION_CODE_FIELD_NUMBER: _ClassVar[int]
    SPECIFIC_DATA_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_FIELD_NUMBER: _ClassVar[int]
    di: int
    pi: int
    status: int
    min_or_nom: int
    max_or_tol: int
    tries_needed: int
    tries_now: int
    clock_event_no: int
    subfunction_code: int
    specific_data: str
    segment: int
    def __init__(self, di: _Optional[int] = ..., pi: _Optional[int] = ..., status: _Optional[int] = ..., min_or_nom: _Optional[int] = ..., max_or_tol: _Optional[int] = ..., tries_needed: _Optional[int] = ..., tries_now: _Optional[int] = ..., clock_event_no: _Optional[int] = ..., subfunction_code: _Optional[int] = ..., specific_data: _Optional[str] = ..., segment: _Optional[int] = ...) -> None: ...

class DeviceDigitalAlarm(_message.Message):
    __slots__ = ("di", "condition", "mask", "alarm_text_id")
    DI_FIELD_NUMBER: _ClassVar[int]
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    MASK_FIELD_NUMBER: _ClassVar[int]
    ALARM_TEXT_ID_FIELD_NUMBER: _ClassVar[int]
    di: int
    condition: int
    mask: int
    alarm_text_id: int
    def __init__(self, di: _Optional[int] = ..., condition: _Optional[int] = ..., mask: _Optional[int] = ..., alarm_text_id: _Optional[int] = ...) -> None: ...

class DeviceAnalogAlarm(_message.Message):
    __slots__ = ("di", "alarm_text_id")
    DI_FIELD_NUMBER: _ClassVar[int]
    ALARM_TEXT_ID_FIELD_NUMBER: _ClassVar[int]
    di: int
    alarm_text_id: int
    def __init__(self, di: _Optional[int] = ..., alarm_text_id: _Optional[int] = ...) -> None: ...

class AlarmTextIdList(_message.Message):
    __slots__ = ("alarm_text_id",)
    ALARM_TEXT_ID_FIELD_NUMBER: _ClassVar[int]
    alarm_text_id: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, alarm_text_id: _Optional[_Iterable[int]] = ...) -> None: ...

class DeviceAlarmText(_message.Message):
    __slots__ = ("alarm_text_id", "length", "priority", "hand_code", "sound_id", "speech_id", "spare", "text", "url")
    ALARM_TEXT_ID_FIELD_NUMBER: _ClassVar[int]
    LENGTH_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    HAND_CODE_FIELD_NUMBER: _ClassVar[int]
    SOUND_ID_FIELD_NUMBER: _ClassVar[int]
    SPEECH_ID_FIELD_NUMBER: _ClassVar[int]
    SPARE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    URL_FIELD_NUMBER: _ClassVar[int]
    alarm_text_id: int
    length: int
    priority: int
    hand_code: int
    sound_id: int
    speech_id: int
    spare: int
    text: str
    url: str
    def __init__(self, alarm_text_id: _Optional[int] = ..., length: _Optional[int] = ..., priority: _Optional[int] = ..., hand_code: _Optional[int] = ..., sound_id: _Optional[int] = ..., speech_id: _Optional[int] = ..., spare: _Optional[int] = ..., text: _Optional[str] = ..., url: _Optional[str] = ...) -> None: ...

class DeviceAlarmTextList(_message.Message):
    __slots__ = ("device_alarm_text",)
    DEVICE_ALARM_TEXT_FIELD_NUMBER: _ClassVar[int]
    device_alarm_text: _containers.RepeatedCompositeFieldContainer[DeviceAlarmText]
    def __init__(self, device_alarm_text: _Optional[_Iterable[_Union[DeviceAlarmText, _Mapping]]] = ...) -> None: ...

class AlarmInfo(_message.Message):
    __slots__ = ("device_name", "alarm_block", "device_analog_alarm", "device_digital_alarm")
    DEVICE_NAME_FIELD_NUMBER: _ClassVar[int]
    ALARM_BLOCK_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ANALOG_ALARM_FIELD_NUMBER: _ClassVar[int]
    DEVICE_DIGITAL_ALARM_FIELD_NUMBER: _ClassVar[int]
    device_name: str
    alarm_block: AlarmBlock
    device_analog_alarm: DeviceAnalogAlarm
    device_digital_alarm: DeviceDigitalAlarm
    def __init__(self, device_name: _Optional[str] = ..., alarm_block: _Optional[_Union[AlarmBlock, _Mapping]] = ..., device_analog_alarm: _Optional[_Union[DeviceAnalogAlarm, _Mapping]] = ..., device_digital_alarm: _Optional[_Union[DeviceDigitalAlarm, _Mapping]] = ...) -> None: ...

class AlarmInfoReply(_message.Message):
    __slots__ = ("alarm_info",)
    ALARM_INFO_FIELD_NUMBER: _ClassVar[int]
    alarm_info: _containers.RepeatedCompositeFieldContainer[AlarmInfo]
    def __init__(self, alarm_info: _Optional[_Iterable[_Union[AlarmInfo, _Mapping]]] = ...) -> None: ...
