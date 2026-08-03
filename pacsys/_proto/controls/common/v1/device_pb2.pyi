from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Value(_message.Message):
    __slots__ = ("scalar", "scalarArr", "raw", "text", "textArr", "anaAlarm", "digAlarm", "basicStatus")
    class ScalarArray(_message.Message):
        __slots__ = ("value",)
        VALUE_FIELD_NUMBER: _ClassVar[int]
        value: _containers.RepeatedScalarFieldContainer[float]
        def __init__(self, value: _Optional[_Iterable[float]] = ...) -> None: ...
    class TextArray(_message.Message):
        __slots__ = ("value",)
        VALUE_FIELD_NUMBER: _ClassVar[int]
        value: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, value: _Optional[_Iterable[str]] = ...) -> None: ...
    class AnalogAlarm(_message.Message):
        __slots__ = ("minimum", "maximum", "alarmEnable", "alarmStatus", "abort", "abortInhibit", "triesNeeded", "triesNow")
        MINIMUM_FIELD_NUMBER: _ClassVar[int]
        MAXIMUM_FIELD_NUMBER: _ClassVar[int]
        ALARMENABLE_FIELD_NUMBER: _ClassVar[int]
        ALARMSTATUS_FIELD_NUMBER: _ClassVar[int]
        ABORT_FIELD_NUMBER: _ClassVar[int]
        ABORTINHIBIT_FIELD_NUMBER: _ClassVar[int]
        TRIESNEEDED_FIELD_NUMBER: _ClassVar[int]
        TRIESNOW_FIELD_NUMBER: _ClassVar[int]
        minimum: float
        maximum: float
        alarmEnable: bool
        alarmStatus: bool
        abort: bool
        abortInhibit: bool
        triesNeeded: int
        triesNow: int
        def __init__(self, minimum: _Optional[float] = ..., maximum: _Optional[float] = ..., alarmEnable: bool = ..., alarmStatus: bool = ..., abort: bool = ..., abortInhibit: bool = ..., triesNeeded: _Optional[int] = ..., triesNow: _Optional[int] = ...) -> None: ...
    class DigitalAlarm(_message.Message):
        __slots__ = ("nominal", "mask", "alarmEnable", "alarmStatus", "abort", "abortInhibit", "triesNeeded", "triesNow")
        NOMINAL_FIELD_NUMBER: _ClassVar[int]
        MASK_FIELD_NUMBER: _ClassVar[int]
        ALARMENABLE_FIELD_NUMBER: _ClassVar[int]
        ALARMSTATUS_FIELD_NUMBER: _ClassVar[int]
        ABORT_FIELD_NUMBER: _ClassVar[int]
        ABORTINHIBIT_FIELD_NUMBER: _ClassVar[int]
        TRIESNEEDED_FIELD_NUMBER: _ClassVar[int]
        TRIESNOW_FIELD_NUMBER: _ClassVar[int]
        nominal: int
        mask: int
        alarmEnable: bool
        alarmStatus: bool
        abort: bool
        abortInhibit: bool
        triesNeeded: int
        triesNow: int
        def __init__(self, nominal: _Optional[int] = ..., mask: _Optional[int] = ..., alarmEnable: bool = ..., alarmStatus: bool = ..., abort: bool = ..., abortInhibit: bool = ..., triesNeeded: _Optional[int] = ..., triesNow: _Optional[int] = ...) -> None: ...
    class BasicStatus(_message.Message):
        __slots__ = ("value",)
        class ValueEntry(_message.Message):
            __slots__ = ("key", "value")
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: str
            def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
        VALUE_FIELD_NUMBER: _ClassVar[int]
        value: _containers.ScalarMap[str, str]
        def __init__(self, value: _Optional[_Mapping[str, str]] = ...) -> None: ...
    SCALAR_FIELD_NUMBER: _ClassVar[int]
    SCALARARR_FIELD_NUMBER: _ClassVar[int]
    RAW_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    TEXTARR_FIELD_NUMBER: _ClassVar[int]
    ANAALARM_FIELD_NUMBER: _ClassVar[int]
    DIGALARM_FIELD_NUMBER: _ClassVar[int]
    BASICSTATUS_FIELD_NUMBER: _ClassVar[int]
    scalar: float
    scalarArr: Value.ScalarArray
    raw: bytes
    text: str
    textArr: Value.TextArray
    anaAlarm: Value.AnalogAlarm
    digAlarm: Value.DigitalAlarm
    basicStatus: Value.BasicStatus
    def __init__(self, scalar: _Optional[float] = ..., scalarArr: _Optional[_Union[Value.ScalarArray, _Mapping]] = ..., raw: _Optional[bytes] = ..., text: _Optional[str] = ..., textArr: _Optional[_Union[Value.TextArray, _Mapping]] = ..., anaAlarm: _Optional[_Union[Value.AnalogAlarm, _Mapping]] = ..., digAlarm: _Optional[_Union[Value.DigitalAlarm, _Mapping]] = ..., basicStatus: _Optional[_Union[Value.BasicStatus, _Mapping]] = ...) -> None: ...
