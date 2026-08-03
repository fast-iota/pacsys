import datetime

from google.protobuf import timestamp_pb2 as _timestamp_pb2
from pacsys._proto.controls.common.v1 import device_pb2 as _device_pb2
from pacsys._proto.controls.common.v1 import status_pb2 as _status_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ReadingList(_message.Message):
    __slots__ = ("drf",)
    DRF_FIELD_NUMBER: _ClassVar[int]
    drf: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, drf: _Optional[_Iterable[str]] = ...) -> None: ...

class Reading(_message.Message):
    __slots__ = ("timestamp", "data", "status")
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    timestamp: _timestamp_pb2.Timestamp
    data: _device_pb2.Value
    status: _status_pb2.Status
    def __init__(self, timestamp: _Optional[_Union[datetime.datetime, _timestamp_pb2.Timestamp, _Mapping]] = ..., data: _Optional[_Union[_device_pb2.Value, _Mapping]] = ..., status: _Optional[_Union[_status_pb2.Status, _Mapping]] = ...) -> None: ...

class Readings(_message.Message):
    __slots__ = ("reading",)
    READING_FIELD_NUMBER: _ClassVar[int]
    reading: _containers.RepeatedCompositeFieldContainer[Reading]
    def __init__(self, reading: _Optional[_Iterable[_Union[Reading, _Mapping]]] = ...) -> None: ...

class ReadingReply(_message.Message):
    __slots__ = ("index", "readings", "status")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    READINGS_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    index: int
    readings: Readings
    status: _status_pb2.Status
    def __init__(self, index: _Optional[int] = ..., readings: _Optional[_Union[Readings, _Mapping]] = ..., status: _Optional[_Union[_status_pb2.Status, _Mapping]] = ...) -> None: ...

class Setting(_message.Message):
    __slots__ = ("device", "value")
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    device: str
    value: _device_pb2.Value
    def __init__(self, device: _Optional[str] = ..., value: _Optional[_Union[_device_pb2.Value, _Mapping]] = ...) -> None: ...

class SettingList(_message.Message):
    __slots__ = ("setting",)
    SETTING_FIELD_NUMBER: _ClassVar[int]
    setting: _containers.RepeatedCompositeFieldContainer[Setting]
    def __init__(self, setting: _Optional[_Iterable[_Union[Setting, _Mapping]]] = ...) -> None: ...

class SettingReply(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: _containers.RepeatedCompositeFieldContainer[_status_pb2.Status]
    def __init__(self, status: _Optional[_Iterable[_Union[_status_pb2.Status, _Mapping]]] = ...) -> None: ...
