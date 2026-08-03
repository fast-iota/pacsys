from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Status(_message.Message):
    __slots__ = ("facility_code", "status_code", "message")
    FACILITY_CODE_FIELD_NUMBER: _ClassVar[int]
    STATUS_CODE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    facility_code: int
    status_code: int
    message: str
    def __init__(self, facility_code: _Optional[int] = ..., status_code: _Optional[int] = ..., message: _Optional[str] = ...) -> None: ...
