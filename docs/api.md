# API Reference

Complete reference for pacsys functions, classes, and types.

---

## Simple API

These functions use a global backend that is automatically initialized on first use.

::: pacsys.read
    options:
      show_root_heading: true
      heading_level: 3

::: pacsys.get
    options:
      show_root_heading: true
      heading_level: 3

::: pacsys.get_many
    options:
      show_root_heading: true
      heading_level: 3

::: pacsys.write
    options:
      show_root_heading: true
      heading_level: 3

::: pacsys.write_many
    options:
      show_root_heading: true
      heading_level: 3

::: pacsys.subscribe
    options:
      show_root_heading: true
      heading_level: 3

::: pacsys.configure
    options:
      show_root_heading: true
      heading_level: 3

::: pacsys.shutdown
    options:
      show_root_heading: true
      heading_level: 3

---

## Backend Factories

Create explicit backend instances for more control.

::: pacsys.dpm
    options:
      show_root_heading: true
      heading_level: 3

::: pacsys.grpc
    options:
      show_root_heading: true
      heading_level: 3

::: pacsys.dmq
    options:
      show_root_heading: true
      heading_level: 3

::: pacsys.acl
    options:
      show_root_heading: true
      heading_level: 3

::: pacsys.dpm_http
    options:
      show_root_heading: true
      heading_level: 3

::: pacsys.devdb
    options:
      show_root_heading: true
      heading_level: 3

::: pacsys.supervised
    options:
      show_root_heading: true
      heading_level: 3

---

## Types

### Reading

::: pacsys.types.Reading
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - value
        - value_type
        - name
        - drf
        - units
        - timestamp
        - cycle
        - facility_code
        - error_code
        - message
        - is_success
        - is_error
        - is_warning
        - ok

### WriteResult

::: pacsys.types.WriteResult
    options:
      show_root_heading: true
      heading_level: 3

### SubscriptionHandle

::: pacsys.types.SubscriptionHandle
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - readings
        - stop
        - stopped

### ValueType

::: pacsys.types.ValueType
    options:
      show_root_heading: true
      heading_level: 3

---

## Authentication

### KerberosAuth

::: pacsys.auth.KerberosAuth
    options:
      show_root_heading: true
      heading_level: 3

### JWTAuth

::: pacsys.auth.JWTAuth
    options:
      show_root_heading: true
      heading_level: 3

---

## Errors

### DeviceError

::: pacsys.errors.DeviceError
    options:
      show_root_heading: true
      heading_level: 3

### AuthenticationError

::: pacsys.errors.AuthenticationError
    options:
      show_root_heading: true
      heading_level: 3

### ReadError

::: pacsys.errors.ReadError
    options:
      show_root_heading: true
      heading_level: 3

### ACLError

::: pacsys.errors.ACLError
    options:
      show_root_heading: true
      heading_level: 3

---

## Device Classes

Object-oriented interface for device access.

### Device

::: pacsys.device.Device
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - drf
        - name
        - request
        - has_event
        - is_periodic
        - read
        - get
        - setting
        - status
        - analog_alarm
        - digital_alarm
        - description
        - digital_status
        - info
        - write
        - control
        - on
        - off
        - reset
        - set_analog_alarm
        - set_digital_alarm
        - subscribe
        - with_backend
        - with_event
        - with_range

### ScalarDevice

::: pacsys.device.ScalarDevice
    options:
      show_root_heading: true
      heading_level: 3

### ArrayDevice

::: pacsys.device.ArrayDevice
    options:
      show_root_heading: true
      heading_level: 3

### TextDevice

::: pacsys.device.TextDevice
    options:
      show_root_heading: true
      heading_level: 3

---

## SSH Utility

### SSHClient

::: pacsys.ssh.SSHClient
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - exec
        - exec_stream
        - exec_many
        - forward
        - sftp
        - open_channel
        - remote_process
        - acl_session
        - acl
        - close

### RemoteProcess

::: pacsys.ssh.RemoteProcess
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - send_line
        - send_bytes
        - read_until
        - read_for
        - alive
        - close

### SSHHop

::: pacsys.ssh.SSHHop
    options:
      show_root_heading: true
      heading_level: 3

### CommandResult

::: pacsys.ssh.CommandResult
    options:
      show_root_heading: true
      heading_level: 3

### ACLSession

::: pacsys.acl_session.ACLSession
    options:
      show_root_heading: true
      heading_level: 3
      members:
        - send
        - close

### SSH Exceptions

::: pacsys.ssh.SSHError
    options:
      show_root_heading: true
      heading_level: 4

::: pacsys.ssh.SSHConnectionError
    options:
      show_root_heading: true
      heading_level: 4

::: pacsys.ssh.SSHCommandError
    options:
      show_root_heading: true
      heading_level: 4

::: pacsys.ssh.SSHTimeoutError
    options:
      show_root_heading: true
      heading_level: 4
