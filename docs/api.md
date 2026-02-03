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

::: pacsys.acl
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
        - name
        - units
        - timestamp
        - is_success
        - is_error
        - is_warning
        - ok
        - message

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
        - read
        - get
        - with_backend
        - with_event

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
