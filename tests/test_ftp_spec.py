"""Tests for pacsys.acnet.ftp_spec -- FTP/snapshot event string parser."""

import pytest

from pacsys.acnet.ftp_spec import (
    ClockSample,
    ClockTrigger,
    DeviceTrigger,
    ExternalSample,
    ExternalTrigger,
    FTPSpec,
    PeriodicSample,
    SnapshotSpec,
    StateTrigger,
    parse_ftp_event,
)


# ---- FTP basic ----


class TestFTPParsing:
    def test_ftp_basic(self):
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=e,2,1000;null")
        assert isinstance(spec, FTPSpec)
        assert spec.rate_hz == 60.0
        assert spec.duration_s == 1.0
        assert isinstance(spec.trigger, ClockTrigger)
        assert spec.trigger.events == (0x2,)
        assert spec.trigger.delay_ms == 1000
        assert spec.rearm is None

    def test_ftp_null_trigger_and_rearm(self):
        spec = parse_ftp_event("f,type=ftp,rate=15.0,dur=2.5;null;null")
        assert isinstance(spec, FTPSpec)
        assert spec.rate_hz == 15.0
        assert spec.duration_s == 2.5
        assert spec.trigger is None
        assert spec.rearm is None

    def test_ftp_no_trigger_section(self):
        spec = parse_ftp_event("f,type=ftp,rate=100.0,dur=0.5")
        assert isinstance(spec, FTPSpec)
        assert spec.trigger is None
        assert spec.rearm is None


# ---- Snapshot basic ----


class TestSnapshotParsing:
    def test_snapshot_periodic_sample(self):
        spec = parse_ftp_event("f,type=snp,rate=2048,dur=1.0,npts=2048,pref=rate,smpl=p;trig=e,2,0;null")
        assert isinstance(spec, SnapshotSpec)
        assert spec.rate_hz == 2048
        assert spec.duration_s == 1.0
        assert spec.num_points == 2048
        assert spec.preference == "rate"
        assert isinstance(spec.sample, PeriodicSample)
        assert isinstance(spec.trigger, ClockTrigger)
        assert spec.trigger.events == (0x2,)
        assert spec.trigger.delay_ms == 0
        assert spec.rearm is None

    def test_snapshot_clock_sample(self):
        spec = parse_ftp_event("f,type=snp,rate=1000,dur=2.0,npts=1000,pref=dur,smpl=e,02,ff,ff,ff;trig=e,2,0;null")
        assert isinstance(spec, SnapshotSpec)
        assert isinstance(spec.sample, ClockSample)
        assert spec.sample.events == (0x02, 0xFF, 0xFF, 0xFF)

    def test_snapshot_external_sample(self):
        spec = parse_ftp_event("f,type=snp,rate=500,dur=1.0,npts=500,pref=rate,smpl=x,mod=2;trig=e,2,0;null")
        assert isinstance(spec, SnapshotSpec)
        assert isinstance(spec.sample, ExternalSample)
        assert spec.sample.modifier == 2

    def test_snapshot_external_sample_positional(self):
        """Accept bare positional modifier for smpl=x (consistent with trig=x)."""
        spec = parse_ftp_event("f,type=snp,rate=500,dur=1.0,npts=500,pref=rate,smpl=x,2;trig=e,2,0;null")
        assert isinstance(spec.sample, ExternalSample)
        assert spec.sample.modifier == 2

    def test_snapshot_rate_truncated_to_int(self):
        """Java truncates double to int for snap rate."""
        spec = parse_ftp_event("f,type=snp,rate=100.7,dur=1.0,npts=100,pref=rate,smpl=p;trig=e,2,0;null")
        assert spec.rate_hz == 100  # int(float("100.7"))

    def test_snapshot_preference_both(self):
        spec = parse_ftp_event("f,type=snp,rate=1000,dur=1.0,npts=1000,pref=both,smpl=p;null;null")
        assert spec.preference == "both"

    def test_snapshot_preference_none(self):
        spec = parse_ftp_event("f,type=snp,rate=1000,dur=1.0,npts=1000,pref=none,smpl=p;null;null")
        assert spec.preference == "none"


# ---- Trigger types ----


class TestTriggerParsing:
    def test_clock_trigger_single_event(self):
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=e,02,0;null")
        assert isinstance(spec.trigger, ClockTrigger)
        assert spec.trigger.events == (0x02,)
        assert spec.trigger.delay_ms == 0

    def test_clock_trigger_multiple_events(self):
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=e,00,FE,FE,FE,1000;null")
        assert isinstance(spec.trigger, ClockTrigger)
        assert spec.trigger.events == (0x00, 0xFE, 0xFE, 0xFE)
        assert spec.trigger.delay_ms == 1000

    def test_clock_trigger_8_events(self):
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=e,01,02,03,04,05,06,07,08,500;null")
        assert isinstance(spec.trigger, ClockTrigger)
        assert spec.trigger.events == (0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08)
        assert spec.trigger.delay_ms == 500

    def test_device_trigger_canonical(self):
        """Java canonical format: trig=d,<device>,mask=<hex>,val=<hex>,dly=<ms>."""
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=d,M:OUTTMP,mask=ff,val=1,dly=100;null")
        assert isinstance(spec.trigger, DeviceTrigger)
        assert spec.trigger.device == "M:OUTTMP"
        assert spec.trigger.mask == 0xFF
        assert spec.trigger.value == 1
        assert spec.trigger.delay_ms == 100

    def test_device_trigger_defaults(self):
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=d,Z:ACLTST;null")
        assert isinstance(spec.trigger, DeviceTrigger)
        assert spec.trigger.device == "Z:ACLTST"
        assert spec.trigger.mask == 0
        assert spec.trigger.value == 0
        assert spec.trigger.delay_ms == 0

    def test_device_trigger_partial_fields(self):
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=d,G:SYBSET,mask=f;null")
        assert isinstance(spec.trigger, DeviceTrigger)
        assert spec.trigger.device == "G:SYBSET"
        assert spec.trigger.mask == 0xF
        assert spec.trigger.value == 0
        assert spec.trigger.delay_ms == 0

    def test_external_trigger_mod_format(self):
        """Java canonical format: trig=x,mod=<n>."""
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=x,mod=2;null")
        assert isinstance(spec.trigger, ExternalTrigger)
        assert spec.trigger.modifier == 2

    def test_external_trigger_positional(self):
        """Also accept bare positional modifier for convenience."""
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=x,2;null")
        assert isinstance(spec.trigger, ExternalTrigger)
        assert spec.trigger.modifier == 2

    def test_external_trigger_default(self):
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=x;null")
        assert isinstance(spec.trigger, ExternalTrigger)
        assert spec.trigger.modifier == 0

    def test_state_trigger(self):
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=s,M:OUTTMP,1,500,>=;null")
        assert isinstance(spec.trigger, StateTrigger)
        assert spec.trigger.device == "M:OUTTMP"
        assert spec.trigger.value == 1
        assert spec.trigger.delay_ms == 500
        assert spec.trigger.flag == ">="

    def test_state_trigger_default_flag(self):
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=s,M:OUTTMP,1,0;null")
        assert isinstance(spec.trigger, StateTrigger)
        assert spec.trigger.flag == "="


# ---- ReArm ----


class TestReArmParsing:
    def test_rearm_enabled_with_delay(self):
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=e,2,0;rearm=true,dly=p,60000,false,nmhr=30")
        assert spec.rearm is not None
        assert spec.rearm.enabled is True
        assert spec.rearm.delay_event == "p,60000,false"
        assert spec.rearm.max_per_hour == 30

    def test_rearm_disabled(self):
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=e,2,0;rearm=false")
        assert spec.rearm is not None
        assert spec.rearm.enabled is False
        assert spec.rearm.delay_event is None
        assert spec.rearm.max_per_hour == -1

    def test_rearm_dly_null_maps_to_none(self):
        """Java emits literal 'null' for absent delay events."""
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=e,2,0;rearm=true,dly=null,nmhr=5")
        assert spec.rearm is not None
        assert spec.rearm.enabled is True
        assert spec.rearm.delay_event is None
        assert spec.rearm.max_per_hour == 5

    def test_rearm_complex_delay_with_commas(self):
        """Delay event string containing commas (e.g. clock event 'e,1,100')."""
        spec = parse_ftp_event("f,type=ftp,rate=1.0,dur=1.0;trig=e,1,0;rearm=true,dly=e,1,100,nmhr=5")
        assert spec.rearm is not None
        assert spec.rearm.delay_event == "e,1,100"
        assert spec.rearm.max_per_hour == 5

    def test_rearm_null(self):
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=e,2,0;null")
        assert spec.rearm is None


# ---- Java-style complex string ----


class TestJavaTestString:
    def test_full_java_string(self):
        """Test the comprehensive Java-style event string."""
        s = (
            "f,type=snp,rate=100,dur=20.48,npts=2048,pref=rate,smpl=p;"
            "trig=e,00,FE,FE,FE,1000;"
            "rearm=true,dly=p,60000,false,nmhr=30"
        )
        spec = parse_ftp_event(s)
        assert isinstance(spec, SnapshotSpec)
        assert spec.rate_hz == 100
        assert spec.duration_s == 20.48
        assert spec.num_points == 2048
        assert spec.preference == "rate"
        assert isinstance(spec.sample, PeriodicSample)

        assert isinstance(spec.trigger, ClockTrigger)
        assert spec.trigger.events == (0x00, 0xFE, 0xFE, 0xFE)
        assert spec.trigger.delay_ms == 1000

        assert spec.rearm is not None
        assert spec.rearm.enabled is True
        assert spec.rearm.delay_event == "p,60000,false"
        assert spec.rearm.max_per_hour == 30


# ---- Frozen dataclasses ----


class TestFrozenDataclasses:
    def test_ftp_spec_frozen(self):
        spec = parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;null;null")
        with pytest.raises(AttributeError):
            spec.rate_hz = 99.0  # type: ignore[misc]

    def test_snapshot_spec_frozen(self):
        spec = parse_ftp_event("f,type=snp,rate=100,dur=1.0,npts=100,pref=rate,smpl=p;null;null")
        with pytest.raises(AttributeError):
            spec.rate_hz = 99  # type: ignore[misc]


# ---- Error cases ----


class TestErrors:
    def test_empty_string(self):
        with pytest.raises(ValueError, match="too short"):
            parse_ftp_event("")

    def test_short_string(self):
        with pytest.raises(ValueError, match="too short"):
            parse_ftp_event("f,type=ft")

    def test_bad_type(self):
        with pytest.raises(ValueError, match="Unknown event type"):
            parse_ftp_event("f,type=xyz,rate=60.0,dur=1.0;null;null")

    def test_invalid_preference(self):
        with pytest.raises(ValueError, match="Invalid preference"):
            parse_ftp_event("f,type=snp,rate=100,dur=1.0,npts=100,pref=bad,smpl=p;null;null")

    def test_unknown_trigger_type(self):
        with pytest.raises(ValueError, match="Unknown trigger type"):
            parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=z;null")

    def test_unknown_sample_type(self):
        with pytest.raises(ValueError, match="Unknown sample type"):
            parse_ftp_event("f,type=snp,rate=100,dur=1.0,npts=100,pref=rate,smpl=q;null;null")

    def test_ftp_missing_rate(self):
        with pytest.raises(ValueError, match="missing required 'rate'"):
            parse_ftp_event("f,type=ftp,dur=1.0;null;null")

    def test_ftp_missing_dur(self):
        with pytest.raises(ValueError, match="missing required 'dur'"):
            parse_ftp_event("f,type=ftp,rate=60.0;null;null")

    def test_snapshot_missing_fields(self):
        with pytest.raises(ValueError, match="missing required fields"):
            parse_ftp_event("f,type=snp,rate=100,dur=1.0;null;null")

    def test_rearm_invalid_boolean(self):
        with pytest.raises(ValueError, match="Invalid rearm enabled"):
            parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;null;rearm=maybe")

    def test_clock_trigger_too_many_events(self):
        with pytest.raises(ValueError, match="max 8 events"):
            parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=e,01,02,03,04,05,06,07,08,09,0;null")

    def test_clock_sample_too_many_events(self):
        with pytest.raises(ValueError, match="max 4 events"):
            parse_ftp_event("f,type=snp,rate=100,dur=1.0,npts=100,pref=rate,smpl=e,01,02,03,04,05;null;null")

    def test_external_trigger_modifier_out_of_range(self):
        with pytest.raises(ValueError, match="modifier must be 0-3"):
            parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=x,5;null")

    def test_external_sample_modifier_out_of_range(self):
        with pytest.raises(ValueError, match="modifier must be 0-3"):
            parse_ftp_event("f,type=snp,rate=100,dur=1.0,npts=100,pref=rate,smpl=x,mod=7;null;null")

    def test_device_trigger_unknown_field(self):
        with pytest.raises(ValueError, match="Unknown device trigger field"):
            parse_ftp_event("f,type=ftp,rate=60.0,dur=1.0;trig=d,M:OUTTMP,foo=1;null")
