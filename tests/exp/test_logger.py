"""Tests for DataLogger."""

import csv
import time

import pytest

from pacsys.exp._logger import DataLogger
from pacsys.exp._writers import CsvWriter
from pacsys.testing import FakeBackend


@pytest.fixture
def fake():
    return FakeBackend()


class TestDataLogger:
    def test_logs_readings_to_csv(self, fake, tmp_path):
        path = tmp_path / "log.csv"
        with DataLogger(
            ["M:OUTTMP@p,1000"],
            writer=CsvWriter(path),
            flush_interval=0.05,
            backend=fake,
        ):
            fake.emit_reading("M:OUTTMP@p,1000", 72.5)
            fake.emit_reading("M:OUTTMP@p,1000", 73.0)
            time.sleep(0.15)  # wait for flush

        rows = list(csv.reader(open(path)))
        assert len(rows) >= 3  # header + 2 readings

    def test_context_manager_stops(self, fake, tmp_path):
        path = tmp_path / "log.csv"
        dl = DataLogger(
            ["M:OUTTMP@p,1000"],
            writer=CsvWriter(path),
            backend=fake,
        )
        with dl:
            pass
        assert not dl.running

    def test_empty_devices_raises(self, fake, tmp_path):
        path = tmp_path / "log.csv"
        with pytest.raises(ValueError, match="devices cannot be empty"):
            DataLogger([], writer=CsvWriter(path), backend=fake)

    def test_double_start_raises(self, fake, tmp_path):
        path = tmp_path / "log.csv"
        dl = DataLogger(
            ["M:OUTTMP@p,1000"],
            writer=CsvWriter(path),
            backend=fake,
        )
        dl.start()
        with pytest.raises(RuntimeError, match="already running"):
            dl.start()
        dl.stop()

    def test_final_flush_on_stop(self, fake, tmp_path):
        """Readings buffered at stop time are flushed."""
        path = tmp_path / "log.csv"
        dl = DataLogger(
            ["M:OUTTMP@p,1000"],
            writer=CsvWriter(path),
            flush_interval=999,  # won't auto-flush
            backend=fake,
        )
        dl.start()
        fake.emit_reading("M:OUTTMP@p,1000", 72.5)
        time.sleep(0.05)
        dl.stop()

        rows = list(csv.reader(open(path)))
        assert len(rows) == 2  # header + 1 reading
