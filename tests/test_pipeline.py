"""Regression checks using generated wire records. All recordings here are synthetic."""

import csv
import io
import json
from math import pi, sin
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from src.build_dataset import FIELDS, add_session
from src.data_collection.csi import BINS, Packet, Windows, parse, window_features
from src.data_collection.serial_capture import capture


def line(seq, timestamp, level=3):
    payload = ";".join(map(str, [127, 127, 127, 127] + [level, 4] * 62))
    return f'CSI,{seq},{timestamp},-58,-95,6,128,1,"{payload}"\r\n'


class Wire:
    def __init__(self, chunks):
        self.chunks = iter(chunks)

    def readline(self):
        try:
            return next(self.chunks)
        except StopIteration:
            raise KeyboardInterrupt


class PipelineTest(unittest.TestCase):
    def test_header_and_invalid_word(self):
        self.assertIsNone(parse('CSI,sequence,rx_timestamp_us,rssi,noise_floor,channel,len,first_word_invalid,"csi_bytes"'))
        self.assertEqual(parse(line(1, 0)).magnitudes(), [5] * 62)
        for text in [line(1, 0).replace(",128,", ",127,"), line(1, 0).replace("127;", "200;", 1)]:
            with self.assertRaises(ValueError):
                parse(text)

    def test_capture_replay(self):
        records = [line(i+1, i*50_000).encode() for i in range(800)]
        chunks = [b"boot\r\n", b"CSI,wrong\n", records[0][:17], records[0][17:], *records[1:]]
        args = SimpleNamespace(room="test", block="synthetic", node="rx", count=1,
                               activity="fixture", notes="", seconds=0, settle=0)
        output = io.StringIO()
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            capture(Wire(chunks), folder, args)
            self.assertEqual((folder / "serial.log").read_bytes(), b"".join(chunks))
            meta = json.loads((folder / "session.json").read_text())
            self.assertEqual((meta["first_line"], meta["last_line"], meta["malformed"]), (3, 802, 1))
            with (folder / "csi_packets.csv").open() as packet_file:
                rows = list(csv.DictReader(packet_file))
                self.assertEqual(len(rows), 800)
                self.assertEqual(rows[0]["mean_magnitude"], "5.0")
                self.assertNotIn(None, rows[0])
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            # 40 s at 20 packets/s: 30 s windows ending at 30, 32, 34, 36 and 38 s.
            self.assertEqual(add_session(folder, writer), 5)
        output.seek(0)
        windows = list(csv.DictReader(output))
        self.assertTrue(all(float(row["amp_scale"]) == 5 and float(row["carrier_sd"]) == 0 for row in windows))

    def test_gaps_restarts_and_wrap(self):
        for seq, ts in [(31, 3_400_000), (1, 3_100_000), (31, 0)]:
            windows = Windows(window_us=2_000_000, hop_us=1_000_000)
            emitted = [windows.push(parse(line(i+1, i*100_000))) for i in range(30)]
            self.assertEqual(sum(row is not None for row in emitted), 1)
            self.assertIsNone(windows.push(parse(line(seq, ts))))
            self.assertEqual((windows.segment, len(windows.samples)), (1, 1))


def packet(i, t_s, values):
    iq = [0] * 128
    for k, b in enumerate(BINS):
        iq[2*b + 1] = max(-128, min(127, round(values[k])))
    return Packet(i, round(t_s * 1e6), -40, -95, 13, 0, iq)


class FeatureTest(unittest.TestCase):
    def test_gain_steps_cancel(self):
        shape = [20 + (k % 7) for k in range(len(BINS))]
        packets = [packet(i, i / 50, [v * (1 if i % 3 else 2) for v in shape]) for i in range(500)]
        row = window_features(packets)
        self.assertLess(row["carrier_sd"], 1e-6)
        self.assertGreater(row["amp_scale"], 1.2 * mean_of(shape))

    def test_breathing_band(self):
        # Four transmitter shape states chosen at random per frame, like the D-Link,
        # with a 0.3 Hz breathing-like modulation on half of the subcarriers.
        n = len(BINS)
        states = [[30 + 12 * sin(2 * pi * (k + 5 * s) / n) for k in range(n)] for s in range(4)]
        packets = []
        for i in range(500):
            shape = states[(i * 7 + i // 3) % 4]
            breath = 1 + 0.05 * sin(2 * pi * 0.3 * i / 50)
            packets.append(packet(i, i / 50, [v * (breath if k % 2 else 1) for k, v in enumerate(shape)]))
        row = window_features(packets)
        self.assertGreater(row["state_separation"], 0.9)
        self.assertGreater(row["breath_power"], row["motion_power"] + 1)

    def test_breathing_rhythm(self):
        # 30 s of 4 random shape states, mild noise, and 12 breaths/min on some subcarriers.
        n, rng = len(BINS), __import__("numpy").random.default_rng(1)
        states = [[30 + 12 * sin(2 * pi * (k + 5 * s) / n) for k in range(n)] for s in range(4)]
        def window(breathing):
            packets = []
            for i in range(3000):
                breath = 1 + (0.03 * sin(2 * pi * 0.2 * i / 100) if breathing else 0)
                shape = states[int(rng.integers(4))]
                values = [v * (breath if k % 3 == 0 else 1) * (1 + rng.normal(0, 0.01)) for k, v in enumerate(shape)]
                packets.append(packet(i, i / 100, values))
            return window_features(packets)
        person, empty = window(True), window(False)
        self.assertAlmostEqual(person["breath_rate"], 12, delta=2)
        self.assertGreater(person["breath_snr"], empty["breath_snr"] + 0.5)
        self.assertGreater(person["breath_peak"], empty["breath_peak"])

def mean_of(values):
    return sum(values) / len(values)

class Board(Wire):
    def __init__(self, chunks):
        super().__init__(chunks)
        self.sent = []

    def write(self, data):
        self.sent.append(data)


class SourceTest(unittest.TestCase):
    def run_capture(self, chunks, expect=None):
        args = SimpleNamespace(room="test", block="synthetic", node="rx", count=0, activity="fixture",
                               notes="", seconds=0, settle=0, verify_source=True, expect_bssid=expect)
        with TemporaryDirectory() as temp:
            board = Board(chunks)
            capture(board, Path(temp), args)
            return board, json.loads((Path(temp) / "session.json").read_text())

    def test_records_source(self):
        records = [line(i+1, i*100_000).encode() for i in range(5)]
        scan = [b"SCAN BEGIN\n", b"AP 02:00:00:00:00:01 channel=6 rssi=-30 source=1 ssid=D-Link\n",
                b"AP 02:00:00:00:00:02 channel=10 rssi=-33 source=0 ssid=Other\n", b"SCAN END ESP_OK count=2\n"]
        board, meta = self.run_capture([records[0], *scan, *records[1:]])
        self.assertEqual(board.sent, [b"SCAN\n"])
        self.assertEqual((meta["source_bssid"], meta["source_ssid"], meta["source_channel"]),
                         ("02:00:00:00:00:01", "D-Link", 6))
        self.assertEqual((meta["valid_packets"], meta["first_line"], meta["count"]), (4, 6, 0))

    def test_silent_board_aborts(self):
        args = SimpleNamespace(room="test", block="r01", node="rx", count=0, activity="fixture", notes="",
                               seconds=120, settle=0, wait=0, verify_source=False, expect_bssid=None)
        with TemporaryDirectory() as temp:
            capture(Wire([b""] * 5), Path(temp), args)
            meta = json.loads((Path(temp) / "session.json").read_text())
        self.assertEqual((meta["count"], meta["valid_packets"], meta["completed"]), (None, 0, False))

    def test_wrong_source_aborts(self):
        scan = [b"AP 02:00:00:00:00:03 channel=6 rssi=-36 source=1 ssid=Redmi\n", b"SCAN END ESP_OK count=1\n"]
        _, meta = self.run_capture([*scan, line(1, 0).encode()], expect="02:00:00:00:00:01")
        self.assertIsNone(meta["count"])
        self.assertEqual(meta["valid_packets"], 0)


class CameraTest(unittest.TestCase):
    @staticmethod
    def body(shoulder, hip, knee, ankle):
        xy = [[0, 0]] * 17
        conf = [0.0] * 17
        for (a, b), point in zip(((5, 6), (11, 12), (13, 14), (15, 16)), (shoulder, hip, knee, ankle)):
            if point is not None:
                xy[a] = xy[b] = list(point)
                conf[a] = conf[b] = 0.9
        return xy, conf

    def test_activity_rules(self):
        from collections import deque
        from src.data_collection.camera_labels import classify
        standing = self.body((100, 100), (100, 200), (100, 290), (100, 380))
        sitting = self.body((100, 100), (100, 200), (180, 205), (180, 290))
        lying = self.body((100, 300), (220, 310), (320, 315), (420, 320))
        legs_hidden = self.body((100, 100), (100, 200), None, None)
        self.assertEqual(classify(*standing, 300, deque(), 0)[0], "standing")
        self.assertEqual(classify(*sitting, 250, deque(), 0)[0], "sitting")
        self.assertEqual(classify(*lying, 100, deque(), 0)[0], "lying")
        self.assertEqual(classify(*legs_hidden, 300, deque(), 0)[0], "upright")
        self.assertEqual(classify(*self.body(None, None, None, None), 300, deque(), 0)[0], "unknown")
        history = deque()
        for i in range(12):        # hips move 30 px per 0.1 s = 1 body height/s for a 300 px body
            label = classify(*self.body((100 + 30 * i, 100), (100 + 30 * i, 200), (100 + 30 * i, 290),
                                        (100 + 30 * i, 380)), 300, history, i / 10)[0]
        self.assertEqual(label, "walking")

    def test_camera_windows_align_by_host_time(self):
        from src.build_dataset import camera_fields
        offset = 1_000_000.0       # laptop clock = ESP32 seconds + offset
        times = [offset + t / 10 for t in range(400)]
        rows = [(t, 1, "sitting" if i < 300 else "walking", False) for i, t in enumerate(times)]
        cam = camera_fields((times, rows, offset), dict(start_us=0, end_us=29_900_000))
        self.assertEqual((cam["cam_persons"], cam["cam_activity"], cam["cam_frames"]), (1, "sitting", 300))
        cam = camera_fields((times, rows, offset), dict(start_us=25_000_000, end_us=39_900_000))
        self.assertEqual(cam["cam_activity"], "walking")
        self.assertAlmostEqual(cam["cam_agree"], 100 / 150, places=2)


if __name__ == "__main__":
    unittest.main()
