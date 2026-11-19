"""Regression checks using generated wire records. All recordings here are synthetic."""

import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from src.build_dataset import FIELDS, add_session
from src.data_collection.csi import Windows, parse
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
        records = [line(i+1, i*100_000).encode() for i in range(40)]
        chunks = [b"boot\r\n", b"CSI,wrong\n", records[0][:17], records[0][17:], *records[1:]]
        args = SimpleNamespace(room="test", block="synthetic", node="rx", count=1,
                               activity="fixture", notes="", seconds=0, settle=0)
        output = io.StringIO()
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            capture(Wire(chunks), folder, args)
            self.assertEqual((folder / "serial.log").read_bytes(), b"".join(chunks))
            meta = json.loads((folder / "session.json").read_text())
            self.assertEqual((meta["first_line"], meta["last_line"], meta["malformed"]), (3, 42, 1))
            with (folder / "csi_packets.csv").open() as packet_file:
                rows = list(csv.DictReader(packet_file))
                self.assertEqual(len(rows), 40)
                self.assertEqual(rows[0]["mean_magnitude"], "5.0")
                self.assertNotIn(None, rows[0])
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            self.assertEqual(add_session(folder, writer), 2)
        output.seek(0)
        windows = list(csv.DictReader(output))
        self.assertTrue(all(float(row["amp_mean"]) == 5 and float(row["carrier_sd"]) == 0 for row in windows))

    def test_gaps_restarts_and_wrap(self):
        for seq, ts in [(11, 2_000_000), (1, 1_000_000), (11, 0)]:
            windows = Windows()
            for i in range(10):
                windows.push(parse(line(i+1, i*100_000)))
            self.assertIsNone(windows.push(parse(line(seq, ts))))
            self.assertIsNone(windows.finish())
            self.assertEqual(windows.rejected, 2)


if __name__ == "__main__":
    unittest.main()
