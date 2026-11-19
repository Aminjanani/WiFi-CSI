"""Classic ESP32 serial protocol and v1 features for a 20 MHz LLTF link."""

import csv
from dataclasses import dataclass
from statistics import mean, median, pstdev
from math import hypot


PACKET_FIELDS = ("sequence", "rx_timestamp_us", "rssi", "noise_floor", "channel",
                 "len", "first_word_invalid", "csi_bytes", "mean_magnitude")
FEATURES = ("rssi_mean", "rssi_sd", "amp_mean", "amp_spread", "carrier_sd",
            "carrier_step")
WINDOW_US = 2_000_000


@dataclass
class Packet:
    sequence: int
    timestamp: int
    rssi: int
    noise: int
    channel: int
    invalid: int
    iq: list[int]

    def magnitudes(self):
        start = 4 if self.invalid else 0
        return [hypot(self.iq[i], self.iq[i + 1]) for i in range(start, len(self.iq), 2)]

    def csv_row(self):
        return (self.sequence, self.timestamp, self.rssi, self.noise, self.channel,
                len(self.iq), self.invalid, ";".join(map(str, self.iq)),
                round(mean(self.magnitudes()), 5))


def parse(line):
    """Return None for diagnostics including the firmware's boot-time CSV legend."""
    if not line.startswith("CSI,") or line.startswith("CSI,sequence,"):
        return None
    fields = next(csv.reader([line], strict=True))
    if len(fields) != 9:
        raise ValueError("expected 9 wire fields")
    seq, ts, rssi, noise, channel, length, invalid = map(int, fields[1:8])
    iq = [int(x) for x in fields[8].split(";")]
    if not (0 <= seq < 2**32 and 0 <= ts < 2**32 and 1 <= channel <= 14
            and -128 <= rssi <= 127 and -128 <= noise <= 127 and invalid in (0, 1)):
        raise ValueError("invalid header")
    if not (4 * invalid < length <= 384 and length == len(iq) and length % 2 == 0
            and all(-128 <= x <= 127 for x in iq)):
        raise ValueError("invalid signed IQ payload")
    return Packet(seq, ts, rssi, noise, channel, invalid, iq)


def window_features(packets):
    # Pilot accepts only classic ESP32 LLTF, 20 MHz, with no secondary channel.
    # Bin order 0..31,-32..-1. Use common occupied bins and omit +1 so the
    # mask stays identical even when the first hardware word was invalid.
    bins = (*range(2, 27), *range(38, 64))
    rows = [[hypot(p.iq[2*i], p.iq[2*i+1]) for i in bins] for p in packets]
    columns = list(zip(*rows))
    return dict(zip(FEATURES, (
        mean(p.rssi for p in packets), pstdev(p.rssi for p in packets),
        mean(map(mean, rows)), mean(map(pstdev, rows)), median(map(pstdev, columns)),
        mean(mean(abs(b-a) for a, b in zip(prev, curr)) for prev, curr in zip(rows, rows[1:])),
    )))


class Windows:
    """Non-overlapping device-time buckets shared by replay and live prediction."""

    def __init__(self):
        self.samples = []
        self.origin = None
        self.last = None
        self.segment = 0
        self.rejected = 0

    def reset(self):
        self.rejected += int(bool(self.samples))
        self.samples.clear()
        self.origin = self.last = None
        self.segment += 1

    def finish(self):
        samples, self.samples = self.samples, []
        if not samples:
            return None
        span = samples[-1].timestamp - samples[0].timestamp
        if len(samples) < 10 or span < 1_500_000:
            self.rejected += 1
            return None
        return dict(segment=self.segment, start_us=samples[0].timestamp,
                    end_us=samples[-1].timestamp, packets=len(samples),
                    channel=samples[0].channel, **window_features(samples))

    def push(self, p):
        if len(p.iq) != 128:
            self.reset()
            return None
        prev = self.last
        if prev and (p.sequence <= prev.sequence or p.timestamp <= prev.timestamp
                     or p.channel != prev.channel or p.timestamp - prev.timestamp > 400_000):
            self.reset()
        if self.origin is None:
            self.origin = p.timestamp
        result = None
        if self.last and (p.timestamp-self.origin)//WINDOW_US != (self.last.timestamp-self.origin)//WINDOW_US:
            result = self.finish()
        self.samples.append(p)
        self.last = p
        return result


def breaks_stream(line):
    return line.startswith(("Wi-Fi lost.", "CSI capture STOPPED.", "ESP32 Router-based",
                            "Connected AP BSSID:", "SCAN BEGIN"))
