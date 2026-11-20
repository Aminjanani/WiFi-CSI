"""Classic ESP32 serial protocol and v2 features for a 20 MHz LLTF link."""

from collections import deque
import csv
from dataclasses import dataclass
from math import hypot
from statistics import mean

import numpy as np


PACKET_FIELDS = ("sequence", "rx_timestamp_us", "rssi", "noise_floor", "channel",
                 "sig_mode", "rate", "mcs", "cwb", "stbc", "sig_len",
                 "len", "first_word_invalid", "csi_bytes", "mean_magnitude")
# Frame-format fields (sig_mode..sig_len) are blank for logs from firmware before they existed.
FORMAT_FIELDS = ("sig_mode", "rate", "mcs", "cwb", "stbc", "sig_len")
SCHEMA = 3
# amp_scale is the only feature left on the raw, gain-affected scale. Every other
# amplitude feature uses packets divided by their own mean, which cancels AGC steps.
FEATURES = ("rssi_mean", "rssi_sd", "amp_scale", "shape_spread", "carrier_sd",
            "carrier_step", "pc1_share", "breath_power", "motion_power", "state_separation",
            "breath_snr", "breath_rate", "breath_peak", "breath_stable")
# Raw received power and gain drift by 5+ dB in an empty room within an hour, and move
# when people change seats, so the models leave them out (kept in the CSV for audits).
DRIFTING = ("rssi_mean", "amp_scale")
MODEL_FEATURES = tuple(f for f in FEATURES if f not in DRIFTING)
# Default model inputs: what a living body does (movement, breathing rhythm). Static
# shape features also fingerprint chairs, door and blankets; over rounds r01+r02 they
# cost presence accuracy (12/12 recordings with these five, 10/12 with all non-drifting).
DYNAMIC_FEATURES = ("carrier_step", "motion_power", "breath_snr", "breath_stable", "breath_peak")
BREATH_HZ = (0.15, 0.6)  # 9-36 breaths/min searched; resting adults sit near 10-16
CALM_BPM = (8, 20)       # a steady rhythm in this range counts as breathing
STABLE_US = 60_000_000   # breath_stable looks back over this much device time
SHAPE_STATES = 4         # distinct per-frame spectrum shapes seen from the D-Link
FLOOR = -12.0           # log10 band power reported for a perfectly flat signal
MIN_SEPARATION = 0.5     # below this share of variance, treat the stream as one shape (walking: ~0.76, still: ~0.997)
WINDOW_US = 30_000_000   # 30 s: 5-8 breaths, 0.033 Hz resolution for the breathing rate
HOP_US = 2_000_000       # a new estimate every 2 s
MAX_GAP_US = 400_000
GRID_HZ = 20             # uniform resampling rate for spectral features
# Bin order 0..31,-32..-1. Common occupied bins, omitting +1 so the mask stays
# identical when the first hardware word is invalid.
BINS = (*range(2, 27), *range(38, 64))


@dataclass
class Packet:
    sequence: int
    timestamp: int
    rssi: int
    noise: int
    channel: int
    invalid: int
    iq: list[int]
    sig_mode: int | None = None
    rate: int | None = None
    mcs: int | None = None
    cwb: int | None = None
    stbc: int | None = None
    sig_len: int | None = None

    def magnitudes(self):
        start = 4 if self.invalid else 0
        return [hypot(self.iq[i], self.iq[i + 1]) for i in range(start, len(self.iq), 2)]

    def csv_row(self):
        fmt = ("" if v is None else v for v in (self.sig_mode, self.rate, self.mcs, self.cwb,
                                                  self.stbc, self.sig_len))
        return (self.sequence, self.timestamp, self.rssi, self.noise, self.channel, *fmt,
                len(self.iq), self.invalid, ";".join(map(str, self.iq)),
                round(mean(self.magnitudes()), 5))


def parse(line):
    """Return None for diagnostics including the firmware's boot-time CSV legend."""
    if not line.startswith("CSI,") or line.startswith("CSI,sequence,"):
        return None
    fields = next(csv.reader([line], strict=True))
    fmt = {}
    if len(fields) == 15:    # firmware with frame-format fields
        fmt = dict(zip(FORMAT_FIELDS, map(int, fields[6:12])))
        fields = fields[:6] + fields[12:]
    if len(fields) != 9:
        raise ValueError("expected 9 or 15 wire fields")
    seq, ts, rssi, noise, channel, length, invalid = map(int, fields[1:8])
    iq = [int(x) for x in fields[8].split(";")]
    if not (0 <= seq < 2**32 and 0 <= ts < 2**32 and 1 <= channel <= 14
            and -128 <= rssi <= 127 and -128 <= noise <= 127 and invalid in (0, 1)):
        raise ValueError("invalid header")
    if not (4 * invalid < length <= 384 and length == len(iq) and length % 2 == 0
            and all(-128 <= x <= 127 for x in iq)):
        raise ValueError("invalid signed IQ payload")
    return Packet(seq, ts, rssi, noise, channel, invalid, iq, **fmt)


def shape_states(norm, k=SHAPE_STATES, restarts=4, iterations=20):
    """Cluster gain-normalized packets into the transmitter's shape states.

    SHAPE_STATES is a property of the router, measured once in an empty room
    (the D-Link uses 4). Set it to 1 for a router whose frames share one shape:
    clustering a stream that has no discrete states would erase slow real changes.
    k-means++ uses a fixed seed, so replay and live prediction give the same labels.
    Returns per-packet labels and the share of variance explained by the states.
    """
    n = len(norm)
    if k <= 1 or n < 4 * k:
        return np.zeros(n, dtype=int), 0.0
    total = ((norm - norm.mean(axis=0)) ** 2).sum()
    if total <= 1e-12:
        return np.zeros(n, dtype=int), 0.0
    rng = np.random.default_rng(0)
    best = None
    for _ in range(restarts):
        centers = [norm[rng.integers(n)]]
        for _ in range(1, k):
            distance = np.min([((norm - c) ** 2).sum(axis=1) for c in centers], axis=0)
            centers.append(norm[rng.choice(n, p=distance / distance.sum())] if distance.sum() > 0
                           else norm[rng.integers(n)])
        centers = np.array(centers)
        for _ in range(iterations):
            labels = np.argmin(((norm[:, None, :] - centers[None]) ** 2).sum(axis=2), axis=1)
            centers = np.array([norm[labels == c].mean(axis=0) if (labels == c).any() else centers[c]
                                for c in range(k)])
        within = ((norm - centers[labels]) ** 2).sum()
        if best is None or within < best[0]:
            best = (within, labels)
    separation = float(1 - best[0] / total)
    if separation < MIN_SEPARATION:      # no clear states in this window: leave it unaligned
        return np.zeros(n, dtype=int), separation
    return best[1], separation


def window_features(packets):
    """Features for one window of classic ESP32 LLTF packets (20 MHz, no secondary channel)."""
    t = np.array([p.timestamp for p in packets], dtype=float) / 1e6
    amp = np.array([[hypot(p.iq[2*i], p.iq[2*i+1]) for i in BINS] for p in packets])
    rssi = np.array([p.rssi for p in packets], dtype=float)
    scale = amp.mean(axis=1)
    norm = amp / np.maximum(scale, 1e-9)[:, None]
    # The D-Link sends each frame in one of 4 fixed spectrum shapes, chosen at random
    # (measured in an empty room: 25 % each, 2.6 % spread within a shape, ~50 % between).
    # Divide every packet by its own shape's mean so only changes caused by the room remain.
    labels, separation = shape_states(norm)
    aligned = norm.copy()
    for state in np.unique(labels):
        aligned[labels == state] /= norm[labels == state].mean(axis=0)
    # Resample to a uniform grid so step size and spectra do not depend on packet rate.
    # Packets are averaged within each grid bin (anti-aliasing at ~100 packets/s);
    # empty bins are interpolated from their neighbours (sparser, older 10 packets/s logs).
    slot = np.floor((t - t[0]) * GRID_HZ).astype(int)
    bins = slot[-1] + 1
    counts = np.bincount(slot, minlength=bins)
    filled = counts > 0
    centers = t[0] + (np.arange(bins) + 0.5) / GRID_HZ
    grid = centers
    series = np.empty((bins, aligned.shape[1]))
    for j in range(aligned.shape[1]):
        sums = np.bincount(slot, weights=aligned[:, j], minlength=bins)
        series[:, j] = np.interp(centers, centers[filled], sums[filled] / counts[filled])
    series -= series.mean(axis=0)
    _, singular, vt = np.linalg.svd(series, full_matrices=False)
    variance = singular ** 2
    total_var = variance.sum()
    pc1_share = float(variance[0] / total_var) if total_var > 1e-12 else 0.0
    # Absolute band power (log10 variance) of the strongest common pattern. A share of
    # total power would look like "breathing" even in an empty room, where it is all noise.
    breath = motion = FLOOR
    if total_var > 1e-12 and len(grid) >= 8:
        signal = series @ vt[0]
        taper = np.hanning(len(signal))
        spectrum = np.abs(np.fft.rfft(signal * taper)) ** 2 * 2 / (len(signal) * (taper ** 2).sum())
        freq = np.fft.rfftfreq(len(signal), 1 / GRID_HZ)
        band = lambda lo, hi: spectrum[(freq >= lo) & (freq < hi)].sum()
        breath = max(FLOOR, float(np.log10(band(0.1, 0.5) + 1e-12)))
        motion = max(FLOOR, float(np.log10(band(0.5, 3.0) + 1e-12)))
    snr, rate, peak = breathing(series)
    return dict(zip(FEATURES, (
        float(rssi.mean()), float(rssi.std()), float(scale.mean()),
        float(norm.std(axis=1).mean()), float(np.median(aligned.std(axis=0))),
        float(np.abs(np.diff(series, axis=0)).mean()) if len(grid) > 1 else 0.0,
        pc1_share, float(breath), float(motion), separation, snr, rate, peak,
        0.0,  # breath_stable is filled in by Windows from the recent windows
    )))


def breathing(series, best=5):
    """Breathing evidence in one window of aligned subcarriers sampled at GRID_HZ.

    Each subcarrier's strongest peak in BREATH_HZ is compared with its own noise
    floor at 1-4 Hz. The best subcarriers are those passing nearest a breathing chest.
    Returns (log10 peak-to-floor ratio, rate in breaths/min, share of band power at the peak).
    In an empty room the "peak" is noise, so its rate wanders from window to window.
    """
    if len(series) < 8 * GRID_HZ:
        return 0.0, 0.0, 0.0
    taper = np.hanning(len(series))
    power = np.abs(np.fft.rfft(series * taper[:, None], axis=0)) ** 2
    freq = np.fft.rfftfreq(len(series), 1 / GRID_HZ)
    band = (freq >= BREATH_HZ[0]) & (freq <= BREATH_HZ[1])
    floor = np.median(power[(freq > 1.0) & (freq < 4.0)], axis=0)
    if not band.any() or not (floor > 0).all():
        return 0.0, 0.0, 0.0
    ratio = power[band].max(axis=0) / floor
    chosen = np.argsort(ratio)[-best:]
    combined = power[band][:, chosen].sum(axis=1)
    top = int(np.argmax(combined))
    peak = combined[max(0, top - 1):top + 2].sum() / combined.sum()
    return (float(np.log10(np.median(ratio[chosen]))), float(freq[band][top] * 60), float(peak))


class Windows:
    """Sliding device-time windows shared by dataset replay and live prediction.

    A window of window_us is evaluated every hop_us. It is kept when it averages
    at least 5 packets/s and covers 75 % of its length. Gaps over 400 ms, backward
    time or sequence, or a channel change start a new segment.
    """

    def __init__(self, window_us=WINDOW_US, hop_us=HOP_US):
        self.window_us, self.hop_us = window_us, hop_us
        self.samples = deque()
        self.next_end = None
        self.last = None
        self.segment = 0
        self.rejected = 0
        self.rates = deque()     # (end_us, breaths/min) of recent windows in this segment

    def reset(self):
        self.samples.clear()
        self.rates.clear()
        self.next_end = self.last = None
        self.segment += 1

    def finish(self):
        """Drop the incomplete tail. Kept for callers that flush at end of stream."""
        self.reset()
        return None

    def _emit(self, end):
        samples = [q for q in self.samples if q.timestamp >= end - self.window_us]
        if len(samples) < 2:
            self.rejected += 1
            return None
        span = samples[-1].timestamp - samples[0].timestamp
        if len(samples) < 5 * self.window_us / 1e6 or span < 0.75 * self.window_us:
            self.rejected += 1
            return None
        row = dict(segment=self.segment, start_us=samples[0].timestamp,
                   end_us=samples[-1].timestamp, packets=len(samples),
                   channel=samples[0].channel, **window_features(samples))
        # A breathing person keeps one calm rate from window to window; noise wanders.
        # Share of the last minute's windows within CALM_BPM and 3 breaths/min of their median.
        self.rates.append((end, row["breath_rate"]))
        while self.rates and self.rates[0][0] < end - STABLE_US:
            self.rates.popleft()
        recent = np.array([r for _, r in self.rates])
        centre = np.median(recent)
        calm = (recent >= CALM_BPM[0]) & (recent <= CALM_BPM[1]) & (np.abs(recent - centre) <= 3)
        row["breath_stable"] = float(calm.mean())
        return row

    def push(self, p):
        if len(p.iq) != 128:
            self.reset()
            return None
        prev = self.last
        if prev and (p.sequence <= prev.sequence or p.timestamp <= prev.timestamp
                     or p.channel != prev.channel or p.timestamp - prev.timestamp > MAX_GAP_US):
            self.reset()
        if self.next_end is None:
            self.next_end = p.timestamp + self.window_us
        result = None
        if p.timestamp >= self.next_end:
            result = self._emit(self.next_end)
            while self.next_end <= p.timestamp:
                self.next_end += self.hop_us
        self.samples.append(p)
        self.last = p
        while self.samples and self.samples[0].timestamp < self.next_end - self.window_us:
            self.samples.popleft()
        return result


def breaks_stream(line):
    return line.startswith(("Wi-Fi lost.", "CSI capture STOPPED.", "ESP32 Router-based",
                            "Connected AP BSSID:", "SCAN BEGIN"))
