"""Replay labeled serial logs into the window-level CSV, with webcam labels when recorded."""

import argparse
from bisect import bisect_left, bisect_right
from collections import Counter
import csv
from datetime import datetime
import json
from statistics import median
from pathlib import Path
from tempfile import NamedTemporaryFile

from .data_collection.csi import FEATURES, Windows, breaks_stream, parse


CONTEXT = ("session", "block", "room", "node", "source_bssid", "activity", "count")
# Webcam labels per window: frames seen, most common person count and scene
# ("empty", "sitting", "walking", "sitting+walking"...), the share of frames that agree
# with that scene, and whether a fall was flagged. Blank for sessions without a camera.
CAMERA_FIELDS = ("label_source", "cam_frames", "cam_persons", "cam_activity", "cam_agree", "cam_fall")
FIELDS = (*CONTEXT, "segment", "start_us", "end_us", "packets", "channel", *FEATURES, *CAMERA_FIELDS)


def host_seconds(text):
    return datetime.fromisoformat(text).timestamp()


def camera_track(directory):
    """Camera rows on the laptop clock plus the offset that maps ESP32 time onto it."""
    path = directory / "camera_labels.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf8", newline="") as handle:
        rows = [(host_seconds(r["host_utc"]), int(r["persons"]), r["scene"], r["fall"] == "1")
                for r in csv.DictReader(handle)]
    with (directory / "csi_packets.csv").open(encoding="utf8", newline="") as handle:
        offsets = [host_seconds(r["host_received_utc"]) - int(r["rx_timestamp_us"]) / 1e6
                   for r in csv.DictReader(handle)]
    if not rows or not offsets:
        return None
    rows.sort()
    return [r[0] for r in rows], rows, median(offsets)


def camera_fields(track, row):
    if track is None:
        return {}
    times, rows, offset = track
    lo = bisect_left(times, row["start_us"] / 1e6 + offset)
    hi = bisect_right(times, row["end_us"] / 1e6 + offset)
    seen = rows[lo:hi]
    if not seen:
        return dict(cam_frames=0)
    scene, votes = Counter(r[2] for r in seen).most_common(1)[0]
    return dict(cam_frames=len(seen), cam_persons=Counter(r[1] for r in seen).most_common(1)[0][0],
                cam_activity=scene, cam_agree=round(votes / len(seen), 3),
                cam_fall=int(any(r[3] for r in seen)))


def add_session(directory, writer, source=None):
    meta = json.loads((directory / "session.json").read_text(encoding="utf8"))
    track = camera_track(directory)
    camera_labeled = meta.get("label_source") == "camera" and track is not None
    if (meta.get("count") is None and not camera_labeled) or meta.get("first_line") is None:
        print(f"Skipping {directory.name}: unlabeled or empty measurement.")
        return 0
    if meta["schema"] != 1 or (meta.get("count") is not None and
                               (type(meta["count"]) is not int or meta["count"] not in (0, 1, 2))):
        raise ValueError(f"Unsupported label or schema in {directory}")
    if meta["session"] != directory.name or not all(meta[k] for k in ("block", "room", "node")):
        raise ValueError(f"Missing or inconsistent session identity in {directory}")
    if source and (meta.get("source_bssid") or "").lower() != source.lower():
        print(f"Skipping {directory.name}: CSI source {meta.get('source_bssid')} is not {source}.")
        return 0
    context = {key: meta.get(key) for key in CONTEXT}
    context["label_source"] = "camera" if camera_labeled else "hand"
    windows = Windows()
    written = bad = unsupported = 0
    with (directory / "serial.log").open(encoding="utf8", errors="replace", newline="\n") as stream:
        for line_no, line in enumerate(stream, 1):
            if line_no < meta["first_line"]:
                continue
            if line_no > meta["last_line"]:
                break
            try:
                packet = parse(line.rstrip("\r\n"))
            except (ValueError, csv.Error):
                bad += 1
                windows.reset()
                continue
            if packet is None:
                if breaks_stream(line):
                    windows.reset()
                continue
            unsupported += int(len(packet.iq) != 128)
            row = windows.push(packet)
            if row:
                cam = camera_fields(track, row)
                if camera_labeled:
                    if not cam.get("cam_frames"):
                        continue            # no camera frames: no label for this window
                    cam["count"] = cam["cam_persons"]
                writer.writerow({**context, **row, **cam})
                written += 1
    windows.finish()
    print(f"{directory.name}: {written} windows, {windows.rejected} rejected buckets, "
          f"{bad} malformed and {unsupported} unsupported packets.")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/windows.csv"))
    parser.add_argument("--source", help="Only include sessions recorded from this AP BSSID")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    handle = NamedTemporaryFile(dir=args.out.parent, delete=False, mode="w", encoding="utf8",
                                newline="")
    temp = Path(handle.name)
    try:
        with handle as output:
            writer = csv.DictWriter(output, fieldnames=FIELDS)
            writer.writeheader()
            # Sessions may sit in subfolders such as data/raw/dlink/session_*.
            for directory in sorted({p.parent for p in args.raw.rglob("serial.log")}):
                if (directory / "session.json").exists():
                    count += add_session(directory, writer, args.source)
                else:
                    print(f"Skipping {directory.name}: missing session.json.")
        if not count:
            parser.exit(1, "No eligible labeled windows. Check the capture summaries.\n")
        temp.replace(args.out)
    finally:
        temp.unlink(missing_ok=True)
    print(f"Wrote {count} windows to {args.out}")


if __name__ == "__main__":
    main()
