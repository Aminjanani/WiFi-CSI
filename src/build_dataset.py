"""Replay labeled serial logs into the v1 window-level people-counting CSV."""

import argparse
import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from .data_collection.csi import FEATURES, Windows, breaks_stream, parse


CONTEXT = ("session", "block", "room", "node", "activity", "count")
FIELDS = (*CONTEXT, "segment", "start_us", "end_us", "packets", "channel", *FEATURES)


def add_session(directory, writer):
    meta = json.loads((directory / "session.json").read_text(encoding="utf8"))
    if meta.get("count") is None or meta.get("first_line") is None:
        print(f"Skipping {directory.name}: unlabeled or empty measurement.")
        return 0
    if meta["schema"] != 1 or type(meta["count"]) is not int or meta["count"] not in (0, 1, 2):
        raise ValueError(f"Unsupported label or schema in {directory}")
    if meta["session"] != directory.name or not all(meta[k] for k in ("block", "room", "node")):
        raise ValueError(f"Missing or inconsistent session identity in {directory}")
    context = {key: meta[key] for key in CONTEXT}
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
                writer.writerow({**context, **row})
                written += 1
    row = windows.finish()
    if row:
        writer.writerow({**context, **row})
        written += 1
    print(f"{directory.name}: {written} windows, {windows.rejected} rejected buckets, "
          f"{bad} malformed and {unsupported} unsupported packets.")
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/windows.csv"))
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
            for directory in sorted(p for p in args.raw.iterdir() if p.is_dir()):
                if (directory / "session.json").exists():
                    count += add_session(directory, writer)
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
