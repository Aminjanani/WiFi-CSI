"""Audit recorded sessions before building the dataset, and correct or discard labels."""

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path


def stats(directory, meta):
    first, last = meta.get("first_line"), meta.get("last_line") or 0
    stamps = []
    path = directory / "csi_packets.csv"
    if first is not None and path.exists():
        with path.open(encoding="utf8", newline="") as handle:
            for row in csv.DictReader(handle):
                if first <= int(row["log_line"]) <= last:
                    stamps.append(int(row["rx_timestamp_us"]))
    gaps = sum(b - a > 400_000 or b <= a for a, b in zip(stamps, stamps[1:]))
    span = (stamps[-1] - stamps[0]) / 1e6 if len(stamps) > 1 else 0.0
    return len(stamps), span, gaps


def problems(meta, packets, span, gaps, source):
    found = []
    if meta.get("count") is None:
        found.append("unlabeled")
    if not packets:
        found.append("no measured data (skipped by the builder)")
    elif meta.get("planned_seconds") and not meta.get("completed"):
        found.append("stopped early")
    if not meta.get("source_bssid"):
        found.append("source unknown")
    elif source and meta["source_bssid"].lower() != source.lower():
        found.append("other source")
    if span and packets / span < 25:  # firmware targets about 50 packets/s
        found.append(f"low rate {packets / span:.1f}/s")
    if gaps:
        found.append(f"{gaps} gaps")
    if meta.get("malformed"):
        found.append(f"{meta['malformed']} malformed")
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw"))
    parser.add_argument("--source", help="Expected AP BSSID, e.g. aa:bb:cc:dd:ee:ff")
    parser.add_argument("--set-count", nargs=2, metavar=("SESSION", "COUNT"),
                        help="Relabel a session: COUNT is 0, 1, 2 or none (exclude from training)")
    args = parser.parse_args()
    sessions = sorted({p.parent for p in args.raw.rglob("session.json")})

    if args.set_count:
        name, value = args.set_count
        matches = [d for d in sessions if d.name == name]
        if len(matches) != 1:
            parser.error(f"session {name} not found under {args.raw}")
        if value.lower() not in ("0", "1", "2", "none", "null"):
            parser.error("COUNT must be 0, 1, 2 or none")
        path = matches[0] / "session.json"
        meta = json.loads(path.read_text(encoding="utf8"))
        old = meta["count"]
        meta["count"] = None if value.lower() in ("none", "null") else int(value)
        if meta["count"] is None:
            meta["label_source"] = None      # also stop camera labels from being used
        meta.setdefault("relabels", []).append(dict(old=old, new=meta["count"]))
        path.write_text(json.dumps(meta, indent=2), encoding="utf8")
        print(f"{name}: count {old} -> {meta['count']}")
        return

    grid = defaultdict(lambda: defaultdict(int))
    print(f"{'session':32} {'round':10} {'n':>2} {'activity':22} {'source':17} {'pkts':>5} {'sec':>5}  issues")
    for directory in sessions:
        meta = json.loads((directory / "session.json").read_text(encoding="utf8"))
        packets, span, gaps = stats(directory, meta)
        issues = problems(meta, packets, span, gaps, args.source)
        count = meta.get("count")
        if count is not None and packets and "other source" not in issues:
            grid[meta.get("block")][count] += 1
        print(f"{directory.name:32} {str(meta.get('block'))[:10]:10} {'-' if count is None else count:>2} "
              f"{str(meta.get('activity'))[:22]:22} {str(meta.get('source_bssid') or '?'):17} "
              f"{packets:5} {span:5.0f}  {', '.join(issues) or 'ok'}")
    print("\nLabeled sessions per round (count 0 / 1 / 2):")
    for block, counts in sorted(grid.items()):
        missing = [str(c) for c in (0, 1, 2) if not counts[c]]
        note = f"  missing count {', '.join(missing)}" if missing else ""
        print(f"  {block:12} {counts[0]} / {counts[1]} / {counts[2]}{note}")
    if len(grid) < 2:
        print("  Evaluation needs at least two rounds, each with all three counts.")


if __name__ == "__main__":
    main()
