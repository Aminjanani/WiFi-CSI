"""Log the receiver's serial bytes and calculate packet magnitudes during capture."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import time

import serial

if __package__:
    from .csi import FEATURES, PACKET_FIELDS, Windows, breaks_stream, parse
else:
    from csi import FEATURES, PACKET_FIELDS, Windows, breaks_stream, parse


def forward_commands(port):
    try:
        while True:
            command = input().strip().upper()
            if command in {"START", "STOP", "STATUS", "SCAN"}:
                port.write((command + "\n").encode("ascii"))
    except (EOFError, OSError, serial.SerialException):
        return


def capture(port, directory, args, model=None):
    metadata = dict(schema=1, session=directory.name, room=args.room, node=args.node,
                    block=args.block, count=args.count, activity=args.activity,
                    notes=args.notes, started_utc=datetime.now(timezone.utc).isoformat(),
                    first_line=None, last_line=0, valid_packets=0, malformed=0)
    windows = Windows()
    first_packet_at = None
    report_at = time.monotonic()
    report_packets = 0
    pending = bytearray()
    line_no = 0
    with (directory / "serial.log").open("xb") as raw, (directory / "csi_packets.csv").open(
            "x", newline="", encoding="utf8") as computed:
        writer = csv.writer(computed)
        writer.writerow(("log_line", "host_received_utc", *PACKET_FIELDS))
        try:
            while True:
                now = time.monotonic()
                if now - report_at >= 5:
                    valid = metadata["valid_packets"]
                    print(f"CSI {((valid - report_packets) / (now - report_at)):.1f} packets/s "
                          f"| valid {valid} | malformed {metadata['malformed']}")
                    if model is not None and valid == report_packets:
                        print("Count unknown: no fresh CSI.")
                    report_at, report_packets = now, valid
                if first_packet_at is not None and args.seconds and now >= first_packet_at + args.settle + args.seconds:
                    break
                chunk = port.readline()
                if not chunk:
                    continue
                raw.write(chunk)
                raw.flush()
                pending.extend(chunk)
                if not chunk.endswith(b"\n"):
                    continue
                line_no += 1
                line = pending.decode("utf8", errors="replace").rstrip("\r\n")
                pending.clear()
                try:
                    packet = parse(line)
                except (ValueError, csv.Error):
                    metadata["malformed"] += 1
                    windows.reset()
                    continue
                if packet is None:
                    if breaks_stream(line):
                        windows.reset()
                    if line:
                        print(line)
                    continue
                writer.writerow((line_no, datetime.now(timezone.utc).isoformat(), *packet.csv_row()))
                computed.flush()
                metadata["valid_packets"] += 1
                if first_packet_at is None:
                    first_packet_at = time.monotonic()
                    print(f"CSI received. Settling for {args.settle} seconds.")
                if time.monotonic() < first_packet_at + args.settle:
                    continue
                if metadata["first_line"] is None:
                    metadata["first_line"] = line_no
                    print("Measurement started.")
                if model is not None:
                    row = windows.push(packet)
                    if row:
                        count = int(model.predict([[row[name] for name in FEATURES]])[0])
                        print(f"RX {row['end_us']} us | estimated count {count} | present {count > 0}")
        except KeyboardInterrupt:
            pass
        metadata["last_line"] = line_no
    (directory / "session.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    print(f"{metadata['valid_packets']} valid packets, {metadata['malformed']} malformed. {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--room", default="bedroom")
    parser.add_argument("--node", default="rx01")
    parser.add_argument("--block", default=datetime.now().date().isoformat(),
                        help="Independent recording day shared by all counts and simultaneous receivers")
    parser.add_argument("--count", type=int, choices=(0, 1, 2), help="Known constant headcount")
    parser.add_argument("--activity", default="unspecified", help="Operator description of this scenario")
    parser.add_argument("--notes", default="", help="Placement, layout, confounders or participant codes")
    parser.add_argument("--settle", type=int, default=0, help="Discard initial seconds from training")
    parser.add_argument("--seconds", type=int, default=0, help="Measure for this many seconds, 0 until Ctrl+C")
    parser.add_argument("--model", type=Path, help="Trusted local v1 random forest for live estimates")
    args = parser.parse_args()
    if min(args.seconds, args.settle) < 0:
        parser.error("time intervals must be nonnegative")
    model = None
    if args.model:
        import joblib
        artifact = joblib.load(args.model)
        if artifact["schema"] != 1 or tuple(artifact["features"]) != FEATURES:
            parser.error("incompatible feature schema")
        model = artifact["model"]
    session = "session_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = args.out / session
    try:
        with serial.Serial(args.port, args.baud, timeout=0.25) as port:
            directory.mkdir(parents=True, exist_ok=False)
            print(f"Saving to {directory}. Commands: START, STOP, STATUS, SCAN. Ctrl+C ends the recording.")
            threading.Thread(target=forward_commands, args=(port,), daemon=True).start()
            capture(port, directory, args, model)
    except serial.SerialException as error:
        parser.exit(2, f"Serial error: {error}\n")


if __name__ == "__main__":
    main()
