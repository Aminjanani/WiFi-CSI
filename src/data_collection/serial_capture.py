"""Log the receiver's serial bytes and calculate packet magnitudes during capture."""

import argparse
import csv
from datetime import datetime, timezone
import json
import re
import subprocess
import sys
from pathlib import Path
import threading
import time

import serial

if __package__:
    from .csi import FEATURES, PACKET_FIELDS, SCHEMA, Windows, breaks_stream, parse
else:
    from csi import FEATURES, PACKET_FIELDS, SCHEMA, Windows, breaks_stream, parse


def open_board(name, baud, **options):
    """Open the ESP32's USB serial port without resetting the board.

    DTR and RTS drive the board's auto-reset circuit. Opening with them asserted
    either reboots the ESP32 or, depending on timing, holds it in reset (RTS on,
    DTR off), so it sends nothing. Release both before the port opens.
    """
    port = serial.Serial(baudrate=baud, **options)
    port.port = name
    port.dtr = False
    port.rts = False
    port.open()
    return port


def forward_commands(port):
    try:
        while True:
            command = input().strip().upper()
            if command in {"START", "STOP", "STATUS", "SCAN"}:
                port.write((command + "\n").encode("ascii"))
    except (EOFError, OSError, serial.SerialException):
        return


SOURCE_LINE = re.compile(r"^AP ([0-9a-f:]{17}) channel=(\d+) rssi=(-?\d+) source=1 ssid=(.*)$", re.I)
BOOT_SOURCE = re.compile(r"^Connected AP BSSID: ([0-9a-f:]{17}) channel (\d+)", re.I)


def capture(port, directory, args, model=None):
    metadata = dict(schema=1, session=directory.name, room=args.room, node=args.node,
                    block=args.block, count=args.count, activity=args.activity,
                    notes=args.notes, started_utc=datetime.now(timezone.utc).isoformat(),
                    first_line=None, last_line=0, valid_packets=0, malformed=0,
                    source_bssid=None, source_ssid=None, source_channel=None,
                    planned_seconds=args.seconds, measured_seconds=0.0, completed=False,
                    camera=bool(getattr(args, "camera", False)),
                    label_source=("camera" if getattr(args, "camera", False) and args.count is None else "hand"))
    expect = (getattr(args, "expect_bssid", None) or "").lower() or None
    # Ask the receiver which AP it is following before the settle clock starts.
    verify = getattr(args, "verify_source", False) and hasattr(port, "write")
    if verify:
        port.write(b"SCAN\n")
    verify_deadline = time.monotonic() + 15
    opened = time.monotonic()
    bytes_seen = 0
    measure_started = None
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
                    metadata["completed"] = True
                    break
                if verify and now > verify_deadline:
                    verify = False
                    if expect:
                        print(f"ABORT: the receiver did not answer SCAN, so {expect} could not be confirmed.")
                        metadata["count"] = metadata["label_source"] = None
                        break
                    print("WARNING: receiver did not report its CSI source. Continuing without verification.")
                if first_packet_at is None and now - opened >= getattr(args, "wait", 30):
                    metadata["count"] = metadata["label_source"] = None
                    if not bytes_seen:
                        print("ABORT: no bytes at all from the board. Press its EN/RST button or replug USB, "
                              "and close any other program using the port.")
                    else:
                        print("ABORT: the board is talking but sends no CSI. Check Wi-Fi with STATUS "
                              "(wifi=ready, capture=running) and that the router is on.")
                    break
                chunk = port.readline()
                if not chunk:
                    continue
                bytes_seen += len(chunk)
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
                    found = SOURCE_LINE.match(line) or BOOT_SOURCE.match(line)
                    if found:
                        bssid, channel = found.group(1).lower(), int(found.group(2))
                        if metadata["source_bssid"] not in (None, bssid):
                            print(f"WARNING: CSI source changed to {bssid} during the recording.")
                            metadata["notes"] = (metadata["notes"] + " [source changed]").strip()
                            metadata["count"] = metadata["label_source"] = None
                        metadata["source_bssid"], metadata["source_channel"] = bssid, channel
                        if found.re is SOURCE_LINE:
                            metadata["source_ssid"] = found.group(4)
                        if expect and bssid != expect:
                            print(f"ABORT: receiver follows {bssid}, expected {expect}.")
                            metadata["count"] = metadata["label_source"] = None
                            break
                    if line.startswith("SCAN END") and verify:
                        verify = False
                        if metadata["source_bssid"] is None:
                            print("WARNING: the scan did not identify the CSI source AP.")
                            if expect:
                                print(f"ABORT: could not confirm {expect}.")
                                metadata["count"] = metadata["label_source"] = None
                                break
                        else:
                            print(f"CSI source: {metadata['source_ssid'] or '?'} "
                                  f"{metadata['source_bssid']} channel {metadata['source_channel']}")
                    if breaks_stream(line):
                        windows.reset()
                    # Boot ROM output is 115200 baud and arrives garbled at 921600; it stays in serial.log only.
                    if line and not line.startswith("AP ") and line.isascii() and line.isprintable():
                        print(line)
                    continue
                if verify:
                    continue
                writer.writerow((line_no, datetime.now(timezone.utc).isoformat(), *packet.csv_row()))
                computed.flush()
                metadata["valid_packets"] += 1
                if first_packet_at is None:
                    if packet.channel != (metadata["source_channel"] or packet.channel):
                        continue
                    first_packet_at = time.monotonic()
                    print(f"CSI received. Settling for {args.settle} seconds.")
                if time.monotonic() < first_packet_at + args.settle:
                    continue
                if metadata["first_line"] is None:
                    metadata["first_line"] = line_no
                    measure_started = time.monotonic()
                    print("Measurement started.")
                if model is not None:
                    row = windows.push(packet)
                    if row:
                        value = int(model.predict([[row[name] for name in args.model_features]])[0])
                        if getattr(args, "task", "count") == "presence":
                            print(f"RX {row['end_us']} us | last 10 s | present {value > 0}")
                        else:
                            print(f"RX {row['end_us']} us | last 10 s | estimated count {value} | present {value > 0}")
        except KeyboardInterrupt:
            print("Stopped early.")
        metadata["last_line"] = line_no
        if measure_started is not None:
            metadata["measured_seconds"] = round(time.monotonic() - measure_started, 1)
    (directory / "session.json").write_text(json.dumps(metadata, indent=2), encoding="utf8")
    print(f"{metadata['valid_packets']} valid packets, {metadata['malformed']} malformed, "
          f"{metadata['measured_seconds']} s measured. {directory}")
    if args.seconds and not metadata["completed"] and metadata["count"] is not None:
        print("Recording ended before the planned duration. Review it with src.review_sessions.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--baud", type=int, default=921600, help="Must match the firmware console (921600 since the 50 Hz update)")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument("--room", default="bedroom")
    parser.add_argument("--node", default="rx01")
    parser.add_argument("--round", "--block", dest="block",
                        help="Recording round, e.g. r01: one shuffled set of 0, 1 and 2 people. "
                             "Required with --count. Stored as 'block' in session.json")
    parser.add_argument("--count", type=int, choices=(0, 1, 2), help="Known constant headcount")
    parser.add_argument("--activity", default="unspecified", help="Operator description of this scenario")
    parser.add_argument("--notes", default="", help="Placement, layout, confounders or participant codes")
    parser.add_argument("--settle", type=int, default=0, help="Discard initial seconds from training")
    parser.add_argument("--wait", type=int, default=30, help="Abort if no CSI arrives within this many seconds")
    parser.add_argument("--seconds", type=int, default=0, help="Measure for this many seconds, 0 until Ctrl+C")
    parser.add_argument("--expect-bssid", help="Abort unless the receiver follows this AP, e.g. aa:bb:cc:dd:ee:ff")
    parser.add_argument("--no-verify", dest="verify_source", action="store_false",
                        help="Skip the start-of-recording SCAN that records the CSI source AP")
    parser.add_argument("--model", type=Path, help="Trusted local random forest from src.train_baseline for live estimates")
    parser.add_argument("--camera", action="store_true",
                        help="Also run the webcam labeler (no images saved). Without --count, "
                             "the camera supplies the labels; --round is still required")
    parser.add_argument("--camera-show", action="store_true", help="Show the webcam preview window")
    parser.add_argument("--camera-device", type=int, default=0)
    args = parser.parse_args()
    if (args.count is not None or args.camera) and not args.block:
        parser.error("labeled recordings need --round, e.g. --round r01")
    if min(args.seconds, args.settle) < 0:
        parser.error("time intervals must be nonnegative")
    model = None
    if args.model:
        import joblib
        artifact = joblib.load(args.model)
        if artifact["schema"] != SCHEMA or not set(artifact["features"]) <= set(FEATURES):
            parser.error("incompatible feature schema")
        model = artifact["model"]
        args.task = artifact.get("task", "count")
        args.model_features = tuple(artifact["features"])
    session = "session_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = args.out / session
    try:
        with open_board(args.port, args.baud, timeout=0.25) as port:
            directory.mkdir(parents=True, exist_ok=False)
            print(f"Saving to {directory}. Commands: START, STOP, STATUS, SCAN. Ctrl+C ends the recording.")
            threading.Thread(target=forward_commands, args=(port,), daemon=True).start()
            camera = None
            if args.camera:
                command = [sys.executable, "-m", "src.data_collection.camera_labels",
                           "--out", str(directory / "camera_labels.csv"), "--device", str(args.camera_device)]
                camera = subprocess.Popen(command + (["--show"] if args.camera_show else []))
            try:
                capture(port, directory, args, model)
            finally:
                if camera is not None:
                    camera.terminate()   # rows are flushed per frame, so nothing is lost
                    camera.wait(timeout=10)
    except serial.SerialException as error:
        parser.exit(2, f"Serial error: {error}\n")


if __name__ == "__main__":
    main()
