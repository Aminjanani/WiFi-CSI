"""Capture ESP32 CSI serial output to a raw log and a CSI-only CSV file.

Usage:
    python serial_capture.py --port COM4
    python serial_capture.py --port COM4 --out data/raw

Stop with Ctrl+C. Send START / STOP / STATUS in Arduino Serial Monitor instead
of running this script at the same time (only one program can own the COM port).
"""

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import threading

try:
    import serial
except ImportError:
    print("pyserial is missing. Install it with: python -m pip install pyserial")
    raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description="Save ESP32 CSI serial records")
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM4")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--out", default="data/raw", help="Output parent directory")
    args = parser.parse_args()

    session_name = "session_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.out) / session_name
    session_dir.mkdir(parents=True, exist_ok=False)
    raw_path = session_dir / "serial.log"
    csv_path = session_dir / "csi_packets.csv"

    print(f"Opening {args.port} at {args.baud} baud...")
    print("Close Arduino Serial Monitor/Plotter before starting this script.")
    print("Capturing. Type START, STOP, or STATUS and press Enter to control the ESP32.")
    print("Press Ctrl+C to finish. Keep Arduino Serial Monitor closed while this script runs.")

    packet_count = 0
    try:
        # timeout keeps Ctrl+C responsive. ESP32 may reset when the port opens;
        # wait briefly for its boot messages before writing the session log.
        with serial.Serial(args.port, args.baud, timeout=1) as ser, \
             raw_path.open("w", encoding="utf-8", newline="") as raw_file, \
             csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            csv_file.write('sequence,rx_timestamp_us,rssi,noise_floor,channel,len,first_word_invalid,csi_bytes\n')
            csv_file.flush()

            def command_input():
                while True:
                    try:
                        command = input("ESP32> ").strip().upper()
                    except EOFError:
                        return
                    if command in {"START", "STOP", "STATUS"}:
                        ser.write((command + "\n").encode("ascii"))
                    elif command:
                        print("Valid commands: START, STOP, STATUS")

            threading.Thread(target=command_input, daemon=True).start()
            time.sleep(2)
            ser.reset_input_buffer()
            while True:
                data = ser.readline()
                if not data:
                    continue
                line = data.decode("utf-8", errors="replace").rstrip("\r\n")
                raw_file.write(line + "\n")
                raw_file.flush()

                if line.startswith("CSI,"):
                    csv_file.write(line + "\n")
                    csv_file.flush()
                    packet_count += 1
                    if packet_count % 100 == 0:
                        print(f"Saved {packet_count} CSI packets", end="\r", flush=True)
                elif line:
                    print(line)

    except KeyboardInterrupt:
        print("\nCapture stopped by user.")
    except serial.SerialException as exc:
        print(f"Serial error: {exc}", file=sys.stderr)
        raise SystemExit(2)
    finally:
        print(f"CSI packets saved: {packet_count}")
        print(f"Raw log: {raw_path}")
        print(f"CSI CSV: {csv_path}")


if __name__ == "__main__":
    main()