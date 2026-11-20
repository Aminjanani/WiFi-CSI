"""Set the native ESP-IDF receiver's router credentials over its USB cable."""

import argparse
from getpass import getpass
import time

import serial

if __package__:
    from .serial_capture import open_board
else:
    from serial_capture import open_board


def wait_status(port, timeout, ready=False):
    deadline = time.monotonic() + timeout
    next_probe = 0
    while time.monotonic() < deadline:
        if time.monotonic() >= next_probe:
            port.write(b"STATUS\n")
            next_probe = time.monotonic() + 1
        line = port.readline().decode("ascii", errors="replace").strip()
        if line.startswith("Wi-Fi lost."):
            print(line)
        if line.startswith("IDF-CSI STATUS ") and (not ready or "wifi=ready " in line):
            return line
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--ssid", help="Router name, including exact spacing and capitalization")
    args = parser.parse_args()
    name = args.ssid if args.ssid is not None else input("2.4 GHz router SSID: ")
    ssid = name.encode("utf8")
    password = getpass(f"Password for {name} (hidden, Enter for an open network): ").encode("utf8")
    if not 1 <= len(ssid) <= 32 or len(password) > 64 or b"\0" in ssid + password:
        parser.error("expected a 1-32 byte SSID and up to 64 password bytes, without NUL")
    try:
        with open_board(args.port, 921600, timeout=0.25, write_timeout=2) as port:
            if not wait_status(port, 12):
                parser.exit(1, "Receiver did not identify as IDF-CSI. Check the firmware and port.\n")
            port.write(f"JOIN {ssid.hex()} {password.hex()}\n".encode("ascii"))
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline:
                reply = port.readline().strip()
                if reply == b"WIFI SAVED":
                    break
                if reply.startswith(b"WIFI ERROR"):
                    parser.exit(1, reply.decode("ascii", errors="replace") + "\n")
            else:
                parser.exit(1, "Receiver did not acknowledge saving the credentials.\n")
            print("Credentials saved on the board. Waiting for the router and gateway.")
            status = wait_status(port, 40, ready=True)
            if not status:
                parser.exit(1, "Router connection timed out. Check the Wi-Fi details and run again.\n")
            print(status)
    except serial.SerialException as exc:
        parser.exit(2, f"USB serial error: {exc}\n")


if __name__ == "__main__":
    main()
