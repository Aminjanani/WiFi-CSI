"""Forward a completed raw text recording to an MQTT broker from the laptop."""

import argparse
import json
import os
from pathlib import Path
import re
import time


def batches(stream, size=32):
    batch = []
    for line in stream:
        batch.append(line.rstrip("\r\n"))
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def main():
    import paho.mqtt.client as mqtt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("--broker", required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--plain", action="store_true", help="Use a plaintext broker for a local test")
    args = parser.parse_args()
    meta = json.loads((args.session / "session.json").read_text(encoding="utf8"))
    for key in ("room", "node", "session"):
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", meta[key]):
            parser.error(f"{key} must be a single topic component made of letters, digits, dot, dash or underscore")
    topic = f"wifi-csi/v1/{meta['room']}/{meta['node']}/raw"
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    if os.getenv("MQTT_USERNAME"):
        client.username_pw_set(os.environ["MQTT_USERNAME"], os.getenv("MQTT_PASSWORD"))
    if not args.plain:
        client.tls_set()
    client.connect(args.broker, args.port or (1883 if args.plain else 8883))
    client.loop_start()
    first = 1
    try:
        deadline = time.monotonic() + 10
        while not client.is_connected():
            if time.monotonic() >= deadline:
                parser.exit(1, "Broker did not accept the connection.\n")
            time.sleep(0.05)
        with (args.session / "serial.log").open(encoding="utf8", errors="replace", newline="\n") as stream:
            for lines in batches(stream):
                event = dict(schema=1, session=meta["session"], first_line=first, lines=lines)
                receipt = client.publish(topic, json.dumps(event), qos=1, retain=False)
                receipt.wait_for_publish(timeout=10)
                if not receipt.is_published():
                    parser.exit(1, f"Missing acknowledgement at line {first}. Replay this session to retry.\n")
                first += len(lines)
    finally:
        client.disconnect()
        client.loop_stop()
    print(f"Broker acknowledged {first-1} text lines on {topic}.")


if __name__ == "__main__":
    main()
