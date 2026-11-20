"""Label headcount and activity from the laptop webcam while CSI is being recorded.

Runs YOLO11-pose on each frame and turns body keypoints into one of: walking, running,
standing, sitting, upright (standing or sitting, legs hidden), lying, unknown. A fast
drop from upright to lying is flagged as a fall. No images are saved: each processed
frame becomes one CSV row with the laptop's UTC time, the person count, each tracked
person's activity, and their keypoints normalized to the frame size.

    python -m src.data_collection.camera_labels --out data/interim/camera_test.csv --show --seconds 30

Usually started by serial_capture --camera, which writes camera_labels.csv into the
CSI session folder so the dataset builder can align the two by host time.
"""

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
import json
from math import atan2, degrees, hypot
from pathlib import Path
import time

FIELDS = ("host_utc", "frame", "persons", "scene", "fall", "people")
WALK_SPEED = 0.25     # hip movement in body heights per second
RUN_SPEED = 1.2
LYING_ANGLE = 55      # torso angle from vertical, degrees
FALL_DROP = 0.5       # hip drop in body heights within FALL_SECONDS
FALL_SECONDS = 1.5
MIN_CONF = 0.5        # keypoint confidence needed to use a keypoint
# COCO keypoint indices
SHOULDERS, HIPS, KNEES, ANKLES = (5, 6), (11, 12), (13, 14), (15, 16)


def midpoint(xy, conf, pair):
    if min(conf[pair[0]], conf[pair[1]]) < MIN_CONF:
        return None
    return ((xy[pair[0]][0] + xy[pair[1]][0]) / 2, (xy[pair[0]][1] + xy[pair[1]][1]) / 2)


def classify(xy, conf, height, history, now):
    """Return (activity, speed in body heights/s, torso angle, fall flag) for one person."""
    shoulder, hip = midpoint(xy, conf, SHOULDERS), midpoint(xy, conf, HIPS)
    if shoulder is None or hip is None or height <= 0:
        return "unknown", 0.0, None, False
    angle = degrees(atan2(abs(hip[0] - shoulder[0]), abs(hip[1] - shoulder[1]) + 1e-9))
    history.append((now, hip, height))
    while history and history[0][0] < now - 2.0:
        history.popleft()
    past = [h for h in history if h[0] <= now - 0.8] or [history[0]]
    t0, hip0, h0 = past[-1]
    scale = (height + h0) / 2
    speed = hypot(hip[0] - hip0[0], hip[1] - hip0[1]) / scale / max(now - t0, 1e-3) if now > t0 else 0.0
    fall = False
    if angle > LYING_ANGLE:
        recent = [h for h in history if h[0] >= now - FALL_SECONDS]
        drop = (hip[1] - min(h[1][1] for h in recent)) / scale if recent else 0.0
        fall = drop > FALL_DROP
        return "lying", speed, angle, fall
    if speed > RUN_SPEED:
        return "running", speed, angle, fall
    if speed > WALK_SPEED:
        return "walking", speed, angle, fall
    knee = midpoint(xy, conf, KNEES)
    if knee is None:
        return "upright", speed, angle, fall
    thigh = hypot(knee[0] - hip[0], knee[1] - hip[1])
    torso = hypot(hip[0] - shoulder[0], hip[1] - shoulder[1])
    # Sitting: thigh roughly horizontal (knee near hip height) or very short in the image.
    if thigh < 0.45 * torso or abs(knee[1] - hip[1]) < 0.5 * thigh:
        return "sitting", speed, angle, fall
    return "standing", speed, angle, fall


def scene_label(activities):
    """One label per frame: 'empty', a single activity, or a sorted '+'-joined mix."""
    if not activities:
        return "empty"
    return "+".join(sorted(activities))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="CSV file to create")
    parser.add_argument("--device", type=int, default=0, help="Webcam index")
    parser.add_argument("--model", default="models/vision/yolo11n-pose.pt")
    parser.add_argument("--imgsz", type=int, default=480)
    parser.add_argument("--conf", type=float, default=0.5, help="Minimum person detection confidence")
    parser.add_argument("--seconds", type=float, default=0, help="Stop after this long, 0 = until stopped")
    parser.add_argument("--show", action="store_true", help="Show a live preview window (nothing is saved)")
    args = parser.parse_args()

    import cv2
    from ultralytics import YOLO

    model = YOLO(args.model)
    camera = cv2.VideoCapture(args.device, cv2.CAP_DSHOW)
    if not camera.isOpened():
        parser.exit(2, f"Camera {args.device} could not be opened.\n")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tracks = {}
    started = time.monotonic()
    frame_no = 0
    with args.out.open("x", newline="", encoding="utf8") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        print(f"Camera labels -> {args.out}. No images are saved.", flush=True)
        try:
            while not args.seconds or time.monotonic() - started < args.seconds:
                ok, frame = camera.read()
                if not ok:
                    print("Camera frame unavailable.", flush=True)
                    time.sleep(0.2)
                    continue
                now = time.monotonic()
                stamp = datetime.now(timezone.utc).isoformat()
                result = model.track(frame, persist=True, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
                frame_no += 1
                height_px, width_px = frame.shape[:2]
                people, activities, fell = [], [], False
                if result.keypoints is not None and len(result.boxes):
                    ids = result.boxes.id
                    for k in range(len(result.boxes)):
                        track = int(ids[k]) if ids is not None else -(k + 1)
                        xy = result.keypoints.xy[k].tolist()
                        conf = result.keypoints.conf[k].tolist() if result.keypoints.conf is not None else [1.0] * 17
                        x1, y1, x2, y2 = result.boxes.xyxy[k].tolist()
                        label, speed, angle, fall = classify(xy, conf, y2 - y1, tracks.setdefault(track, deque()), now)
                        fell |= fall
                        activities.append(label)
                        people.append(dict(id=track, activity=label, speed=round(speed, 3),
                                           torso_deg=None if angle is None else round(angle, 1),
                                           box=[round(x1 / width_px, 3), round(y1 / height_px, 3),
                                                round(x2 / width_px, 3), round(y2 / height_px, 3)],
                                           kp=[[round(x / width_px, 3), round(y / height_px, 3), round(c, 2)]
                                               for (x, y), c in zip(xy, conf)]))
                for track in [t for t, h in tracks.items() if h and h[-1][0] < now - 5]:
                    del tracks[track]
                scene = scene_label(activities)
                writer.writerow((stamp, frame_no, len(people), scene, int(fell),
                                 json.dumps(people, separators=(",", ":"))))
                handle.flush()
                if args.show:
                    view = result.plot(boxes=True)
                    cv2.putText(view, f"{len(people)} people | {scene}{' | FALL' if fell else ''}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imshow("CSI camera labels (not recorded)", view)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except KeyboardInterrupt:
            pass
    camera.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"Camera labels: {frame_no} frames.", flush=True)


if __name__ == "__main__":
    main()
