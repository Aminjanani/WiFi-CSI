# Wi-Fi CSI people counting

The first experiment estimates **0, 1 or 2 people in one bedroom** using a 2.4 GHz router, one original ESP32 and a laptop. Presence is derived from the estimated count. One recording has one independently known count. A motionless person still counts.

## 01 · Set up the link

Place the router and receiver on different sides of the occupied area. Mark their positions, heights and antenna orientations. Fix the router's 2.4 GHz channel and select 20 MHz bandwidth. The ESP32 pings the gateway every 15 ms (about 100 CSI frames/s measured with the D-Link) and captures CSI from received frames whose source matches the associated AP. Other AP traffic can contribute frames. The nominal ping rate and actual CSI rate are different quantities. [Espressif router capture example](https://github.com/espressif/esp-csi/tree/master/examples/get-started/csi_recv_router)

The native ESP-IDF 5.5.2 project is in `firmware/esp32_csi_idf`. It targets the original ESP32. From an active ESP-IDF PowerShell, use the USB port shown for your board. On the development laptop the installed EIM activation script and current port are shown below.

```powershell
. 'C:\Espressif\tools\Microsoft.30aaf64.PowerShell_profile.ps1'
cd 'E:\Wifi CSI\WiFi-CSI\firmware\esp32_csi_idf'
idf.py build
idf.py -p COM13 flash
```

The receiver uses its saved station credentials on boot. To set or replace them, run the helper below from the Git checkout. The password is entered privately over local USB and stored by the ESP-IDF Wi-Fi driver in board NVS. Close `idf.py monitor` (Ctrl+]) and the logger before opening the port in another program.

```powershell
cd 'E:\Wifi CSI\WiFi-CSI'
# On a fresh checkout, create the Python environment once.
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m serial.tools.list_ports -v
.\.venv\Scripts\python -m src.data_collection.configure_wifi --port COM13
```

A successful join reports `wifi=ready`. Check for an increasing frame count or a nonzero packet rate in the logger. `START` and `STOP` control CSI output. `STATUS` reports connection state, frames and queue drops. The console runs at **921600 baud** (`CONFIG_ESP_CONSOLE_UART_CUSTOM`), because 100 lines/s do not fit through 115200. Boot ROM messages still use 115200 and appear garbled in `serial.log`, which is harmless. Each CSI line also carries the frame format: `CSI,seq,us,rssi,noise,channel,sig_mode,rate,mcs,cwb,stbc,sig_len,len,first_word_invalid,"iq"`. The parser still reads the older 9-field lines.

## 02 · Record independent sessions

Between trials, type `SCAN` in the logger or serial monitor for one passive 2.4 GHz scan. Each result gives a BSSID, channel, RSSI in dBm and SSID. The current CSI source carries `source=1`. The receiver pauses CSI during the scan and restores its previous capture state afterward.

Replace `COM13` with your port. This example gives you 20 seconds after the first valid packet to leave the room, then measures for 90 seconds and stops automatically. The raw file includes the settling period. The labeled dataset starts afterward. Use a longer delay if leaving and closing the door takes longer.

```powershell
.\.venv\Scripts\python -m src.data_collection.serial_capture --port COM13 --count 0 --activity empty --round r01 --settle 20 --seconds 90
```

For one person set `--count 1 --activity seated-still`. For two, try `--count 2 --activity one-seated-one-walking`. `--notes` records room layout, position, anonymous participant codes or anything unusual. Omit `--count` for an unlabeled recording. One person can start the timer and step out for an empty trial. Keep everyone in position until the timer ends. If someone enters early, discard that session from training by setting its manifest count to JSON `null`. The laptop displays measurements or predictions. The people conducting the trial supply the labels.

Each recording starts with one passive scan so `session.json` stores the CSI source AP (`source_bssid`, `source_ssid`, `source_channel`). Add `--expect-bssid YOUR_AP_BSSID` to abort, and clear the label, if the receiver is following any other AP. `planned_seconds`, `measured_seconds` and `completed` show whether the timer ran out or someone pressed Ctrl+C.

Before building the dataset, audit every session. The review lists the source, packet count, gaps and early stops, plus the sessions per round and count. To discard a bad trial, set its count to `none`. You can also correct a wrong count.

```powershell
.\.venv\Scripts\python -m src.review_sessions --source YOUR_AP_BSSID
.\.venv\Scripts\python -m src.review_sessions --set-count session_YYYYMMDDTHHMMSSffffffZ none
```

`src.build_dataset` searches `data/raw` recursively, so `--out data/raw/dlink` works. `--source <bssid>` keeps sessions from one AP only. Every window carries `source_bssid`.

Each uniquely numbered UTC session directory contains these files.

| File | Contents |
|---|---|
| `serial.log` | Every byte returned by the serial port, including diagnostics and malformed lines. Normally readable as text. |
| `csi_packets.csv` | One row per valid packet, with raw signed IQ, RX timestamp, RSSI, noise floor, channel, invalid-word flag, mean magnitude, host receipt time and original log line number. |
| `session.json` | Room, receiver, recording round (stored as `block`), activity, hand-supplied count and accepted line interval. Written on normal completion or Ctrl+C. An interrupted process can leave a raw-only recording. |

Record in **rounds**. A round is one set of 0, 1 and 2 people in shuffled order, about 15 minutes, all with the same `--round` value (r01, r02…). `--round` is required for labeled recordings. Several rounds per day are fine. Each round is held out once during evaluation, so the model is always tested on recordings it never trained on. Between rounds, change something realistic: door state, a chair, the time of day. Keep at least one round from another day, or after a router reboot, as a final drift check. Start with three rounds of three counts × still/moving, and add more rounds for scenarios that score poorly.

| True count | Include these cases |
|---|---|
| 0 | Empty and quiet, empty with the usual fan running, empty with someone walking outside the room. |
| 1 | Each volunteer separately, seated still off the direct link, standing still near it, then walking. |
| 2 | Both still, one moving, both moving. Swap places. Compare against one active person covering the same path. |

Keep background traffic, laptop position and door state comparable across counts. Include some trials with the same traffic or door change at different counts. Check the actual packet timestamps, queue-drop warnings, oversized packets and accepted windows. A sparse window means unavailable sensing. Counting it as an empty room would bias evaluation.

## 03 · Build the model CSV

```powershell
.\.venv\Scripts\python -m src.build_dataset
```

The builder replays labeled `data/raw/<session>/serial.log` files and writes `data/processed/windows.csv`. Raw IQ stays available for a future neural network. CSV can be inspected in Excel without making a workbook part of capture. A rebuild replaces the derived CSV only after generating eligible rows. Review the per-session acceptance messages.

Feature schema 3 uses **30-second windows evaluated every 2 seconds** on device time. A window needs an average of 5 packets/s covering 75 % of its length. A gap over 400 ms, backward timestamp or sequence, changed channel or stream reset starts a new segment. The expected input is a 128-byte classic ESP32 LLTF record on a 20 MHz link. The feature mask keeps bins +2…+26 and −26…−1. Each pair is signed imaginary then real, and magnitude is `hypot(imaginary, real)`. [Espressif CSI format](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/wifi-driver/wifi-vendor-features.html#wi-fi-channel-state-information)

The classic ESP32 cannot lock its receive gain, and the measured amplitude jumped by up to 6× between packets. Every packet is therefore divided by its own mean amplitude before the shape features are computed. Only `amp_scale` keeps the raw, gain-affected scale.

| Feature | Meaning |
|---|---|
| `rssi_mean`, `rssi_sd` | Received power and its jitter |
| `amp_scale` | Mean raw amplitude (gain-affected) |
| `shape_spread` | Unevenness of the normalized spectrum within a packet |
| `carrier_sd` | Median temporal SD of each normalized subcarrier |
| `carrier_step` | Mean change between consecutive 20 Hz samples (motion) |
| `pc1_share` | Fraction of variance in the strongest common pattern |
| `breath_power` | log10 power at 0.1–0.5 Hz in that pattern (breathing) |
| `motion_power` | log10 power at 0.5–3 Hz (body movement) |
| `state_separation` | How cleanly packets fall into the router's shape states (drops with large motion) |

Spectral features come from the first principal component of the normalized subcarriers, averaged into 20 Hz bins. Overlapping windows from one session are strongly correlated, so they are evaluated by round, never by window.

**Transmitter shape states.** Every D-Link frame arrives in one of 4 fixed spectrum shapes, chosen at random, about 25 % each and independent of the previous frame. This was measured in an empty room. Within a shape, subcarriers vary by about 2.6 %. Between shapes they differ by about 50 %. All of these frames are the same 67-byte HT20 ping reply, so frame format does not explain it; the likely cause is multi-antenna transmission with a per-frame phase. In each window, packets are clustered into `SHAPE_STATES = 4` states with seeded k-means, and each packet is divided by its own state's mean. This cut empty-room `carrier_sd` from 0.30 to 0.029. `SHAPE_STATES` belongs to the router. Measure it once in an empty room and set it to 1 for a router without states, because clustering a stream that has no states would erase slow real changes.

For an older `serial.log`, supply a `session.json` matching a new capture, using the directory name as `session`, the correct physical block, and the actual first/last labeled line numbers (one-based, inclusive). Use this only if the operator knows the count and can confirm the same radio configuration. Legacy recordings without labels can still be inspected.

## 04 · Train, evaluate and view a count

Evaluation is **leave-one-round-out**. Each round is held out once, and the model is trained on all the others. The trainer refuses fewer than two rounds. Decide on any `--final-block` drift-check round **before** you look at results, and use it only once.

```powershell
.\.venv\Scripts\python -m src.train_baseline --task presence --out models/presence-v1
.\.venv\Scripts\python -m src.train_baseline --task count --final-block r04 --out models/count-v1
.\.venv\Scripts\python -m src.data_collection.serial_capture --port COM13 --model models/count-v1/forest.joblib
```

`--features dynamic` (the default) uses only movement and breathing: `carrier_step`, `motion_power`, `breath_snr`, `breath_stable` and `breath_peak`. In rounds r01 and r02, the static shape features also tracked chairs, door and blankets, and an empty room drifted 5 dB in an hour. With dynamic features, leave-one-round-out presence got 12 of 12 recordings and 94 % of windows right; with all non-drifting features, it got 10 of 12. `--features nondrifting` and `--features all` exist for comparison. Breathing evidence comes from 30-second windows: `breath_snr` is the strongest 9–36 breaths/min peak against a 1–4 Hz noise floor on the 5 best subcarriers, `breath_rate` is that peak's rate, and `breath_stable` is the share of the last minute's windows with a calm (8–20/min) and consistent rate. Start with `--task presence` (empty vs. occupied). Once that works, try `--task count`. The trainer compares a majority guess, an RSSI-only forest and the full forest on identical out-of-round predictions. `metrics.json` reports window accuracy, **session vote accuracy** (one vote per recording, which is the honest unit), macro-F1, the confusion matrix, per-round accuracy, empty-room false positives, occupied recall and feature importance. `cv_predictions.csv` holds every held-out window. A count classifier can return only 0, 1 or 2.

After training, the laptop prints a timestamped estimate and presence for each accepted live window. Predictions pause if qualifying data stops. On a future dashboard show stale data as unknown. A small temporal 1D CNN over time and subcarriers is the next model experiment once these baselines and hard cases are measured. Use the same held-out sessions for comparison and reserve a fresh day after tuning.

## 04b · Label CSI with the webcam

The laptop camera can label every moment of a recording instead of one hand label per session. `src.data_collection.camera_labels` runs YOLO11-pose (`models/vision/yolo11n-pose.pt`, downloaded on first use) and saves **no images**. It writes one row per frame to `camera_labels.csv` with the laptop's UTC time, the person count, each tracked person's activity and their keypoints, normalized to the frame.

Activity comes from simple, checkable rules on the keypoints. `lying` means the torso is more than 55° from vertical. `running` and `walking` mean the hips move faster than 1.2 and 0.25 body heights per second. `sitting` means a thigh is roughly horizontal. `standing` means legs are visible and straight. `upright` means legs are hidden, so the person could be sitting or standing. `unknown` means shoulders or hips are not visible. A hip drop of more than half a body height into `lying` within 1.5 s is flagged as a fall. Each frame becomes a scene: `empty`, one activity, or a `+`-joined mix such as `sitting+walking`.

```powershell
.\.venv\Scripts\python -m pip install -r requirements-camera.txt
.\.venv\Scripts\yolo settings sync=False
.\.venv\Scripts\python -m src.data_collection.camera_labels --out data/interim/camera_check.csv --show --seconds 30
.\.venv\Scripts\python -m src.data_collection.serial_capture --port COM13 --out data/raw/dlink --expect-bssid YOUR_AP_BSSID --round c01 --camera --camera-show --settle 30 --seconds 900 --notes "Corners. Camera-labeled free activity"
```

The first command installs the camera dependencies. Run `yolo settings sync=False` once to turn off Ultralytics' anonymous usage analytics. Position the laptop so its camera sees the bed, desk and walking area. The 30-second check shows what it detects; press `q` to close the preview. Without `--count`, the camera supplies the labels (`label_source: camera`). The builder aligns frames to each CSI window through the laptop receipt times in `csi_packets.csv`, and adds `cam_frames`, `cam_persons`, `cam_activity`, `cam_agree` and `cam_fall`. For camera sessions, `count` is the camera's person count. Train activity from CSI with:

```powershell
.\.venv\Scripts\python -m src.train_baseline --task activity --out models/activity-v1
```

Windows where fewer than 70 % of camera frames agree (`--min-agree`) or the scene is `unknown` are left out. Treat camera labels as a teacher with its own errors: hand-check a few minutes against the preview, and keep anyone who is outside the camera's view out of the room. Simulate falls slowly onto a mattress. Ultralytics 8.3.40 installs a top-level `tests` package, so the project's `tests/__init__.py` keeps `python -m unittest tests.test_pipeline` pointing at the project's tests.

## 05 · Forward logs with MQTT when a server is available

Capture works locally over USB. The optional laptop publisher replays a completed session, so broker downtime and MQTT traffic do not change a training recording. Install its dependency and substitute your broker hostname and real session directory.

```powershell
.\.venv\Scripts\python -m pip install -r requirements-mqtt.txt
.\.venv\Scripts\python -m src.publish_session data/raw/SESSION_DIRECTORY --broker YOUR_BROKER
```

The topic is `wifi-csi/v1/<room>/<node>/raw`. Messages are QoS 1, unretained JSON with `schema`, `session`, `first_line` and up to 32 text `lines`, including original diagnostics and CSI. Use `MQTT_USERNAME` and `MQTT_PASSWORD` if the broker requires credentials. TLS uses the system trust store by default. The receiver should deduplicate by room, node, session and line number. Replay after interruption can redeliver acknowledged lines. The local log preserves original bytes if boot output contains invalid text.

## 06 · Extend hardware and dashboard

Start with one receiver. If a stable single link misses a quiet corner or confuses one mover with two people, add the second ESP32 as **another receiver** at a different angle to the same router. Record both simultaneously with distinct node IDs and the same block. Compare receiver A, receiver B and fused estimates on the same held-out trials. The current model scores one link at a time. Fusion needs a shared trial ID and clock alignment with measured skew. Host serial receipt times are approximate. If the router cannot generate repeatable downlink traffic, a dedicated ESP32 transmitter plus one receiver is a different experiment that still gives one link.

For localization, start with one person and marked zones such as desk, doorway and bed. Record ground-truth zone and evaluate a confusion matrix before drawing a dot. For activity, first label still versus moving, then test standing versus walking and walking versus running. Include quiet standing and standing while talking in matched positions. CSI measures channel changes, so a gesture correlated with speech can look like a talking detector. Multi-person activity needs time-aligned labels for each person and location. Room count alone cannot supply them.

The future server can subscribe to separate window-estimate messages keyed by room, model version, receiver set and time interval. Display count, presence, age, data quality and validated zone/activity. A weak or missing measurement should produce an unknown state. Keep independent operator labels outside predicted events.

See the [research review](reports/evaluation/reference-review.md) for the paper, RuView implementation, release corrections and public datasets.
