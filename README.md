# Wi-Fi CSI people counting

The first experiment estimates **0, 1 or 2 people in one bedroom** using a 2.4 GHz router, one original ESP32 and a laptop. Presence is derived from the estimated count. One recording has one independently known count. A motionless person still counts.

## 01 · Set up the link

Place the router and receiver on different sides of the occupied area. Mark their positions, heights and antenna orientations. Fix the router's 2.4 GHz channel and select 20 MHz bandwidth. The ESP32 pings the gateway every 100 ms and captures CSI from received frames whose source matches the associated AP. Other AP traffic can contribute frames. The nominal ping rate and actual CSI rate are different quantities. [Espressif router capture example](https://github.com/espressif/esp-csi/tree/master/examples/get-started/csi_recv_router)

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

A successful join reports `wifi=ready`. Check for an increasing frame count or a nonzero packet rate in the logger. `START` and `STOP` control CSI output. `STATUS` reports connection state, frames and queue drops. CSI uses the same signed-byte CSV format as the Arduino version.

## 02 · Record independent sessions

Replace `COM13` with your port. This example gives you 20 seconds after the first valid packet to leave the room, then measures for 90 seconds and stops automatically. The raw file includes the settling period. The labeled dataset starts afterward. Use a longer delay if leaving and closing the door takes longer.

```powershell
.\.venv\Scripts\python -m src.data_collection.serial_capture --port COM13 --count 0 --activity empty --block day01 --settle 20 --seconds 90
```

For one person set `--count 1 --activity seated-still`. For two, try `--count 2 --activity one-seated-one-walking`. `--notes` records room layout, position, anonymous participant codes or anything unusual. Omit `--count` for an unlabeled recording. One person can start the timer and step out for an empty trial. Keep everyone in position until the timer ends. If someone enters early, discard that session from training by setting its manifest count to JSON `null`. The laptop displays measurements or predictions. The people conducting the trial supply the labels.

Each uniquely numbered UTC session directory contains these files.

| File | Contents |
|---|---|
| `serial.log` | Every byte returned by the serial port, including diagnostics and malformed lines. Normally readable as text. |
| `csi_packets.csv` | One row per valid packet, with raw signed IQ, RX timestamp, RSSI, noise floor, channel, invalid-word flag, mean magnitude, host receipt time and original log line number. |
| `session.json` | Room, receiver, recording block, activity, hand-supplied count and accepted line interval. Written on normal completion or Ctrl+C. An interrupted process can leave a raw-only recording. |

Keep one `--block` value per recording day across all counts and receivers. Use the same `--room` and `--node` for the same installation. Record several independent trials per scenario. A suggested starting schedule is three 90-second repeats per count on each of three days. Include all counts each day and vary their order. Extend collection when a scenario has little coverage.

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

V1 uses two-second, non-overlapping device-time windows. The expected input is a 128-byte classic ESP32 LLTF record on a 20 MHz link. The feature mask keeps common active bins +2…+26 and −26…−1. This omits DC, guard bins and both possibly invalid first-word pairs consistently. The packet CSV mean uses all available pairs after discarding the first four bytes when flagged. Each pair is signed imaginary then real. Magnitude is `hypot(imaginary, real)`. [Espressif CSI format](https://docs.espressif.com/projects/esp-idf/en/stable/esp32/api-guides/wifi-driver/wifi-vendor-features.html#wi-fi-channel-state-information)

A window needs at least 10 packets spanning 1.5 seconds. A gap over 400 ms, backward timestamp, repeated/backward sequence, changed channel, or stream reset starts a new segment. Treat these as pilot settings to revisit after inspecting hardware traces. Features include mean and SD of RSSI, average amplitude, within-packet amplitude spread, typical temporal SD across subcarriers, and mean absolute change between consecutive amplitude vectors. Packet count, timestamps, node, session, and annotations are available for auditing outside the model inputs.

For an older `serial.log`, supply a `session.json` matching a new capture, using the directory name as `session`, the correct physical block, and the actual first/last labeled line numbers (one-based, inclusive). Use this only if the operator knows the count and can confirm the same radio configuration. Legacy recordings without labels can still be inspected.

## 04 · Train, evaluate and view a count

Choose the held-out day **before** inspecting its results. Collect all three counts on both sides of the split. The pilot asks how the same room behaves on another day. More rooms and people require their own independent holdouts.

```powershell
.\.venv\Scripts\python -m src.train_baseline --test-block day03 --out models/bedroom-v1
.\.venv\Scripts\python -m src.data_collection.serial_capture --port COM13 --model models/bedroom-v1/forest.joblib
```

The trainer compares a majority guess, an RSSI-only forest and a six-feature random forest on identical windows. It writes `metrics.json`, `held_out_predictions.csv` and `forest.joblib`. Review the 0/1/2 confusion matrix, exact accuracy, macro-F1, mean absolute count error, empty-room false positives and occupied recall. Inspect errors by activity and by session. Count windows from the same trial as correlated observations. A classifier trained on 0/1/2 can return only those counts. Three occupants require a new labeled experiment.

After training, the laptop prints a timestamped estimate and presence for each accepted live window. Predictions pause if qualifying data stops. On a future dashboard show stale data as unknown. A small temporal 1D CNN over time and subcarriers is the next model experiment once these baselines and hard cases are measured. Use the same held-out sessions for comparison and reserve a fresh day after tuning.

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
