#include <Arduino.h>
#include <WiFi.h>
#include <WiFiManager.h>
#include <string.h>

#include "esp_wifi.h"
#include "esp_err.h"
#include "ping/ping_sock.h"
#include "lwip/ip4_addr.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"

// Wi-Fi provisioning: based on the WiFiManager connection code supplied by you
static const char *SETUP_AP_SSID = "ESP32-CSI-Setup";
static const char *SETUP_AP_PASSWORD = "SetupESP32!7391";

static const uint32_t WIFI_RETRY_INTERVAL_MS = 5000;
static const uint32_t PING_INTERVAL_MS = 100;  // 10 pings/second for the first test

WiFiManager wifiManager;
unsigned long lastWiFiAttempt = 0;

// CSI capture and buffering
static constexpr size_t MAX_CSI_BYTES = 384;
static constexpr UBaseType_t CSI_QUEUE_LENGTH = 16;

struct CsiFrame {
  uint32_t sequence;
  uint32_t rxTimestampUs;
  int8_t rssi;
  int8_t noiseFloor;
  uint8_t channel;
  uint8_t firstWordInvalid;
  uint16_t length;
  int8_t bytes[MAX_CSI_BYTES];
};

QueueHandle_t csiQueue = nullptr;
uint8_t routerBssid[6] = {0};
uint32_t csiSequence = 0;
volatile uint32_t droppedFrames = 0;
volatile uint32_t oversizedFrames = 0;
volatile bool csiCaptureEnabled = false;

// Continuous ICMP ping session to the router's default gateway
esp_ping_handle_t pingHandle = nullptr;
bool csiSessionActive = false;

void printMacAddress(const uint8_t *mac) {
  for (int i = 0; i < 6; ++i) {
    if (i > 0) Serial.print(':');
    if (mac[i] < 0x10) Serial.print('0');
    Serial.print(mac[i], HEX);
  }
}

void configModeCallback(WiFiManager *manager) {
  Serial.println();
  Serial.println("Entered Wi-Fi configuration mode.");
  Serial.print("Setup AP SSID: ");
  Serial.println(manager->getConfigPortalSSID());
  Serial.print("Setup AP IP: ");
  Serial.println(WiFi.softAPIP());
}

bool connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setAutoReconnect(true);
  wifiManager.setDebugOutput(true);

  // If saved credentials cannot connect, allow the user to configure Wi-Fi.
  wifiManager.setConnectTimeout(20);
  wifiManager.setAPCallback(configModeCallback);

  Serial.println("Starting Wi-Fi configuration...");

  const bool connected = wifiManager.autoConnect(
      SETUP_AP_SSID,
      SETUP_AP_PASSWORD
  );

  if (!connected) {
    Serial.println("Wi-Fi setup failed or configuration portal did not complete.");
    return false;
  }

  Serial.println("Wi-Fi connected!");
  Serial.print("ESP32 IP: ");
  Serial.println(WiFi.localIP());
  Serial.print("Router gateway: ");
  Serial.println(WiFi.gatewayIP());

  lastWiFiAttempt = millis();
  return true;
}

// CSI RX callback
// Keep this callback short: copy the received CSI into a FreeRTOS queue only.
// The callback filters frames by the BSSID of the connected router.
void csiReceiveCallback(void *context, wifi_csi_info_t *info) {
  if (!csiCaptureEnabled || info == nullptr || info->buf == nullptr || csiQueue == nullptr) {
    return;
  }

  const uint8_t *expectedBssid = static_cast<const uint8_t *>(context);
  if (expectedBssid != nullptr && memcmp(info->mac, expectedBssid, 6) != 0) {
    return;
  }

  if (info->len <= 0 || static_cast<size_t>(info->len) > MAX_CSI_BYTES) {
    oversizedFrames++;
    return;
  }

  CsiFrame frame = {};
  frame.sequence = ++csiSequence;
  frame.rxTimestampUs = info->rx_ctrl.timestamp;
  frame.rssi = info->rx_ctrl.rssi;
  frame.noiseFloor = info->rx_ctrl.noise_floor;
  frame.channel = info->rx_ctrl.channel;
  frame.firstWordInvalid = info->first_word_invalid ? 1 : 0;
  frame.length = static_cast<uint16_t>(info->len);

  memcpy(frame.bytes, info->buf, frame.length);

  // STOP may have been received while this callback was copying the frame.
  if (!csiCaptureEnabled) return;

  if (xQueueSend(csiQueue, &frame, 0) != pdTRUE) {
    droppedFrames++;
  }
}

// Configure the ESP32 CSI receiver and identify the currently connected AP.
// LLTF-only configuration follows Espressif's router-receiver example and is
// a conservative starting point for router compatibility on classic ESP32.
bool initializeCSI() {
  wifi_ap_record_t apInfo = {};
  esp_err_t err = esp_wifi_sta_get_ap_info(&apInfo);
  if (err != ESP_OK) {
    Serial.printf("Could not read AP information: %s\n", esp_err_to_name(err));
    return false;
  }

  memcpy(routerBssid, apInfo.bssid, sizeof(routerBssid));

  // The pilot dataset uses the classic ESP32 20 MHz LLTF subcarrier layout.
  err = esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20);
  if (err != ESP_OK) {
    Serial.printf("Could not select 20 MHz bandwidth: %s\n", esp_err_to_name(err));
    return false;
  }

  Serial.print("Connected AP BSSID: ");
  printMacAddress(routerBssid);
  Serial.println();
  Serial.printf("Connected AP channel: %u\n", apInfo.primary);

  // Disable modem sleep to make the initial CSI sampling more consistent.
  err = esp_wifi_set_ps(WIFI_PS_NONE);
  if (err != ESP_OK) {
    Serial.printf("Warning: esp_wifi_set_ps failed: %s\n", esp_err_to_name(err));
  }

  wifi_csi_config_t csiConfig = {};
  csiConfig.lltf_en = true;
  csiConfig.htltf_en = false;
  csiConfig.stbc_htltf2_en = false;
  csiConfig.ltf_merge_en = true;
  csiConfig.channel_filter_en = true;
  csiConfig.manu_scale = true;
  csiConfig.shift = true;

  err = esp_wifi_set_csi_config(&csiConfig);
  if (err != ESP_OK) {
    Serial.printf("esp_wifi_set_csi_config failed: %s\n", esp_err_to_name(err));
    return false;
  }

  err = esp_wifi_set_csi_rx_cb(csiReceiveCallback, routerBssid);
  if (err != ESP_OK) {
    Serial.printf("esp_wifi_set_csi_rx_cb failed: %s\n", esp_err_to_name(err));
    return false;
  }

  err = esp_wifi_set_csi(true);
  if (err != ESP_OK) {
    Serial.printf("esp_wifi_set_csi(true) failed: %s\n", esp_err_to_name(err));
    return false;
  }

  csiCaptureEnabled = true;
  Serial.println("CSI receiver enabled.");
  return true;
}

// Start continuous pings to the router's gateway. The router's echo replies
// create downlink Wi-Fi packets from the AP to the ESP32, which produce CSI.
bool startRouterPing() {
  const IPAddress gateway = WiFi.gatewayIP();
  if (gateway == IPAddress(0, 0, 0, 0)) {
    Serial.println("Invalid gateway IP; cannot start router ping.");
    return false;
  }

  ip4_addr_t gatewayAddress;
  IP4_ADDR(&gatewayAddress, gateway[0], gateway[1], gateway[2], gateway[3]);

  esp_ping_config_t pingConfig = ESP_PING_DEFAULT_CONFIG();
  pingConfig.target_addr.u_addr.ip4.addr = gatewayAddress.addr;
  pingConfig.target_addr.type = ESP_IPADDR_TYPE_V4;
  pingConfig.count = 0;  // 0 = continuous ping session
  pingConfig.interval_ms = PING_INTERVAL_MS;
  pingConfig.task_stack_size = 3072;
  pingConfig.data_size = 1;

  esp_ping_callbacks_t callbacks = {};

  esp_err_t err = esp_ping_new_session(&pingConfig, &callbacks, &pingHandle);
  if (err != ESP_OK) {
    Serial.printf("esp_ping_new_session failed: %s\n", esp_err_to_name(err));
    pingHandle = nullptr;
    return false;
  }

  err = esp_ping_start(pingHandle);
  if (err != ESP_OK) {
    Serial.printf("esp_ping_start failed: %s\n", esp_err_to_name(err));
    esp_ping_delete_session(pingHandle);
    pingHandle = nullptr;
    return false;
  }

  Serial.print("Continuous ping started toward router gateway: ");
  Serial.println(gateway);
  Serial.println("Waiting for CSI frames from the router...");
  return true;
}

void stopRouterPing() {
  if (pingHandle != nullptr) {
    esp_ping_stop(pingHandle);
    esp_ping_delete_session(pingHandle);
    pingHandle = nullptr;
  }
}

// Enable/disable CSI capture without disconnecting from the access point.
void setCsiCapture(bool enable) {
  if (enable == csiCaptureEnabled) {
    Serial.printf("CSI capture is already %s.\n", enable ? "RUNNING" : "STOPPED");
    return;
  }

  if (enable) {
    esp_err_t err = esp_wifi_set_csi(true);
    if (err != ESP_OK) {
      Serial.printf("Could not start CSI: %s\n", esp_err_to_name(err));
      return;
    }
    csiCaptureEnabled = true;
    Serial.println("CSI capture STARTED.");
  } else {
    // Block new callback records first, then disable CSI at the Wi-Fi driver.
    csiCaptureEnabled = false;
    esp_err_t err = esp_wifi_set_csi(false);
    if (err != ESP_OK) {
      Serial.printf("Warning: esp_wifi_set_csi(false): %s\n", esp_err_to_name(err));
    }

    // Discard records captured before STOP so they are not printed after START.
    if (csiQueue != nullptr) {
      xQueueReset(csiQueue);
    }
    Serial.println("CSI capture STOPPED. Wi-Fi connection and router ping remain active.");
  }
}

void printStatus() {
  Serial.println("----- STATUS -----");
  Serial.printf("Wi-Fi: %s\n", WiFi.status() == WL_CONNECTED ? "CONNECTED" : "DISCONNECTED");
  if (WiFi.status() == WL_CONNECTED) {
    Serial.print("IP: ");
    Serial.println(WiFi.localIP());
    Serial.print("Gateway: ");
    Serial.println(WiFi.gatewayIP());
    Serial.print("AP BSSID: ");
    printMacAddress(routerBssid);
    Serial.println();
  }
  Serial.printf("CSI session: %s\n", csiSessionActive ? "READY" : "NOT READY");
  Serial.printf("CSI capture: %s\n", csiCaptureEnabled ? "RUNNING" : "STOPPED");
  Serial.printf("Dropped frames: %lu\n", (unsigned long)droppedFrames);
  Serial.printf("Oversized frames: %lu\n", (unsigned long)oversizedFrames);
  Serial.println("------------------");
}

// Read newline-terminated commands without blocking the CSI processing loop.
void handleSerialCommands() {
  static char command[24];
  static size_t commandLength = 0;

  while (Serial.available() > 0) {
    const char ch = (char)Serial.read();
    if (ch == '\r') continue;

    if (ch == '\n') {
      command[commandLength] = '\0';
      for (size_t i = 0; i < commandLength; ++i) {
        if (command[i] >= 'a' && command[i] <= 'z') command[i] -= ('a' - 'A');
      }

      if (strcmp(command, "STOP") == 0) {
        setCsiCapture(false);
      } else if (strcmp(command, "START") == 0) {
        if (WiFi.status() == WL_CONNECTED && csiSessionActive) {
          setCsiCapture(true);
        } else {
          Serial.println("Cannot START: Wi-Fi/CSI session is not ready yet.");
        }
      } else if (strcmp(command, "STATUS") == 0) {
        printStatus();
      } else if (commandLength > 0) {
        Serial.println("Unknown command. Use START, STOP, or STATUS.");
      }
      commandLength = 0;
    } else if (commandLength < sizeof(command) - 1) {
      command[commandLength++] = ch;
    } else {
      commandLength = 0;
      Serial.println("Command too long. Use START, STOP, or STATUS.");
    }
  }
}

// Print one captured CSI record as CSV to Serial Monitor / Serial Plotter log.
// Data bytes are kept as signed decimal values, separated by semicolons.
void printCsiFrame(const CsiFrame &frame) {
  Serial.printf(
      "CSI,%lu,%lu,%d,%d,%u,%u,%u,\"",
      static_cast<unsigned long>(frame.sequence),
      static_cast<unsigned long>(frame.rxTimestampUs),
      static_cast<int>(frame.rssi),
      static_cast<int>(frame.noiseFloor),
      static_cast<unsigned int>(frame.channel),
      static_cast<unsigned int>(frame.length),
      static_cast<unsigned int>(frame.firstWordInvalid)
  );

  for (uint16_t i = 0; i < frame.length; ++i) {
    if (i > 0) Serial.print(';');
    Serial.print(static_cast<int>(frame.bytes[i]));
  }

  Serial.println("\"");
}

void setup() {
  Serial.begin(115200);
  delay(500);

  Serial.println();
  Serial.println("ESP32 Router-based CSI Receiver starting...");
#ifdef WIFI_CSI_ENABLED
  Serial.printf("Wi-Fi CSI compile-time support: %d\n", WIFI_CSI_ENABLED);
#else
  Serial.println("Wi-Fi CSI compile-time support macro is not exposed by this core.");
#endif

  csiQueue = xQueueCreate(CSI_QUEUE_LENGTH, sizeof(CsiFrame));
  if (csiQueue == nullptr) {
    Serial.println("Failed to create CSI queue. Restarting...");
    delay(1000);
    ESP.restart();
  }

  if (!connectWiFi()) {
    Serial.println("Initial Wi-Fi setup did not complete. Restarting...");
    delay(1000);
    ESP.restart();
  }

  Serial.println("CSV format:");
  Serial.println("CSI,sequence,rx_timestamp_us,rssi,noise_floor,channel,len,first_word_invalid,\"csi_bytes\"");
  Serial.println("Serial commands: START | STOP | STATUS (send with newline)");
}

void loop() {
  handleSerialCommands();
  const unsigned long now = millis();

  // Recover Wi-Fi using the credentials previously saved by WiFiManager.
  if (WiFi.status() != WL_CONNECTED) {
    if (csiSessionActive) {
      csiCaptureEnabled = false;
      esp_wifi_set_csi(false);
      if (csiQueue != nullptr) xQueueReset(csiQueue);
      stopRouterPing();
      csiSessionActive = false;
      Serial.println("Wi-Fi lost. CSI/ping session paused.");
    }

    if (now - lastWiFiAttempt >= WIFI_RETRY_INTERVAL_MS) {
      lastWiFiAttempt = now;
      Serial.println("Wi-Fi disconnected; attempting reconnection...");
      WiFi.reconnect();
    }
    return;
  }

  // Start/restart CSI and ping after the Wi-Fi connection is available.
  if (!csiSessionActive) {
    if (initializeCSI()) {
      if (startRouterPing()) {
        csiSessionActive = true;
      } else {
        csiCaptureEnabled = false;
        esp_wifi_set_csi(false);
        delay(1000);
      }
    } else {
      delay(1000);
    }
  }

  // Do all printing outside the Wi-Fi CSI callback.
  CsiFrame frame;
  while (csiCaptureEnabled && csiQueue != nullptr && xQueueReceive(csiQueue, &frame, 0) == pdTRUE) {
    printCsiFrame(frame);
  }

  static uint32_t lastReportedDrops = 0;
  if (droppedFrames != lastReportedDrops) {
    Serial.printf("Warning: CSI queue dropped %lu frame(s).\n",
                  static_cast<unsigned long>(droppedFrames - lastReportedDrops));
    lastReportedDrops = droppedFrames;
  }

  static uint32_t lastReportedOversized = 0;
  if (oversizedFrames != lastReportedOversized) {
    Serial.printf("Warning: ignored %lu oversized CSI frame(s).\n",
                  static_cast<unsigned long>(oversizedFrames - lastReportedOversized));
    lastReportedOversized = oversizedFrames;
  }
}
