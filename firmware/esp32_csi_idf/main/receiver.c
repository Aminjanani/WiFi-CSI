#include <inttypes.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>

#include "driver/uart.h"
#include "esp_check.h"
#include "esp_event.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "nvs_flash.h"
#include "ping/ping_sock.h"

enum { MAX_CSI = 384, COMMAND_CAPACITY = 208 };
typedef struct {
    uint32_t generation, seq, us;
    int rssi, noise, channel;
    unsigned length, invalid;
    int8_t iq[MAX_CSI];
} frame_t;

typedef struct {
    bool got_ip;
    uint32_t gateway;
    unsigned reason;
} link_event_t;

static QueueHandle_t frames, links;
// The Wi-Fi task and app task share only the callback gate and its counters.
static portMUX_TYPE gate = portMUX_INITIALIZER_UNLOCKED;
static bool accepting;
static uint8_t source_mac[6];
static uint32_t generation, sequence, dropped, invalid_length;

// The app task owns the connection, ping lifecycle, commands and serial output.
static bool configured, ready, paused, reconnect;
static int64_t retry_at;
static esp_ping_handle_t ping;

static void on_csi(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) return;
    frame_t f;
    portENTER_CRITICAL(&gate);
    bool keep = accepting && memcmp(info->mac, source_mac, sizeof(source_mac)) == 0;
    if (keep && (info->len == 0 || info->len > MAX_CSI || info->len % 2)) {
        ++invalid_length;
        keep = false;
    }
    if (keep) {
        f.generation = generation;
        f.seq = ++sequence;
    }
    portEXIT_CRITICAL(&gate);
    if (!keep) return;
    f.us = info->rx_ctrl.timestamp;
    f.rssi = info->rx_ctrl.rssi;
    f.noise = info->rx_ctrl.noise_floor;
    f.channel = info->rx_ctrl.channel;
    f.invalid = info->first_word_invalid;
    f.length = info->len;
    memcpy(f.iq, info->buf, f.length);
    if (xQueueSend(frames, &f, 0) != pdTRUE) {
        portENTER_CRITICAL(&gate);
        ++dropped;
        portEXIT_CRITICAL(&gate);
    }
}

static void update_gate(void)
{
    portENTER_CRITICAL(&gate);
    accepting = ready && !paused;
    ++generation;
    portEXIT_CRITICAL(&gate);
    xQueueReset(frames);
}

static void on_link(void *ctx, esp_event_base_t base, int32_t id, void *data)
{
    link_event_t event = {0};
    if (base == IP_EVENT) {
        event.got_ip = true;
        event.gateway = ((ip_event_got_ip_t *)data)->ip_info.gw.addr;
    } else {
        event.reason = ((wifi_event_sta_disconnected_t *)data)->reason;
    }
    xQueueSend(links, &event, portMAX_DELAY);
}

static void stop_link(void)
{
    ready = false;
    update_gate();
    if (ping) {
        ESP_ERROR_CHECK_WITHOUT_ABORT(esp_ping_stop(ping));
        ESP_ERROR_CHECK_WITHOUT_ABORT(esp_ping_delete_session(ping));
        ping = NULL;
    }
}

static esp_err_t start_link(uint32_t gateway)
{
    wifi_ap_record_t ap;
    ESP_RETURN_ON_ERROR(esp_wifi_sta_get_ap_info(&ap), "csi", "read AP");
    if (!gateway) return ESP_ERR_INVALID_STATE;
    esp_ping_config_t config = ESP_PING_DEFAULT_CONFIG();
    ip_addr_set_ip4_u32(&config.target_addr, gateway);
    config.count = ESP_PING_COUNT_INFINITE;
    config.interval_ms = 100;
    config.timeout_ms = 100;
    config.data_size = 1;
    esp_ping_callbacks_t callbacks = {0};
    ESP_RETURN_ON_ERROR(esp_ping_new_session(&config, &callbacks, &ping), "csi", "create ping");
    ESP_RETURN_ON_ERROR(esp_ping_start(ping), "csi", "start ping");
    portENTER_CRITICAL(&gate);
    memcpy(source_mac, ap.bssid, sizeof(source_mac));
    portEXIT_CRITICAL(&gate);
    printf("Connected AP BSSID: " MACSTR " channel %u\n", MAC2STR(ap.bssid), ap.primary);
    ready = true;
    update_gate();
    return ESP_OK;
}

static void print_frame(const frame_t *f)
{
    // One stdout transaction prevents log messages from splitting a CSV record.
    static char line[128 + 5 * MAX_CSI];
    int n = snprintf(line, sizeof(line), "CSI,%" PRIu32 ",%" PRIu32 ",%d,%d,%d,%u,%u,\"",
                     f->seq, f->us, f->rssi, f->noise, f->channel, f->length, f->invalid);
    for (unsigned i = 0; i < f->length; ++i) {
        n += snprintf(line + n, sizeof(line) - n, "%d%c", f->iq[i], i+1 == f->length ? '"' : ';');
    }
    puts(line);
}

static int hex_digit(char c)
{
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return 10 + c - 'a';
    return -1;
}

static bool decode(const char *hex, uint8_t *out, size_t capacity)
{
    size_t n = strlen(hex);
    if (n % 2 || n / 2 > capacity) return false;
    for (size_t i = 0; i < n; i += 2) {
        int a = hex_digit(hex[i]), b = hex_digit(hex[i+1]);
        if (a < 0 || b < 0 || !(a || b)) return false;
        out[i/2] = (uint8_t)((a << 4) | b);
    }
    return true;
}

static void join_network(char *args)
{
    char *password = strchr(args, ' ');
    wifi_config_t config = {0};
    if (!password) {
        puts("WIFI ERROR invalid arguments");
        return;
    }
    *password++ = 0;
    if (!*args || !decode(args, config.sta.ssid, sizeof(config.sta.ssid)) ||
        !decode(password, config.sta.password, sizeof(config.sta.password))) {
        puts("WIFI ERROR invalid credentials format");
        return;
    }
    stop_link();
    esp_wifi_disconnect();
    esp_err_t err = esp_wifi_set_config(WIFI_IF_STA, &config);
    if (err != ESP_OK) {
        printf("WIFI ERROR %s\n", esp_err_to_name(err));
        return;
    }
    configured = true;
    reconnect = true;
    retry_at = esp_timer_get_time() + 200000;
    puts("WIFI SAVED");
}

static void scan_networks(void)
{
    const bool was_paused = paused;
    paused = true;
    update_gate();
    puts("SCAN BEGIN");
    wifi_scan_config_t config = {.scan_type = WIFI_SCAN_TYPE_PASSIVE, .show_hidden = true};
    esp_err_t err = esp_wifi_scan_start(&config, true);
    uint16_t count = 0;
    if (err == ESP_OK) {
        err = esp_wifi_scan_get_ap_num(&count);
        for (uint16_t i = 0; err == ESP_OK && i < count; ++i) {
            wifi_ap_record_t ap;
            err = esp_wifi_scan_get_ap_record(&ap);
            if (err != ESP_OK) break;
            char name[4 * 32 + 1] = {0};
            size_t n = 0;
            for (size_t j = 0; j < 32 && ap.ssid[j]; ++j) {
                unsigned c = ap.ssid[j];
                if (c >= 32 && c <= 126 && c != '\\') name[n++] = (char)c;
                else n += snprintf(name + n, sizeof(name) - n, "\\x%02x", c);
            }
            printf("AP " MACSTR " channel=%u rssi=%d source=%d ssid=%s\n",
                   MAC2STR(ap.bssid), ap.primary, ap.rssi,
                   ready && memcmp(ap.bssid, source_mac, sizeof(source_mac)) == 0,
                   n ? name : "<hidden>");
        }
        if (err != ESP_OK) esp_wifi_clear_ap_list();
    }
    printf("SCAN END %s count=%u\n", esp_err_to_name(err), count);
    paused = was_paused;
    update_gate();
}

static void command(char *line)
{
    if (!strcasecmp(line, "STATUS")) {
        portENTER_CRITICAL(&gate);
        uint32_t seen = sequence, lost = dropped, bad = invalid_length;
        portEXIT_CRITICAL(&gate);
        printf("IDF-CSI STATUS wifi=%s capture=%s frames=%" PRIu32
               " dropped=%" PRIu32 " invalid_length=%" PRIu32 "\n",
               ready ? "ready" : configured ? "connecting" : "unconfigured",
               paused ? "stopped" : ready ? "running" : "waiting", seen, lost, bad);
    } else if (!strcasecmp(line, "STOP") || !strcasecmp(line, "START")) {
        paused = !strcasecmp(line, "STOP");
        update_gate();
        puts(paused ? "CSI capture STOPPED." : "CSI capture STARTED.");
    } else if (!strcasecmp(line, "SCAN")) {
        scan_networks();
    } else if (!strncmp(line, "JOIN ", 5)) {
        join_network(line + 5);
    } else if (*line) {
        puts("Unknown command. Use START, STOP, STATUS or SCAN.");
    }
}

static void read_commands(void)
{
    static char line[COMMAND_CAPACITY];
    static size_t used;
    static bool overflow;
    uint8_t buf[64];
    int n = uart_read_bytes(UART_NUM_0, buf, sizeof(buf), 0);
    for (int i = 0; i < n; ++i) {
        char c = (char)buf[i];
        if (c == '\r') continue;
        if (c == '\n') {
            line[used] = 0;
            if (!overflow) command(line);
            memset(line, 0, sizeof(line));
            used = 0;
            overflow = false;
        } else if (used < sizeof(line)-1) {
            line[used++] = c;
        } else {
            overflow = true;
        }
    }
}

void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_create_default_wifi_sta() ? ESP_OK : ESP_ERR_NO_MEM);
    frames = xQueueCreate(16, sizeof(frame_t));
    links = xQueueCreate(8, sizeof(link_event_t));
    ESP_ERROR_CHECK(frames && links ? ESP_OK : ESP_ERR_NO_MEM);
    ESP_ERROR_CHECK(uart_driver_install(UART_NUM_0, 512, 0, 0, NULL, 0));
    wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&init));
    ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, WIFI_EVENT_STA_DISCONNECTED, on_link, NULL));
    ESP_ERROR_CHECK(esp_event_handler_register(IP_EVENT, IP_EVENT_STA_GOT_IP, on_link, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BW_HT20));
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    wifi_csi_config_t csi = {
        .lltf_en = true, .ltf_merge_en = true, .channel_filter_en = true,
        .manu_scale = true, .shift = 1,
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(on_csi, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
    wifi_config_t saved;
    ESP_ERROR_CHECK(esp_wifi_get_config(WIFI_IF_STA, &saved));
    configured = saved.sta.ssid[0] != 0;
    reconnect = configured;
    puts("ESP32 Router-based CSI receiver (ESP-IDF). USB commands: STATUS, START, STOP, SCAN.");
    if (!configured) puts("Set Wi-Fi with python -m src.data_collection.configure_wifi --port PORT");

    for (;;) {
        read_commands();
        link_event_t link;
        while (xQueueReceive(links, &link, 0)) {
            stop_link();
            reconnect = !link.got_ip;
            if (!link.got_ip) {
                printf("Wi-Fi lost. Disconnect reason %u.\n", link.reason);
                retry_at = esp_timer_get_time() + 3000000;
            } else if (start_link(link.gateway) != ESP_OK) {
                stop_link();
                esp_wifi_disconnect();
                reconnect = true;
                retry_at = esp_timer_get_time() + 3000000;
            }
        }
        if (configured && reconnect && esp_timer_get_time() >= retry_at) {
            reconnect = esp_wifi_connect() != ESP_OK;
            retry_at = esp_timer_get_time() + 3000000;
        }
        frame_t f;
        if (xQueueReceive(frames, &f, pdMS_TO_TICKS(10)) && ready && !paused && f.generation == generation) {
            print_frame(&f);
        }
    }
}
