/*
 * PadelIQ ESP32-CAM Firmware  (board: AI-Thinker ESP32-CAM)
 * ---------------------------------------------------------------------------
 * Low-latency MJPEG stream for padeliq_live_stream.py
 *
 *   Stream   : http://<ip>:81/stream     (multipart/x-mixed-replace MJPEG)
 *   Snapshot : http://<ip>/capture       (single JPEG)
 *   Status   : http://<ip>/              (simple info page)
 *   mDNS     : http://padeliq-cam.local:81/stream
 *
 * Arduino IDE settings:
 *   Board            : "AI Thinker ESP32-CAM"   (esp32 by Espressif, core 2.x or 3.x)
 *   PSRAM            : Enabled
 *   Partition Scheme : "Huge APP (3MB No OTA/1MB SPIFFS)"
 *   CPU Frequency    : 240MHz,  Flash: 80MHz QIO
 *
 * Flashing: GPIO0 -> GND, press RESET, upload, then remove GPIO0-GND and press RESET.
 * Use a solid 5V / 2A supply: weak USB power causes brownouts and stream drops.
 *
 * PC side:
 *   python padeliq_live_stream.py --source http://<ip>:81/stream
 *   (or run_live.bat -> option [2])
 */

#include "esp_camera.h"
#include "esp_http_server.h"
#include "esp_timer.h"
#include "soc/soc.h"
#include "soc/rtc_cntl_reg.h"
#include <WiFi.h>
#include <ESPmDNS.h>

// ============================ USER SETTINGS =================================
const char *WIFI_SSID     = "YOUR_WIFI_SSID";
const char *WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char *MDNS_NAME     = "padeliq-cam";

// Optional static IP (keeps run_live.bat's saved IP valid). Set USE_STATIC_IP to true.
#define USE_STATIC_IP false
IPAddress STATIC_IP(192, 168, 1, 50);
IPAddress GATEWAY(192, 168, 1, 1);
IPAddress SUBNET(255, 255, 255, 0);
IPAddress DNS1(8, 8, 8, 8);

// FRAMESIZE_SVGA (800x600) with PSRAM, FRAMESIZE_VGA (640x480) without.
// For the highest frame rate on weak Wi-Fi, force VGA by setting this to true.
#define FORCE_VGA false
#define JPEG_QUALITY_PSRAM 12      // 10 = better quality / bigger frames, 15 = faster
#define JPEG_QUALITY_NO_PSRAM 14
#define FLIP_VERTICAL false        // set true if the camera is mounted upside down
#define MIRROR_HORIZONTAL false
// ============================================================================

// AI-Thinker ESP32-CAM pin map
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22
#define LED_FLASH_GPIO     4
#define LED_STATUS_GPIO   33      // on-board red LED (active LOW)

#define PART_BOUNDARY "123456789000000000000987654321"
static const char *STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char *STREAM_BOUNDARY     = "\r\n--" PART_BOUNDARY "\r\n";
static const char *STREAM_PART         = "Content-Type: image/jpeg\r\nContent-Length: %u\r\nX-Timestamp: %lld\r\n\r\n";

httpd_handle_t control_httpd = NULL;   // port 80
httpd_handle_t stream_httpd  = NULL;   // port 81

volatile uint32_t g_frames_sent = 0;
volatile float    g_stream_fps  = 0.0f;
volatile uint8_t  g_stream_clients = 0;
unsigned long     g_last_wifi_check = 0;
unsigned long     g_wifi_lost_since = 0;

// ----------------------------------------------------------------------------
// Camera
// ----------------------------------------------------------------------------
bool initCamera() {
  camera_config_t config = {};
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href  = HREF_GPIO_NUM;
  config.pin_sccb_sda = SIOD_GPIO_NUM;
  config.pin_sccb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn  = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.grab_mode    = CAMERA_GRAB_LATEST;   // always the newest frame => lowest latency

  if (psramFound()) {
    config.frame_size   = FORCE_VGA ? FRAMESIZE_VGA : FRAMESIZE_SVGA;
    config.jpeg_quality = JPEG_QUALITY_PSRAM;
    config.fb_count     = 2;                  // double buffering: capture while sending
    config.fb_location  = CAMERA_FB_IN_PSRAM;
    Serial.println("[CAM] PSRAM found -> SVGA/VGA, 2 frame buffers");
  } else {
    config.frame_size   = FRAMESIZE_VGA;
    config.jpeg_quality = JPEG_QUALITY_NO_PSRAM;
    config.fb_count     = 1;
    config.fb_location  = CAMERA_FB_IN_DRAM;
    Serial.println("[CAM] WARNING: no PSRAM -> VGA, 1 frame buffer (enable PSRAM in Tools menu)");
  }

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("[CAM] init failed: 0x%x\n", err);
    return false;
  }

  sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_vflip(s, FLIP_VERTICAL ? 1 : 0);
    s->set_hmirror(s, MIRROR_HORIZONTAL ? 1 : 0);
    s->set_brightness(s, 0);
    s->set_contrast(s, 1);
    s->set_saturation(s, 0);
    s->set_whitebal(s, 1);
    s->set_awb_gain(s, 1);
    s->set_exposure_ctrl(s, 1);
    s->set_aec2(s, 1);          // better exposure for fast motion under court lights
    s->set_gain_ctrl(s, 1);
    s->set_lenc(s, 1);
    s->set_dcw(s, 1);
    s->set_bpc(s, 1);
    s->set_wpc(s, 1);
  }
  return true;
}

// ----------------------------------------------------------------------------
// HTTP handlers
// ----------------------------------------------------------------------------
static esp_err_t stream_handler(httpd_req_t *req) {
  char part_buf[128];
  esp_err_t res = httpd_resp_set_type(req, STREAM_CONTENT_TYPE);
  if (res != ESP_OK) return res;
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-cache, no-store, must-revalidate");
  httpd_resp_set_hdr(req, "X-Framerate", "30");

  g_stream_clients++;
  Serial.println("[HTTP] stream client connected");
  digitalWrite(LED_STATUS_GPIO, LOW);

  int64_t last_frame_us = esp_timer_get_time();
  uint32_t frames_window = 0;
  int64_t window_start_us = last_frame_us;
  uint8_t consecutive_failures = 0;

  while (true) {
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("[CAM] frame capture failed");
      if (++consecutive_failures > 20) { res = ESP_FAIL; break; }
      delay(5);
      continue;
    }
    consecutive_failures = 0;

    res = httpd_resp_send_chunk(req, STREAM_BOUNDARY, strlen(STREAM_BOUNDARY));
    if (res == ESP_OK) {
      size_t hlen = snprintf(part_buf, sizeof(part_buf), STREAM_PART, (unsigned)fb->len, (long long)fb->timestamp.tv_sec);
      res = httpd_resp_send_chunk(req, part_buf, hlen);
    }
    if (res == ESP_OK) {
      res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);
    }
    esp_camera_fb_return(fb);
    if (res != ESP_OK) break;   // client went away

    g_frames_sent++;
    frames_window++;
    int64_t now_us = esp_timer_get_time();
    if (now_us - window_start_us >= 2000000) {
      g_stream_fps = frames_window * 1000000.0f / (float)(now_us - window_start_us);
      Serial.printf("[HTTP] streaming %.1f fps, RSSI %d dBm\n", g_stream_fps, WiFi.RSSI());
      frames_window = 0;
      window_start_us = now_us;
    }
    last_frame_us = now_us;
  }

  if (g_stream_clients > 0) g_stream_clients--;
  if (g_stream_clients == 0) digitalWrite(LED_STATUS_GPIO, HIGH);
  Serial.println("[HTTP] stream client disconnected");
  return res;
}

static esp_err_t capture_handler(httpd_req_t *req) {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    httpd_resp_send_500(req);
    return ESP_FAIL;
  }
  httpd_resp_set_type(req, "image/jpeg");
  httpd_resp_set_hdr(req, "Content-Disposition", "inline; filename=padeliq.jpg");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  esp_err_t res = httpd_resp_send(req, (const char *)fb->buf, fb->len);
  esp_camera_fb_return(fb);
  return res;
}

static esp_err_t index_handler(httpd_req_t *req) {
  char html[900];
  String ip = WiFi.localIP().toString();
  snprintf(html, sizeof(html),
           "<!doctype html><html><head><meta name='viewport' content='width=device-width'>"
           "<title>PadelIQ Cam</title></head><body style='font-family:sans-serif;background:#111;color:#eee'>"
           "<h2>PadelIQ ESP32-CAM</h2>"
           "<p>Stream URL: <b>http://%s:81/stream</b></p>"
           "<p>Clients: %u &nbsp; FPS: %.1f &nbsp; RSSI: %d dBm &nbsp; Uptime: %lus</p>"
           "<p>PSRAM: %s</p>"
           "<p><a style='color:#0fa' href='/capture'>Snapshot</a></p>"
           "<img src='http://%s:81/stream' style='max-width:100%%'>"
           "</body></html>",
           ip.c_str(), g_stream_clients, g_stream_fps, WiFi.RSSI(), millis() / 1000,
           psramFound() ? "yes" : "NO", ip.c_str());
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, html, HTTPD_RESP_USE_STRLEN);
}

void startServers() {
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.ctrl_port = 32768;
  config.max_open_sockets = 4;
  config.lru_purge_enable = true;          // drop stale sockets instead of refusing new clients

  httpd_uri_t index_uri   = { .uri = "/",        .method = HTTP_GET, .handler = index_handler,   .user_ctx = NULL };
  httpd_uri_t capture_uri = { .uri = "/capture", .method = HTTP_GET, .handler = capture_handler, .user_ctx = NULL };
  if (httpd_start(&control_httpd, &config) == ESP_OK) {
    httpd_register_uri_handler(control_httpd, &index_uri);
    httpd_register_uri_handler(control_httpd, &capture_uri);
  }

  httpd_config_t sconfig = HTTPD_DEFAULT_CONFIG();
  sconfig.server_port = 81;
  sconfig.ctrl_port = 32769;
  sconfig.max_open_sockets = 3;
  sconfig.lru_purge_enable = true;         // a reconnecting PC client replaces its dead socket
  sconfig.send_wait_timeout = 3;           // seconds: give up on a dead client quickly
  sconfig.recv_wait_timeout = 3;
  httpd_uri_t stream_uri = { .uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL };
  if (httpd_start(&stream_httpd, &sconfig) == ESP_OK) {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

// ----------------------------------------------------------------------------
// Wi-Fi with auto-reconnect
// ----------------------------------------------------------------------------
void onWiFiEvent(WiFiEvent_t event) {
  switch (event) {
    case ARDUINO_EVENT_WIFI_STA_GOT_IP:
      Serial.printf("[WiFi] connected, IP %s  ->  http://%s:81/stream\n",
                    WiFi.localIP().toString().c_str(), WiFi.localIP().toString().c_str());
      g_wifi_lost_since = 0;
      break;
    case ARDUINO_EVENT_WIFI_STA_DISCONNECTED:
      if (g_wifi_lost_since == 0) g_wifi_lost_since = millis();
      Serial.println("[WiFi] disconnected - auto-reconnecting...");
      break;
    default:
      break;
  }
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.setHostname(MDNS_NAME);
  WiFi.setSleep(false);                    // modem sleep adds 100+ ms latency spikes
  WiFi.setAutoReconnect(true);
  WiFi.persistent(false);
  WiFi.onEvent(onWiFiEvent);
  if (USE_STATIC_IP) {
    WiFi.config(STATIC_IP, GATEWAY, SUBNET, DNS1);
  }
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  WiFi.setTxPower(WIFI_POWER_19_5dBm);

  Serial.printf("[WiFi] connecting to '%s'", WIFI_SSID);
  unsigned long start = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - start < 20000) {
    delay(250);
    Serial.print(".");
    digitalWrite(LED_STATUS_GPIO, !digitalRead(LED_STATUS_GPIO));
  }
  Serial.println();
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[WiFi] not connected yet - will keep retrying in loop()");
  }
}

// ----------------------------------------------------------------------------
void setup() {
  WRITE_PERI_REG(RTC_CNTL_BROWN_OUT_REG, 0);   // avoid resets on current spikes at Wi-Fi TX
  Serial.begin(115200);
  Serial.setDebugOutput(false);
  Serial.println("\n=== PadelIQ ESP32-CAM ===");

  pinMode(LED_STATUS_GPIO, OUTPUT);
  digitalWrite(LED_STATUS_GPIO, HIGH);
  pinMode(LED_FLASH_GPIO, OUTPUT);
  digitalWrite(LED_FLASH_GPIO, LOW);

  if (!initCamera()) {
    Serial.println("[CAM] fatal: restarting in 3 s");
    delay(3000);
    ESP.restart();
  }

  connectWiFi();
  if (MDNS.begin(MDNS_NAME)) {
    MDNS.addService("http", "tcp", 80);
    Serial.printf("[mDNS] http://%s.local:81/stream\n", MDNS_NAME);
  }
  startServers();
  Serial.println("[HTTP] servers started (port 80 control, port 81 stream)");
}

void loop() {
  unsigned long now = millis();
  if (now - g_last_wifi_check > 5000) {
    g_last_wifi_check = now;
    if (WiFi.status() != WL_CONNECTED) {
      Serial.println("[WiFi] link down - reconnect()");
      WiFi.disconnect();
      WiFi.reconnect();
      // If the router stays unreachable for 2 minutes, a clean reboot is the most reliable fix.
      if (g_wifi_lost_since != 0 && now - g_wifi_lost_since > 120000) {
        Serial.println("[WiFi] offline for 2 min - restarting");
        delay(200);
        ESP.restart();
      }
    }
  }
  delay(50);
}
