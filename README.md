# PadelIQ

Real-time padel analytics from a single camera: player detection and pose, court calibration
(10 m x 20 m homography), speed / distance / coverage per player, ball tracking with shot counting,
heatmaps and an AI coach feed. It runs as a desktop OpenCV window or as a self-hosted web command center.
The camera can be a USB webcam (Logitech BRIO preferred) or **any phone, paired by scanning a QR code**.

## Quick start (Windows)

```bat
pip install -r requirements.txt
run_live.bat
```

or directly:

```bat
:: web command center -> http://localhost:8000
python padeliq_enterprise_server.py --source webcam --open-browser

:: desktop window
python padeliq_live_stream.py --source webcam
python padeliq_live_stream.py --source mobile      :: phone camera via QR code
```

## Sources

| `--source`            | What it opens                                                                 |
|-----------------------|-------------------------------------------------------------------------------|
| `webcam`              | Best colour webcam: Logitech BRIO (or other named USB webcams) preferred, IR sensors skipped |
| `mobile`              | Phone rear camera over the QR portal (`https://<LAN_IP>:8443/mobile-camera`) |
| `0`, `1`              | Webcam by index                                                               |
| `rtsp://...`          | IP camera                                                                     |
| `192.168.x.x` / URL   | ESP32-CAM MJPEG stream (`esp32_padeliq_cam/`)                                 |
| YouTube URL           | Video or live stream via `yt-dlp`                                             |
| `sim` / file path     | Local replay of a match video, looped (put an `.mp4` such as `clean_padel_match.mp4` in the project folder; videos are not in the repo) |
| `auto`                | ESP32 -> webcam -> local replay                                               |

## Mobile QR camera portal

1. Start either engine. The HTTPS portal starts on port 8443. A self-signed `certs/cert.pem` +
   `certs/key.pem` pair is generated on first run for the LAN IP. `certs/` is git-ignored.
2. Show the QR code:
   - **Web:** click **📱 Mobile QR Portal** in the top header bar (or press `M`).
   - **Desktop:** press `M` for the QR overlay, or start with `--source mobile` to fill the window with the
     pairing screen.
3. Scan it with the phone (same Wi-Fi). Accept the local certificate once (`getUserMedia` needs HTTPS on a
   LAN address), then tap **Start Camera**.
4. On the first phone frame, the engine swaps to the phone seamlessly. The current view stays live until
   the phone delivers frames, and the web modal closes itself.

The phone page:
- **Lens switch:** 📷 Rear Wide 1x / Ultra-Wide 0.5x.
- **Live badge:** `🟢 STREAMING LIVE TO STAGE | XX FPS | XX ms` (ack round-trip).
- **🎯 Auto-Detect Court.**
- **Four draggable corner handles:** fine-tune the court, then **Lock** it. Locking confirms the
  calibration and engages the stage lock. While the stage is locked, the phone cannot move the corners.

Transport details:
- **Frames:** binary `canvas.toBlob('image/jpeg', 0.72)` WebSocket messages on `/ws/mobile-cam`.
- **Flow control:** at most 2 frames in flight and `ws.bufferedAmount < 64 KB`, with a server ACK per
  decoded frame.
- **Weak Wi-Fi:** the phone drops to 540p after 2 s of slow ACKs and returns to 720p when they recover.

The QR uses the IP of the real Wi-Fi/Ethernet adapter. Virtual adapters (VMware, Hyper-V vEthernet, WSL,
VirtualBox, VPN) and link-local addresses are skipped by adapter name. Override with `--lan-ip` or
`PADELIQ_LAN_IP`. A phone on the same LAN can also type `https://<LAN_IP>:8443/mobile-camera` without the
token; the page then fetches it from `/api/mobile/pair`, which only answers private-network clients.

If Windows Firewall asks, allow Python on private networks so the phone can reach port 8443.

## Stage pipeline

- **OpenVINO FP16 inference.** Detection (`yolov8n_openvino_model`, static 288x480) and pose
  (`yolov8n-pose_openvino_model`) run through a direct OpenVINO runner (letterbox → infer → NumPy decode →
  NMS). This skips the Ultralytics predict wrapper. PyTorch is the automatic fallback (`--ai-backend`).
- **Court-ROI cropping.** Once the court is confirmed, detection runs on the court polygon's bounding box
  plus a 15 % margin.
- **Adaptive pose stride.** Each player gets a fresh pose every `--pose-stride` ticks (default 2).
  In between, keypoints are warped with the tracked box, then smoothed (keypoint EMA α 0.65).
- **Box smoothing.** One-Euro filter for boxes; EMA for speeds.
- **Cached HUD.** Panels, cards, minimap chrome and tags are pre-rendered and alpha-blended only inside
  their ROI (no full-frame `addWeighted`). Text values refresh at 10 Hz. Boxes, the ball and dots draw on
  every frame.
- **Shared JPEG.** The server encodes each frame once (quality 80, `IMWRITE_JPEG_OPTIMIZE=0`). Every
  MJPEG viewer shares those bytes. Viewers are woken by the render thread instead of polling, so 3 viewers
  get the same fps as 1.
- **Latest-frame-only capture.** Single frame slot with drop-oldest semantics, `CAP_PROP_BUFFERSIZE=1`,
  and webcam MJPG negotiation (DirectShow → MSMF) cached in `webcam_profiles.json`.
- **Camera exposure.**
  - `E` / the dashboard toggle switches between *Auto Exposure* and *Fast Sport Shutter* (manual
    exposure −5, less motion blur, max FPS). The camera keeps running.
  - In sport mode, gain and a gamma lift are applied only when the scene luma drops below 65.
  - `K` opens the driver dialog (set Power Line Frequency 50/60 Hz there for anti-flicker).

### Ball tracker

- **Detection cues (every camera frame):**
  - 3-frame motion mask.
  - HSV ball colour (H 20–55, S 65–255, V 110–255).
  - A bright blur-streak cue near the predicted position.
- **Exclusion:** player shoe and body detections are excluded using ankle keypoints. Fast candidates near
  the wrists are still allowed.
- **Track:** `cv2.KalmanFilter(4,2)` plus a parabola gate; candidate area 8–550 px; coasts up to 6 frames.
- **Speed:** court-plane km/h, clamped to 15–140 and smoothed with an EMA.
- **Shots:** counted from trajectory reversals near a wrist, with an impact ring.
- **HUD (drawn last):** neon ring, 18-frame cyan→gold comet, `BALL: XX km/h | Shots: N` tag, and a
  pulsing minimap dot with a vector line.

### Benchmarks (this laptop, CPU only)

| Metric                                   | Before          | After                      |
|------------------------------------------|-----------------|----------------------------|
| AI tick (detect + pose), median          | 56.6 ms         | 22.2 ms isolated · 28–38 ms in the live loop |
| HUD render, full / compact               | 15.1 / 3.8 ms   | 7.3 / 2.5 ms               |
| Web stream, 1 viewer                     | –               | 57–59 fps, E2E 24 ms median |
| Web stream, 3 viewers                    | 4.2 fps each    | 58.6 fps each              |
| Desktop display (replay)                 | –               | 51–57 fps, capture→display 27–29 ms |
| Ball tracker precision (40-crop sheet)   | ~40–45 %        | ~78 %                      |
| Phone → server ACK round-trip (localhost sim) | –          | 6–8 ms, JPEG decode 4.4 ms |

E2E = camera frame captured (or phone JPEG received) → annotated JPEG ready.

## Stage hotkeys

| Key   | Desktop window                                | Web dashboard                    |
|-------|-----------------------------------------------|----------------------------------|
| `C`   | Calibrate: drag court corners                 | Drag court corners               |
| `Enter` | Confirm the detected court                  | Confirm                          |
| `A`   | Re-scan the court                             | –                                |
| `L`   | Lock / unlock the calibration (geometry keys and scene-cut re-calibration disabled) | Lock button |
| `E`   | Auto exposure ↔ Fast Sport Shutter (USB webcams) | Shutter button                |
| `K`   | Camera driver settings (Power Line Frequency) | Camera settings button          |
| `F`   | Borderless fullscreen                         | (browser F11)                    |
| `F12` | Failover: live camera ↔ clean replay, swapped only once the target has frames | Failover button |
| `M`   | Mobile QR portal overlay                      | Mobile QR Portal modal           |
| `S`   | Start / stop MP4 recording                    | –                                |
| `R`   | Reset player stats                            | –                                |
| `Q`   | Quit                                          | –                                |

If a camera stalls for more than 1 s, the last frame is held with a small *Re-locking Camera…* badge
while the camera reconnects.

## Web API (selection)

- **Video and data:** `GET /video.mjpg` · `GET /api/status` · `GET /api/telemetry` · `WS /ws/telemetry`
- **Source:** `POST /api/source {"type": "webcam|mobile|sim|esp32|rtsp|youtube|auto|url", "value": "..."}`
- **Calibration:** `POST /api/calibration/auto` · `POST /api/calibration/confirm` ·
  `POST /api/calibration/lock {"locked": true}`
- **Camera:** `POST /api/camera/exposure {"mode": "auto|sport|toggle"}` · `POST /api/camera/settings` ·
  `POST /api/failover`
- **Mobile portal:** `GET /mobile-camera` · `GET /api/qr.png` · `GET /api/mobile/pair` (same-LAN token) · `GET /api/mobile/status` · `WS /ws/mobile-cam`
- **Reports:** `GET /api/report.json|csv|html`
- **Docs:** interactive docs at `/api/docs`

## Smoke tests

```bat
python padeliq_live_stream.py --source sim --test-run 100
python padeliq_live_stream.py --source sim --test-run 300 --test-stage-keys --test-failover
```
