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

### Orientation (sideways / portrait-locked phones)

Frames are never stretched.

- **On the phone.** A portrait camera buffer, which is what a portrait-locked phone gives when held sideways,
  is rotated in the canvas before JPEG encoding, so the stage receives a true 1280x720 landscape frame.
  **🔄 Rotate 90°** cycles the transmitted image through 0° → 90° → 180° → 270°. The choice is remembered on
  that phone.
- **In both engines.** Every camera frame is rotated (`0/90/180/270`) before any resize or HUD drawing. A
  frame that would still be portrait is uniformly scaled and pillarboxed onto 16:9, never squashed.
- **Automatic orientation.** On the first phone frame, and on `[A]`, a dedicated orientation score (court
  floor low and widening toward the camera, plus horizontal net/court lines; ~7 ms) checks the current
  rotation. Below 55 %, it tests 90° CW, 90° CCW and 180° and keeps the most upright one. The court is then
  confirmed and locked as soon as auto-calibration lands, without the 3 s review. The court AI's own
  confidence cannot be used for this, because a sideways court still scores ~87 %.
- **`[R]`.** Rotates by 90° and re-calibrates. It only auto-locks when the new orientation scores as
  upright, so a wrong manual rotation is never locked.
- **Desktop window.** The window is letterboxed to its own aspect ratio, so HUD fonts and the minimap stay
  undistorted at any window size or in fullscreen.

## Stage pipeline

- **OpenVINO FP16 inference.** Detection (`yolov8n_openvino_model`, static 288x480) and pose
  (`yolov8n-pose_openvino_model`) run through a direct OpenVINO runner (letterbox → infer → NumPy decode →
  NMS). This skips the Ultralytics predict wrapper. PyTorch is the automatic fallback (`--ai-backend`).
- **Court-ROI cropping.** Once the court is confirmed, detection runs on the court polygon's bounding box
  plus a 15 % margin. If fewer than 4 court players are found for 1.5 s, one full-frame pass runs every
  second, so a wrong or stale ROI cannot hide players for long. Calibration's own person pass is always
  full-frame.
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

### Elevated / balcony cameras (railings, wire mesh, background courts)

- **Foreground railing.** A wide dark horizontal band in the lower frame (balcony rail, hand rail) is
  treated as the end of the visible floor. Nothing below it can become the court, so the walkway strip
  under a railing is never selected. The court solvers, edge extraction and scoring treat the rows behind
  the rail as hidden, not as "off the floor".
- **Wire mesh and bars.** The floor mask is closed vertically (15x45, then 7x25) between 0.30 H and the
  railing, which bridges thin dark horizontal obstructions without reaching a background court at the top.
  Turf colour is sampled from a 5x5 median image, so wires never teach the colour model "dark turf".
  Floor behind a dense mesh is also classified on that median image.
- **Candidate gates.**
  - A court candidate must span at least 28 % of the frame height, and its centroid must not lie in the
    bottom 18 %.
  - The calibration person pass is full-frame. Each candidate earns a bonus for the share of player feet
    (y 18–80 %) that it contains.
  - Ankle keypoints sitting on the railing (feet hidden behind it) are ignored as colour seeds.
- **Players.**
  - A skeleton, when present, must look human: a shoulder above a hip, a plausible torso, legs below the
    hip, consistent shoulder and hip widths.
  - Without a skeleton, the box h/w must be 1.15–4.2 (h/w above 4.2 is always rejected).
  - A box that extends below the railing's bottom edge stands in front of it (a balcony post, or a
    spectator on the walkway) and is rejected.
  - Feet: x −1.2…11.2 m (side openings, foot on the turf); lengthwise up to the back glass (−0.95 m for
    upright players). Nothing behind the back glass is tracked.

### Player slots and teams

**Team A** is the near half (camera side, court y ≥ 10 m): **P1** near-left, **P2** near-right.
**Team B** is the far half: **P3** far-left, **P4** far-right. Left and right seed the first assignment.
Hungarian matching on court positions (velocity-gated) then keeps each identity, and a player is never
assigned across the net.

### Live score (`padeliq_score.py`)

- **Display.** Official padel scoring: 0 → 15 → 30 → 40 → GAME, with golden point at 40 : 40, plus games
  and total points per team. A broadcast banner sits top-centre on the desktop HUD and in the web stream:
  `TEAM A [P1-P2]  15 : 30  TEAM B [P3-P4]`, then `Games | Points | Rally`. A gold
  `🏆 POINT TEAM X!` badge shows for 2.6 s after each point.
- **Automatic points (heuristic, single camera).**
  - A rally starts at the first shot or the first confirmed net crossing.
  - The team whose half the ball enters must play it. A shot hands it to the other team. A ball that
    stays on the hitter's side (the net) returns it to the hitter.
  - If that team does not return the ball within 2.2 s (ball stopped, lost, out or in the net), the other
    team scores.
  - After each point there is a 3.5 s cooldown.
- **Manual override, always available:** `1` / `2` add a point for Team A / B, `0` resets. On the
  dashboard: **+1 Team A [1]**, **+1 Team B [2]**, **Undo**, **Auto points** (on/off) and
  **Reset score [0]**. Undo is web-only.
- **Ball recall matters.** With low ball visibility (e.g. a phone over Wi-Fi), automatic points are less
  reliable, so keep the manual keys at hand.

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
| `A`   | Orientation check + re-scan the court         | Auto-calibrate                   |
| `L`   | Lock / unlock the calibration (geometry keys and scene-cut re-calibration disabled) | Lock button |
| `E`   | Auto exposure ↔ Fast Sport Shutter (USB webcams) | Shutter button                |
| `K`   | Camera driver settings (Power Line Frequency) | Camera settings button          |
| `F`   | Borderless fullscreen                         | (browser F11)                    |
| `F12` | Failover: live camera ↔ clean replay, swapped only once the target has frames | Failover button |
| `M`   | Mobile QR portal overlay                      | Mobile QR Portal modal           |
| `R`   | Rotate the camera feed 90° clockwise + re-calibrate (always allowed; releases and re-takes the lock) | 🔄 Rotate [R] button |
| `1` / `2` | +1 point Team A (P1·P2) / Team B (P3·P4) | +1 Team A [1] / +1 Team B [2]     |
| `0`   | Reset the score (0 : 0, games 0-0)            | Reset score [0]                  |
| `S`   | Start / stop MP4 recording                    | –                                |
| `X`   | Reset player stats                            | –                                |
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
- **Score:** `GET /api/score` · `POST /api/score/point {"team": "A|B"}` · `POST /api/score/undo` ·
  `POST /api/score/reset` · `POST /api/score/auto {"enabled": true}` (also in `/api/telemetry` → `score`)
- **Mobile portal:** `GET /mobile-camera` · `GET /api/qr.png` · `GET /api/mobile/pair` (same-LAN token) · `GET /api/mobile/status` · `WS /ws/mobile-cam`
- **Reports:** `GET /api/report.json|csv|html`
- **Docs:** interactive docs at `/api/docs`

## Smoke tests

```bat
python padeliq_live_stream.py --source sim --test-run 100
python padeliq_live_stream.py --source sim --test-run 300 --test-stage-keys --test-failover
python padeliq_live_stream.py --source sim --test-run 320 --test-score
```

`--test-score` presses `1`, `1`, `2`, checks the score, and saves `scoreboard_verified.jpg`.
