# PadelIQ

Real-time padel analytics from a single camera: player detection and pose, court calibration
(10 m x 20 m homography), speed / distance / coverage per player, ball tracking, heatmaps and an AI
coach feed. It runs as a desktop OpenCV window or as a self-hosted web command center.

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
```

## Sources

| `--source`            | What it opens                                                                 |
|-----------------------|-------------------------------------------------------------------------------|
| `webcam`              | Best colour webcam: Logitech BRIO (or other named USB webcams) preferred, IR sensors skipped |
| `0`, `1`              | Webcam by index                                                               |
| `rtsp://...`          | IP camera                                                                     |
| `192.168.x.x` / URL   | ESP32-CAM MJPEG stream (`esp32_padeliq_cam/`)                                 |
| YouTube URL           | Video or live stream via `yt-dlp`                                             |
| `sim` / file path     | Local replay of a match video, looped (put an `.mp4` such as `clean_padel_match.mp4` in the project folder; videos are not in the repo) |
| `auto`                | ESP32 -> webcam -> local replay                                               |

## Low-latency pipeline

- **Latest-frame-only capture.** Every camera has a single frame slot with drop-oldest semantics and a
  condition variable. Capture, AI and render threads wake when a frame lands and always take the newest
  one. Nothing is queued, so latency stays flat. `CAP_PROP_BUFFERSIZE=1` is set on every device.
- **Webcam mode negotiation.** The engine tries DirectShow with `FOURCC=MJPG` set before width, height
  and FPS (1280x720 @ 60 requested). That mode is only accepted if the driver really switched to MJPG.
  Otherwise it tries MSMF at 1280x720, then any mode that opens. The winning mode is cached per camera in
  `webcam_profiles.json`, so the next start is fast.
- **Decoupled rendering.** The web stream and the desktop window render every camera frame at the
  camera's native rate (the web stream is capped by `--render-fps`, default 60). YOLO + pose
  (`--imgsz 512`) runs in a background worker on the newest frame. Player boxes are predicted with
  constant velocity between AI ticks.
- **MJPEG.** `/video.mjpg` is an async latest-frame generator, sent with `no-store` and
  `X-Accel-Buffering: no` headers at JPEG quality 78. A slow viewer skips to the newest frame instead of
  building a backlog.
- **Thread tuning.** The GIL switch interval is set to 2 ms, and torch uses `cores - 2` threads so
  capture, render and encode always have CPU.
- **Profiler.** The dashboard header shows `Capture fps | Render fps | AI ms | E2E ms`; the same numbers
  are available as `latency` in `GET /api/status`. E2E is measured from frame capture to the annotated
  JPEG being ready to send.

## Web API (selection)

`GET /video.mjpg` · `GET /api/status` · `GET /api/telemetry` · `WS /ws/telemetry` ·
`POST /api/source {"type": "webcam|sim|esp32|rtsp|youtube|auto|url", "value": "..."}` ·
`POST /api/calibration/auto` · `POST /api/calibration/confirm` ·
`GET /api/report.json|csv|html` · interactive docs at `/api/docs`.

## Desktop keys

`Enter` confirm court · `C` drag corners · `A` re-scan court · `R` reset stats · `S` record MP4 · `Q` quit

## Smoke test

```bat
python padeliq_live_stream.py --source sim --test-run 100
```
