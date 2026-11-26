# -*- coding: utf-8 -*-
"""
PadelIQ Real-Time Live Camera Engine
====================================

Sources supported:
  * ESP32-CAM MJPEG streams   (http://<ip>:81/stream)
  * RTSP / HTTP IP cameras    (rtsp://user:pass@host:554/...)
  * USB / built-in webcams    (0, 1, ... — DirectShow fallback on Windows)
  * YouTube videos / lives    (https://www.youtube.com/watch?v=... resolved with `yt-dlp -g`)
  * Local video files         (looped at native FPS = "simulated live stream")
  * auto (default)            ESP32 / PADELIQ_STREAM_URL -> webcam 0/1 -> local .mp4

Keys while streaming:
  [C] calibrate the 4 court floor corners   [R] reset stats
  [S] start/stop MP4 recording              [Q]/[Esc] quit

Smoke test (headless, exits 0 on success):
  python padeliq_live_stream.py --source sim --test-run 100

Automatic court calibration from the court lines (saved per source type):
  python padeliq_live_stream.py --source sim --auto-calibrate
"""
import os

# Must be set before cv2 is imported: quieter logs + much faster MSMF webcam open on Windows.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS", "0")
# RTSP over TCP is far more reliable than UDP on Wi-Fi networks.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import sys
import glob
import json
import math
import time
import socket
import shutil
import subprocess
import threading
import argparse
import traceback
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import court_geometry_ai as CG
import padeliq_orientation as ORI

# Never crash on console encoding (Windows cp1252 consoles, redirected output).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# =====================================================================
# 1. Court standard, file locations and HUD colours
# =====================================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))

COURT_WIDTH_M = 10.0
COURT_LENGTH_M = 20.0
NET_Y_M = COURT_LENGTH_M / 2.0
SERVICE_LINE_OFFSET_M = 6.95                       # from the net, per padel rules
SERVICE_LINE_NEAR_M = NET_Y_M - SERVICE_LINE_OFFSET_M   # 3.05 m
SERVICE_LINE_FAR_M = NET_Y_M + SERVICE_LINE_OFFSET_M    # 16.95 m

LIVE_CALIBRATION_FILE = os.path.join(APP_DIR, "live_court_calibration.json")
SIM_CALIBRATION_FILE = os.path.join(APP_DIR, "sim_court_calibration.json")
WEBCAM_SOURCE_NAMES = ("webcam", "brio", "camera", "usb", "cam")
MOBILE_CALIBRATION_FILE = os.path.join(APP_DIR, "mobile_court_calibration.json")
MOBILE_SOURCE_NAMES = ("mobile", "phone", "qr", "mobile-qr", "mobile_qr")
# Ranked name hints for the external colour webcam (DirectShow friendly names). IR / depth sensors are skipped.
PREFERRED_CAMERA_HINTS = ("brio", "logitech", "c920", "c922", "c930", "streamcam", "elgato", "obsbot", "insta360")
IR_CAMERA_HINTS = (" ir ", "infrared", "ir camera", "depth", "windows hello")
LEGACY_VIDEO_CALIBRATION_FILE = os.path.join(APP_DIR, "court_calibration.json")
VERIFY_SCREENSHOT = os.path.join(APP_DIR, "live_hud_verified.jpg")
PREFERRED_SIM_VIDEOS = ("clean_padel_match.mp4", "padel_input.mp4")

KCONF = CG.KCONF                # pose keypoint confidence threshold
STALL_TIMEOUT_S = 2.5           # watchdog: no frames for this long -> reconnect
WEBCAM_PROFILE_FILE = os.path.join(APP_DIR, "webcam_profiles.json")   # negotiated mode per webcam index
DEFAULT_CAM_SIZE = (1280, 720)
DEFAULT_CAM_FPS = 60            # request the max; the driver clamps to what the sensor/USB mode delivers
DEFAULT_IMGSZ = 480             # YOLO input for the PyTorch fallback (the OpenVINO export is a fixed 288x480 = 16:9)
WEBCAM_STALL_S = 1.0            # USB camera: no frame for 1 s -> hold the last frame and re-initialise in the background
EXPOSURE_MODES = ("auto", "sport")
EXPOSURE_LABELS = {"auto": "Auto Exposure", "sport": "Fast Sport Shutter (Max FPS)"}
SPORT_EXPOSURE = -5             # DirectShow/MSMF log2 seconds: -5 = 1/32 s (-6 = 1/64 s for 60 fps in bright halls)
DARK_LUMA = 65                  # sport mode: lift gain (then software gamma) while the mean frame luminance is below this
F12_KEYS = (0x7B0000, 65481)    # cv2.waitKeyEx code of F12 on Windows / GTK
MAX_PLAYER_SPEED_MS = 8.5       # physical cap for sprinting padel players
HEAT_CELL_M = 0.5               # heatmap resolution (20 x 40 cells over the 10 x 20 m court)
FOOT_PIXEL_SIGMA = 10.0         # foot-point jitter (px; far feet are often cut by the net band) -> metres via local homography scale
KF_ACCEL_SIGMA = 2.0            # m/s^2 process noise of the constant-velocity Kalman filter
MOVE_SPEED_MS = 0.45            # below this filtered speed a player counts as standing (no distance)
BASE_HUD_HEIGHT = 720.0         # HUD is designed at 720p and scaled from there

SLOT_COLORS = {
    1: (0, 110, 255),    # P1 orange
    2: (50, 220, 50),    # P2 green
    3: (0, 220, 220),    # P3 yellow
    4: (180, 105, 255),  # P4 pink/purple
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def log(msg):
    print(f"[PadelIQ {datetime.now():%H:%M:%S}] {msg}", flush=True)


# =====================================================================
# 2. Source resolution (auto / sim / webcam / ESP32 / RTSP / file)
# =====================================================================
@dataclass
class SourceSpec:
    source: object                 # int (webcam index) or str (URL / file path)
    kind: str                      # "webcam" | "stream" | "file" | "youtube" | "mobile" (QR-paired phone)
    label: str = ""
    api_candidates: list = field(default_factory=lambda: [cv2.CAP_ANY])
    page_url: str = ""             # YouTube page URL (re-resolved with yt-dlp on reconnect)
    is_live: bool = True
    fps_hint: float = 0.0
    webcam_profile: dict = None    # negotiated webcam mode (backend / MJPG / size), reused on reconnect
    device_name: str = ""          # DirectShow friendly name (e.g. "Logitech BRIO") when known


STREAM_SCHEMES = ("http://", "https://", "rtsp://", "rtsps://", "rtmp://", "udp://", "tcp://")


def webcam_backends(preference="auto"):
    """Default backend first, DirectShow fallback on Windows (as requested)."""
    pref = (preference or "auto").lower()
    if pref == "dshow":
        return [cv2.CAP_DSHOW]
    if pref == "msmf":
        return [cv2.CAP_MSMF]
    if pref == "any":
        return [cv2.CAP_ANY]
    if os.name == "nt":
        # DirectShow first: it is the only Windows backend where FOURCC=MJPG is honoured (Logitech BRIO & co.);
        # MSMF next (negotiates compressed modes internally).
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    return [cv2.CAP_ANY]


def normalize_esp32_url(value):
    """'192.168.1.50' -> 'http://192.168.1.50:81/stream'. Full URLs pass through."""
    value = (value or "").strip().strip('"')
    if not value:
        return ""
    if value.lower().startswith(STREAM_SCHEMES):
        return value
    host = value.split("/")[0]
    if ":" not in host:
        host += ":81"
    return f"http://{host}/stream"


def is_url_reachable(url, timeout=1.5):
    """Cheap TCP probe of the stream host:port (no full HTTP/RTSP handshake)."""
    try:
        u = urlparse(url)
        default_ports = {"http": 80, "https": 443, "rtsp": 554, "rtsps": 322, "rtmp": 1935}
        port = u.port or default_ports.get(u.scheme, 80)
        if not u.hostname:
            return False
        with socket.create_connection((u.hostname, port), timeout=timeout):
            return True
    except Exception:
        return False


YOUTUBE_HOSTS = ("youtube.com", "youtu.be", "youtube-nocookie.com")
# H.264 first (decodes everywhere), never AV1 (often unsupported by OpenCV's FFmpeg), <= 720p for CPU budget.
YOUTUBE_FORMAT = "bv*[height<=720][vcodec^=avc1]/bv*[height<=720][vcodec!^=av01]/b[height<=720]/bv*/b"


def is_youtube_url(value):
    try:
        host = (urlparse(str(value).strip()).hostname or "").lower()
        return any(host == h or host.endswith("." + h) for h in YOUTUBE_HOSTS)
    except Exception:
        return False


def ytdlp_base_command():
    exe = shutil.which("yt-dlp")
    return [exe] if exe else [sys.executable, "-m", "yt_dlp"]


def resolve_youtube(page_url, timeout=90):
    """
    Resolve a YouTube video / live URL to a direct media URL with `yt-dlp -g`.
    Returns (direct_url, is_live, fps). Raises RuntimeError with a readable reason.
    """
    base = ytdlp_base_command()
    args = ["--no-warnings", "--no-playlist", "-f", YOUTUBE_FORMAT,
            "-O", "%(is_live)s|%(fps)s|%(width)sx%(height)s", "-g", page_url]
    attempts = []
    if shutil.which("node") or shutil.which("deno"):
        rt = "node" if shutil.which("node") else "deno"
        attempts.append(base + ["--js-runtimes", rt] + args)
    attempts.append(base + args)
    last_err = "yt-dlp is not installed (pip install yt-dlp)"
    for cmd in attempts:
        try:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                                 encoding="utf-8", errors="replace", creationflags=flags)
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            last_err = "yt-dlp timed out"
            continue
        lines = [ln.strip() for ln in (res.stdout or "").splitlines() if ln.strip()]
        urls = [ln for ln in lines if ln.startswith("http")]
        meta = next((ln for ln in lines if "|" in ln and not ln.startswith("http")), "")
        if res.returncode == 0 and urls:
            parts = (meta.split("|") + ["", "", ""])[:3]
            is_live = parts[0].strip().lower() == "true"
            try:
                fps = float(parts[1])
            except ValueError:
                fps = 0.0
            return urls[0], is_live, fps
        err_lines = [ln for ln in (res.stderr or "").splitlines() if ln.strip()]
        last_err = err_lines[-1] if err_lines else f"yt-dlp exit code {res.returncode}"
        if "no such option" not in last_err.lower():
            break
    raise RuntimeError(last_err)


def find_simulation_video(explicit=None):
    if explicit:
        path = explicit if os.path.isabs(explicit) else os.path.join(APP_DIR, explicit)
        if os.path.isfile(path):
            return path
        if os.path.isfile(explicit):
            return os.path.abspath(explicit)
        log(f"WARNING: --video '{explicit}' not found, searching the project folder instead.")
    for name in PREFERRED_SIM_VIDEOS:
        preferred = os.path.join(APP_DIR, name)
        if os.path.isfile(preferred):
            return preferred
    for path in sorted(glob.glob(os.path.join(APP_DIR, "*.mp4"))):
        name = os.path.basename(path).lower()
        if "recording" in name or "__tmp__" in name:
            continue
        cap = cv2.VideoCapture(path)
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if ok:
            return path
    return None


def open_capture(source, api, timeout_s, params=None, require_frame=True, configure=None):
    """
    Open cv2.VideoCapture in a helper thread so a hanging driver / dead host can
    never freeze the caller. Returns (cap, first_frame) or (None, None).
    """
    result = {}

    def worker():
        cap = None
        try:
            cap = cv2.VideoCapture(source, api, params) if params else cv2.VideoCapture(source, api)
            if cap is not None and cap.isOpened():
                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # never let the driver queue stale frames
                except Exception:
                    pass
                if configure is not None:
                    configure(cap)                            # FOURCC/size/FPS must be set before the first read
                if require_frame:
                    frame = None
                    for _ in range(5):          # some webcams return a few empty frames first
                        ok, frame = cap.read()
                        if ok and frame is not None and frame.size > 0:
                            result["frame"] = frame
                            break
                        time.sleep(0.05)
        except Exception as exc:
            result["error"] = exc
        result["cap"] = cap

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        # Abandon it; release the handle in the background once the driver returns.
        def reaper():
            t.join()
            c = result.get("cap")
            if c is not None:
                c.release()
        threading.Thread(target=reaper, daemon=True).start()
        return None, None

    cap = result.get("cap")
    if cap is None or not cap.isOpened() or (require_frame and "frame" not in result):
        if cap is not None:
            cap.release()
        return None, None
    return cap, result.get("frame")


def fourcc_str(cap):
    v = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    s = "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))
    return s if s.isprintable() and s.strip() else f"0x{v:08x}"


def _load_webcam_profiles():
    try:
        with open(WEBCAM_PROFILE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_webcam_profile(index, profile):
    data = _load_webcam_profiles()
    data[str(index)] = profile
    try:
        with open(WEBCAM_PROFILE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def webcam_configurer(width, height, fps, mjpg=True):
    """Order matters on Windows: FOURCC=MJPG *before* width/height/FPS, otherwise DirectShow keeps raw YUY2
    (5-10 fps at 720p and a saturated USB bus)."""
    def configure(cap):
        if mjpg:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        cap.set(cv2.CAP_PROP_FPS, float(fps))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return configure


def list_dshow_cameras():
    """DirectShow video-input names in the same order as OpenCV CAP_DSHOW indices (Windows + pygrabber).
    Returns [] when unavailable - callers then fall back to probing indices."""
    if os.name != "nt":
        return []
    try:
        from pygrabber.dshow_graph import FilterGraph
        return [str(n) for n in FilterGraph().get_input_devices()]
    except Exception:
        return []


def _is_ir_name(name):
    n = f" {name.lower()} "
    return any(k in n for k in IR_CAMERA_HINTS)


def _frame_is_colour(frame):
    """IR / monochrome sensors deliver R == G == B."""
    if frame is None or frame.ndim < 3 or frame.shape[2] < 3:
        return False
    sm = frame[::8, ::8].astype(np.int16)
    return float(np.mean(np.abs(sm[..., 0] - sm[..., 1])) + np.mean(np.abs(sm[..., 1] - sm[..., 2]))) > 3.0


def find_best_webcam(backends=None, max_probe=4):
    """-> (index, name, api_candidates) of the best colour webcam, or (None, None, None).
    Order: names matching PREFERRED_CAMERA_HINTS (Logitech BRIO first), then other cameras; IR sensors skipped
    by name and by a monochrome-frame check. With several cameras only DirectShow is used, because its index
    order is the one pygrabber reports (MSMF may number devices differently)."""
    backends = list(backends or webcam_backends("auto"))
    names = list_dshow_cameras()
    if names:
        def rank(i):
            low = names[i].lower()
            pref = next((k for k, h in enumerate(PREFERRED_CAMERA_HINTS) if h in low), len(PREFERRED_CAMERA_HINTS))
            return (_is_ir_name(names[i]), pref, i)
        order = sorted(range(len(names)), key=rank)
        log("Cameras: " + ", ".join(f"[{i}] {n}" for i, n in enumerate(names)))
    else:
        order = list(range(max_probe))
    multi = len(names) > 1
    probe_api = cv2.CAP_DSHOW if os.name == "nt" else backends[0]
    for idx in order:
        name = names[idx] if idx < len(names) else f"camera {idx}"
        if names and _is_ir_name(name):
            log(f"Skipping IR/depth sensor [{idx}] {name}")
            continue
        cap, frame = open_capture(idx, probe_api, timeout_s=6.0)
        if cap is None:
            continue
        cap.release()
        if not _frame_is_colour(frame):
            log(f"Skipping monochrome/IR camera [{idx}] {name}")
            continue
        apis = [cv2.CAP_DSHOW] if (multi and os.name == "nt") else backends
        log(f"Selected webcam [{idx}] {name}")
        return idx, name, apis
    return None, None, None


def probe_webcam(index, backends, timeout_s=4.0):
    for api in backends:
        cap, frame = open_capture(index, api, timeout_s)
        if cap is not None:
            cap.release()
            return api
    return None


def resolve_source(source_arg, esp32_ip=None, video=None, backend="auto"):
    """Turn the --source argument into a SourceSpec. Returns None if nothing usable."""
    s = (str(source_arg) if source_arg is not None else "auto").strip().strip('"')
    low = s.lower()
    backends = webcam_backends(backend)

    def file_spec():
        path = find_simulation_video(video)
        if not path:
            return None
        return SourceSpec(path, "file", f"SIM LOOP: {os.path.basename(path)}")

    if low == "auto":
        candidates = []
        if esp32_ip:
            candidates.append(("ESP32-CAM", normalize_esp32_url(esp32_ip)))
        env_url = os.environ.get("PADELIQ_STREAM_URL", "").strip()
        if env_url:
            candidates.append(("PADELIQ_STREAM_URL", normalize_esp32_url(env_url)))
        for name, url in candidates:
            log(f"[auto] probing {name}: {url}")
            if is_youtube_url(url):
                return SourceSpec(url, "youtube", f"YOUTUBE: {url}", page_url=url)
            if is_url_reachable(url):
                log(f"[auto] -> using network stream {url}")
                return SourceSpec(url, "stream", f"{name}: {url}")
            log(f"[auto]    {name} not reachable.")
        log("[auto] looking for a colour webcam (Logitech BRIO preferred) ...")
        idx, name, apis = find_best_webcam(backends)
        if idx is not None:
            log(f"[auto] -> using webcam [{idx}] {name}")
            return SourceSpec(idx, "webcam", f"WEBCAM {idx}: {name}", apis, device_name=name)
        log("[auto] no camera attached -> falling back to the simulated live stream (local .mp4)")
        return file_spec()

    if low in ("sim", "simulate", "simulated", "test", "file", "video"):
        return file_spec()
    if low in MOBILE_SOURCE_NAMES:
        # Frames are pushed by the phone over WSS (padeliq_mobile_bridge.MobileCamera) - nothing to open here.
        return SourceSpec("mobile", "mobile", "MOBILE QR CAMERA")
    if low in WEBCAM_SOURCE_NAMES:
        idx, name, apis = find_best_webcam(backends)
        if idx is None:
            log("No colour webcam found.")
            return None
        return SourceSpec(idx, "webcam", f"WEBCAM {idx}: {name}", apis, device_name=name)
    if s.isdigit():
        return SourceSpec(int(s), "webcam", f"WEBCAM {s}", backends)
    if is_youtube_url(s):
        return SourceSpec(s, "youtube", f"YOUTUBE: {s}", page_url=s)
    if low.startswith(STREAM_SCHEMES):
        return SourceSpec(s, "stream", _redact_url(s))
    for candidate in (s, os.path.join(APP_DIR, s)):
        if os.path.isfile(candidate):
            return SourceSpec(os.path.abspath(candidate), "file", f"FILE LOOP: {os.path.basename(candidate)}")
    # Bare IP / host[:port] => assume ESP32-CAM
    host = s.split("/")[0].split(":")[0]
    if host.replace(".", "").isdigit() and host.count(".") == 3:
        url = normalize_esp32_url(s)
        return SourceSpec(url, "stream", f"ESP32-CAM: {url}")
    return None


def _redact_url(url):
    try:
        u = urlparse(url)
        if u.password:
            return url.replace(f":{u.password}@", ":****@")
    except Exception:
        pass
    return url


def api_name(api):
    names = {cv2.CAP_ANY: "default", cv2.CAP_DSHOW: "DirectShow", cv2.CAP_MSMF: "MSMF", cv2.CAP_FFMPEG: "FFMPEG"}
    return names.get(api, str(api))


# =====================================================================
# 3. Zero-latency threaded camera with auto-reconnect watchdog
# =====================================================================
class ThreadedLiveCamera:
    """
    Continuously reads the newest frame in the background so the buffer never fills.

    * `_update()` is the supervisor / watchdog: it (re)connects, starts a reader for
      each connection "generation", and if no frame arrives for `stall_timeout`
      seconds it abandons that generation and re-opens cv2.VideoCapture(self.source).
      A reader stuck inside a blocking cap.read() (dead ESP32 / RTSP host) cannot
      freeze anything: it simply notices it is stale when the call returns and
      releases its own handle.
    * Webcams: default backend first, cv2.CAP_DSHOW fallback (Windows) if the default
      fails or takes too long.
    * Files: played at native FPS and looped seamlessly (simulated live stream).
    """

    def __init__(self, source, stall_timeout=STALL_TIMEOUT_S, open_timeout=6.0,
                 cam_width=None, cam_height=None, cam_fps=None, exposure="auto", sport_exposure=SPORT_EXPOSURE):
        if isinstance(source, SourceSpec):
            spec = source
        else:
            spec = resolve_source(source)
            if spec is None:
                raise ValueError(f"Unrecognised video source: {source!r}")
        self.spec = spec
        self.source = spec.source
        self.kind = spec.kind
        self.stall_timeout = float(stall_timeout)
        if self.kind == "webcam":
            self.stall_timeout = min(self.stall_timeout, WEBCAM_STALL_S)
        self.open_timeout = float(open_timeout)
        # Exposure control (webcams). Applied by the reader thread right after (re)connecting and on every toggle.
        self.exposure_mode = exposure if exposure in EXPOSURE_MODES else "auto"
        self.sport_exposure = float(sport_exposure)
        self.exposure_info = {"mode": self.exposure_mode, "label": EXPOSURE_LABELS[self.exposure_mode]}
        self._exp_dirty = True
        self._exp_saved = None
        self._gain_maxed = False
        self._gamma_lut = None
        self._last_luma_t = 0.0
        self._settings_req = False
        self.cam_width, self.cam_height = cam_width, cam_height
        self.cam_fps = cam_fps
        self.mode = ""                      # negotiated webcam mode, e.g. "DirectShow MJPG 1280x720@60"

        # Single-slot "latest frame" (no queue): publish() overwrites, consumers wake on the condition.
        self.lock = threading.Condition()
        self.frame = None
        self.ret = False
        self.frame_id = 0
        self.last_frame_time = 0.0
        self.status = "connecting"          # connecting | live | reconnecting
        self.reconnect_count = 0
        self.active_api = None
        self.native_fps = 0.0
        self.last_error = ""

        self._gen = 0
        self._last_ok = time.time()
        self.running = True
        self.thread = threading.Thread(target=self._update, name="PadelIQCamera", daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------ connect
    def _connect(self):
        """Blocking (with timeouts) attempt to open the source. Returns (cap, first_frame)."""
        if self.kind == "webcam":
            return self._connect_webcam()

        if self.kind == "youtube":
            try:
                log(f"Resolving YouTube stream with yt-dlp -g: {self.spec.page_url}")
                url, is_live, fps = resolve_youtube(self.spec.page_url)
            except Exception as exc:
                log(f"YouTube resolve failed: {exc}")
                self.last_error = f"yt-dlp: {exc}"
                return None, None
            self.spec.is_live = is_live
            self.source = url
            self.native_fps = fps if 1.0 <= fps <= 240.0 else 30.0
            log(f"YouTube {'LIVE' if is_live else 'video'} resolved ({self.native_fps:.0f} fps)")

        if self.kind in ("stream", "youtube"):
            params = []
            for name, val in (("CAP_PROP_OPEN_TIMEOUT_MSEC", 5000), ("CAP_PROP_READ_TIMEOUT_MSEC", 3000)):
                prop = getattr(cv2, name, None)
                if prop is not None:
                    params += [prop, val]
            cap, frame = open_capture(self.source, cv2.CAP_FFMPEG, self.open_timeout + 2.0, params or None)
            if cap is None:
                cap, frame = open_capture(self.source, cv2.CAP_ANY, self.open_timeout)
            if cap is None:
                self.last_error = "could not open stream"
            return cap, frame

        # file
        cap, frame = open_capture(self.source, cv2.CAP_ANY, self.open_timeout)
        if cap is not None:
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            self.native_fps = fps if 1.0 <= fps <= 240.0 else 30.0
        return cap, frame

    def _connect_webcam(self):
        """Negotiate the fastest mode: DirectShow + MJPG (accepted only if the driver really switched to MJPG),
        then MSMF at the requested size, then whatever opens. The winner is cached per webcam index."""
        w = int(self.cam_width or DEFAULT_CAM_SIZE[0])
        h = int(self.cam_height or DEFAULT_CAM_SIZE[1])
        fps = float(self.cam_fps or DEFAULT_CAM_FPS)
        key = f"{self.source}|{self.spec.device_name}" if self.spec.device_name else str(self.source)
        cached = getattr(self.spec, "webcam_profile", None) or _load_webcam_profiles().get(key)
        attempts = []
        if cached and cached.get("api") in (self.spec.api_candidates + [cv2.CAP_ANY]) \
                and cached.get("size") == [w, h]:
            attempts.append((int(cached["api"]), bool(cached.get("mjpg", True)), "any"))
        for api in self.spec.api_candidates:
            attempts.append((api, True, "mjpg" if api == cv2.CAP_DSHOW else "size"))
        attempts.append((self.spec.api_candidates[0], False, "any"))
        seen = set()
        for api, mjpg, accept in attempts:
            if not self.running:
                return None, None
            if (api, mjpg, accept) in seen:
                continue
            seen.add((api, mjpg, accept))
            t0 = time.time()
            cap, frame = open_capture(self.source, api, timeout_s=max(self.open_timeout, 10.0),
                                      configure=webcam_configurer(w, h, fps, mjpg))
            if cap is None:
                log(f"Webcam {self.source}: {api_name(api)}{' MJPG' if mjpg else ''} failed or timed out.")
                continue
            fcc = fourcc_str(cap)
            fh, fw = frame.shape[:2]
            got_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            ok = (accept == "any" or (accept == "mjpg" and fcc == "MJPG" and (fw, fh) == (w, h))
                  or (accept == "size" and (fw, fh) == (w, h)))
            if not ok:
                log(f"Webcam {self.source}: {api_name(api)} gave {fw}x{fh} {fcc} (no MJPG {w}x{h}) - trying next mode")
                cap.release()
                continue
            self.mode = f"{api_name(api)} {fcc} {fw}x{fh}@{got_fps:.0f}"
            self.native_fps = got_fps if 1.0 <= got_fps <= 240.0 else 30.0
            if self.active_api != api or not cached:
                log(f"Webcam {self.source} opened: {self.mode} (requested MJPG {w}x{h}@{fps:.0f}, "
                    f"{time.time() - t0:.1f}s)")
            self.active_api = api
            self.spec.api_candidates = [api] + [a for a in self.spec.api_candidates if a != api]
            profile = {"api": int(api), "mjpg": bool(mjpg), "size": [w, h], "mode": self.mode}
            self.spec.webcam_profile = profile
            if profile != cached:
                save_webcam_profile(key, profile)
            self._exp_dirty = True                        # re-apply the exposure mode on every (re)connect
            return cap, frame
        return None, None

    # ---------------------------------------------------------------- exposure
    def set_exposure_mode(self, mode):
        """'auto' | 'sport' - switched live by the reader thread, no camera restart."""
        if mode not in EXPOSURE_MODES:
            raise ValueError(f"exposure mode must be one of {EXPOSURE_MODES}")
        self.exposure_mode = mode
        self._exp_dirty = True
        return mode

    def toggle_exposure(self):
        return self.set_exposure_mode("sport" if self.exposure_mode == "auto" else "auto")

    def open_driver_settings(self):
        """DirectShow property sheet of the camera (Power Line Frequency 50/60 Hz, white balance, ...)."""
        self._settings_req = True

    @property
    def exposure_label(self):
        return EXPOSURE_LABELS[self.exposure_mode]

    def _auto_exposure_values(self):
        """(manual, auto) values of CAP_PROP_AUTO_EXPOSURE for the active backend."""
        if self.active_api == getattr(cv2, "CAP_V4L2", -1):
            return 1.0, 3.0                               # V4L2_EXPOSURE_MANUAL / APERTURE_PRIORITY
        return 0.0, 1.0                                   # DirectShow / MSMF: 1 = auto, anything else = manual

    def _apply_exposure(self, cap):
        P = cv2
        manual, auto = self._auto_exposure_values()
        res = {}
        try:
            if self.exposure_mode == "sport":
                if self._exp_saved is None:
                    self._exp_saved = {"gain": cap.get(P.CAP_PROP_GAIN), "backlight": cap.get(P.CAP_PROP_BACKLIGHT),
                                       "exposure": cap.get(P.CAP_PROP_EXPOSURE)}
                res["auto_exposure_off"] = bool(cap.set(P.CAP_PROP_AUTO_EXPOSURE, manual))
                res["exposure"] = bool(cap.set(P.CAP_PROP_EXPOSURE, self.sport_exposure))
                res["backlight_off"] = bool(cap.set(P.CAP_PROP_BACKLIGHT, 0))
                self._gain_maxed = False
                self._last_luma_t = 0.0
            else:
                res["auto_exposure_on"] = bool(cap.set(P.CAP_PROP_AUTO_EXPOSURE, auto))
                sv = self._exp_saved or {}
                for k, prop in (("gain", P.CAP_PROP_GAIN), ("backlight", P.CAP_PROP_BACKLIGHT)):
                    if sv.get(k) not in (None, -1.0):
                        res[f"{k}_restored"] = bool(cap.set(prop, sv[k]))
                self._gamma_lut = None
        except cv2.error as exc:
            res["error"] = str(exc)
        self.exposure_info = {"mode": self.exposure_mode, "label": self.exposure_label, "applied": res,
                              "exposure": round(float(cap.get(P.CAP_PROP_EXPOSURE)), 2),
                              "gain": round(float(cap.get(P.CAP_PROP_GAIN)), 2), "software_gamma": False}
        log(f"Camera exposure -> {self.exposure_label} ("
            + ", ".join(f"{k}={'ok' if v is True else ('unsupported' if v is False else v)}" for k, v in res.items()) + ")")

    def _auto_gain(self, cap, frame):
        """Sport shutter in a dark hall: step the hardware gain up; if the driver refuses, lift with a gamma LUT."""
        sm = frame[::16, ::16]
        luma = float(0.114 * sm[..., 0].mean() + 0.587 * sm[..., 1].mean() + 0.299 * sm[..., 2].mean())
        self.exposure_info["luma"] = round(luma, 1)
        if luma < DARK_LUMA:
            if not self._gain_maxed:
                g = cap.get(cv2.CAP_PROP_GAIN)
                ok = cap.set(cv2.CAP_PROP_GAIN, g + 16)
                g2 = cap.get(cv2.CAP_PROP_GAIN)
                if ok and g2 > g:
                    self.exposure_info["gain"] = round(float(g2), 2)
                    return
                self._gain_maxed = True
            if self._gamma_lut is None:
                self._gamma_lut = np.array([((i / 255.0) ** (1 / 1.6)) * 255 for i in range(256)], np.uint8)
                self.exposure_info["software_gamma"] = True
                log(f"Sport shutter: frame luminance {luma:.0f} < {DARK_LUMA} and no more hardware gain -> software gamma lift")
        elif luma > DARK_LUMA + 45 and self._gamma_lut is not None:
            self._gamma_lut = None
            self.exposure_info["software_gamma"] = False

    # ---------------------------------------------------------------- publish
    def _publish(self, frame):
        now = time.time()
        with self.lock:
            self.frame = frame
            self.ret = True
            self.frame_id += 1
            self.last_frame_time = now
            self.lock.notify_all()
        self._last_ok = now
        if self.status != "live":
            if self.status == "reconnecting":
                log("Live stream restored.")
            self.status = "live"

    # ---------------------------------------------------------------- reader
    def _reader_loop(self, cap, gen):
        try:
            period = 1.0 / self.native_fps if (self.loopable and self.native_fps) else 0.0
            next_t = time.perf_counter() + period
            fails = 0
            webcam = self.kind == "webcam"
            while self.running and self._gen == gen:
                if webcam:
                    if self._exp_dirty:
                        self._exp_dirty = False
                        self._apply_exposure(cap)
                    if self._settings_req:
                        self._settings_req = False
                        ok_dlg = bool(cap.set(cv2.CAP_PROP_SETTINGS, 1))
                        log("Camera driver settings dialog " + ("opened (set Power Line Frequency to 50 Hz here)"
                                                                 if ok_dlg else "is not supported by this backend"))
                ok, frame = cap.read()
                if self._gen != gen or not self.running:
                    break
                if ok and frame is not None and frame.size > 0:
                    fails = 0
                    if webcam and self.exposure_mode == "sport":
                        now = time.time()
                        if now - self._last_luma_t > 0.5:
                            self._last_luma_t = now
                            self._auto_gain(cap, frame)
                        if self._gamma_lut is not None:
                            frame = cv2.LUT(frame, self._gamma_lut)
                    self._publish(frame)
                    if period:
                        delay = next_t - time.perf_counter()
                        if delay > 0:
                            time.sleep(delay)
                            next_t += period
                        else:
                            next_t = time.perf_counter() + period   # fell behind: resync
                    continue

                fails += 1
                if self.loopable:
                    # End of file / VOD -> loop seamlessly. Break out (=> reopen) if seeking fails.
                    if fails > 3 or not cap.set(cv2.CAP_PROP_POS_FRAMES, 0):
                        break
                    next_t = time.perf_counter()
                    continue
                if not cap.isOpened():
                    break
                time.sleep(0.005)   # stream hiccup: the watchdog decides when to reconnect
        except Exception as exc:
            log(f"Camera reader error: {exc!r}")
        finally:
            try:
                cap.release()
            except Exception:
                pass

    # ------------------------------------------------------------- supervisor
    def _update(self):
        backoff = 0.5
        while self.running:
            gen = self._gen
            try:
                cap, first = self._connect()
            except Exception as exc:
                log(f"Camera connect error: {exc!r}")
                cap, first = None, None
            if not self.running:
                if cap is not None:
                    cap.release()
                break
            if cap is None:
                if self.status == "live":
                    self.status = "reconnecting"
                elif self.status == "connecting" and self.frame_id > 0:
                    self.status = "reconnecting"
                self._sleep(backoff)
                backoff = min(3.0, backoff * 1.7)
                continue

            backoff = 0.5
            if first is not None:
                self._publish(first)
            self._last_ok = time.time()
            reader = threading.Thread(target=self._reader_loop, args=(cap, gen),
                                      name=f"PadelIQReader-{gen}", daemon=True)
            reader.start()

            # Watchdog
            while self.running and reader.is_alive():
                time.sleep(0.1)
                if self.kind != "file" and (time.time() - self._last_ok) > self.stall_timeout:
                    log(f"No frames for {time.time() - self._last_ok:.1f}s -> reconnecting to {self.spec.label} ...")
                    break

            self._gen += 1   # invalidate the current reader (it releases its own cap)
            if not self.running:
                break
            if self.kind != "file":
                self.status = "reconnecting"
                self.reconnect_count += 1
            reader.join(timeout=0.2)

    @property
    def loopable(self):
        """Recorded media (local file or YouTube VOD): paced at native FPS and looped."""
        return self.kind == "file" or (self.kind == "youtube" and not self.spec.is_live)

    def _sleep(self, seconds):
        end = time.time() + seconds
        while self.running and time.time() < end:
            time.sleep(0.05)

    # ------------------------------------------------------------- public API
    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def read_latest(self):
        """(frame, frame_id, capture_timestamp) or (None, 0, 0.0). Frame is a private copy."""
        with self.lock:
            if self.frame is None:
                return None, 0, 0.0
            return self.frame.copy(), self.frame_id, self.last_frame_time

    def wait_latest(self, last_id, timeout=0.05):
        """Block until a frame newer than last_id is published (or timeout), then return the newest one.
        Consumers never see a queue: if they were slow, intermediate frames are simply skipped."""
        return wait_latest_frame(self, last_id, timeout)

    def read_with_id(self):
        with self.lock:
            if self.frame is None:
                return False, None, 0
            return True, self.frame.copy(), self.frame_id

    @property
    def seconds_since_frame(self):
        return float("inf") if self.last_frame_time == 0 else time.time() - self.last_frame_time

    @property
    def is_degraded(self):
        return self.status != "live" or self.seconds_since_frame > self.stall_timeout

    def release(self):
        self.running = False
        self._gen += 1
        self.thread.join(timeout=2.0)


def wait_latest_frame(cam, last_id, timeout=0.05):
    """Shared by every camera type exposing `lock` (threading.Condition), `frame`, `frame_id`, `last_frame_time`."""
    with cam.lock:
        if cam.frame is None or cam.frame_id == last_id:
            cam.lock.wait(timeout)
        if cam.frame is None:
            return None, 0, 0.0
        return cam.frame.copy(), cam.frame_id, cam.last_frame_time


def cam_wait_latest(cam, last_id, timeout=0.05):
    fn = getattr(cam, "wait_latest", None)
    if fn is not None:
        return fn(last_id, timeout)
    frame, fid, ts = cam.read_latest()
    if frame is None or fid == last_id:
        time.sleep(min(timeout, 0.004))
        frame, fid, ts = cam.read_latest()
    return frame, fid, ts


# =====================================================================
# 4. Court calibration (True AI floor segmentation & 2-VP homography)
# =====================================================================
_SHARED_COURT_AI = None
_SHARED_COURT_AI_LOCK = threading.Lock()


def get_court_ai(pose_detector=None):
    global _SHARED_COURT_AI
    with _SHARED_COURT_AI_LOCK:
        if _SHARED_COURT_AI is None:
            _SHARED_COURT_AI = CG.CourtGeometryAI(pose_detector=pose_detector, app_dir=APP_DIR)
        elif pose_detector is not None and _SHARED_COURT_AI.pose is not pose_detector:
            _SHARED_COURT_AI.pose = pose_detector
        return _SHARED_COURT_AI


def calibration_files_for(kind):
    """(save_path, [load candidates]) — the simulated video and live cameras keep separate calibrations."""
    if kind == "file":
        return SIM_CALIBRATION_FILE, [SIM_CALIBRATION_FILE]   # court_calibration.json belonged to the old annotated clip
    if kind == "mobile":
        return MOBILE_CALIBRATION_FILE, [MOBILE_CALIBRATION_FILE]
    return LIVE_CALIBRATION_FILE, [LIVE_CALIBRATION_FILE]


def load_calibration(path, frame_shape, source_label=None):
    """source_label: when given, the file is only used if it was saved for that exact source."""
    h, w = frame_shape[:2]
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            if source_label is not None and data.get("source_label") != source_label:
                return None
            pts = np.array(data.get("corners", []), dtype=np.float32)
            size = data.get("frame_size")
            if pts.shape == (4, 2) and size and size[0] > 0 and size[1] > 0:
                pts[:, 0] *= w / float(size[0])
                pts[:, 1] *= h / float(size[1])
        else:
            pts = np.array(data, dtype=np.float32)
        return pts if corners_plausible(pts, frame_shape) else None
    except Exception as exc:
        log(f"Ignoring unreadable calibration {os.path.basename(path)}: {exc}")
        return None


def corners_plausible(pts, frame_shape):
    """
    Full-court corners may lie far outside the image (e.g. a camera behind the near
    baseline only sees the far half), but at least 2 must be near the frame and the
    homography must be valid.
    """
    h, w = frame_shape[:2]
    pts = np.asarray(pts, dtype=np.float32)
    if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
        return False
    if np.abs(pts).max() > 60.0 * max(w, h):
        return False
    inside = ((pts[:, 0] >= -0.15 * w) & (pts[:, 0] <= 1.15 * w) &
              (pts[:, 1] >= -0.15 * h) & (pts[:, 1] <= 1.15 * h)).sum()
    return inside >= 2 and compute_homography(pts) is not None


# Calibration references: which 4 court points the user clicks (TL, TR, BR, BL in court metres).
CALIBRATION_REFERENCES = {
    "full": [(0, 0), (COURT_WIDTH_M, 0), (COURT_WIDTH_M, COURT_LENGTH_M), (0, COURT_LENGTH_M)],
    "far_half": [(0, 0), (COURT_WIDTH_M, 0), (COURT_WIDTH_M, NET_Y_M), (0, NET_Y_M)],
    "near_half": [(0, NET_Y_M), (COURT_WIDTH_M, NET_Y_M), (COURT_WIDTH_M, COURT_LENGTH_M), (0, COURT_LENGTH_M)],
}
CALIBRATION_REFERENCE_LABELS = {
    "full": "full court: far-L, far-R, near-R, near-L corners",
    "far_half": "far half: far-L, far-R corners, then net-R, net-L (floor at net)",
    "near_half": "near half: net-L, net-R (floor at net), then near-R, near-L corners",
}


def corners_from_reference(img_pts, reference="full"):
    """Convert 4 clicked image points for a calibration reference into full-court image corners."""
    ref = CALIBRATION_REFERENCES.get(reference)
    if ref is None:
        return None
    try:
        img = np.asarray(img_pts, dtype=np.float32).reshape(4, 2)
        if abs(cv2.contourArea(img)) < 100.0 or not cv2.isContourConvex(img.reshape(-1, 1, 2)):
            return None
        H_c2i = cv2.getPerspectiveTransform(np.array(ref, dtype=np.float32), img)
        full = np.array([CALIBRATION_REFERENCES["full"]], dtype=np.float32)
        corners = cv2.perspectiveTransform(full, H_c2i)[0]
        return corners.astype(np.float32) if compute_homography(corners) is not None else None
    except Exception:
        return None


def save_calibration(path, corners, frame_shape, extra=None):
    h, w = frame_shape[:2]
    try:
        data = {"frame_size": [int(w), int(h)],
                "corners": [[float(x), float(y)] for x, y in corners]}
        if extra:
            data.update(extra)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, path)          # atomic: never leaves a half-written calibration
        log(f"Calibration saved -> {os.path.basename(path)}")
    except Exception as exc:
        log(f"WARNING: could not save calibration: {exc}")


COURT_DST_PTS = np.array([[0, 0], [COURT_WIDTH_M, 0], [COURT_WIDTH_M, COURT_LENGTH_M], [0, COURT_LENGTH_M]],
                         dtype=np.float32)


def compute_homography(corners):
    return CG.compute_homography(corners)


# ---------------------------------------------------------------------------
# Automatic court calibration from padel line geometry
# ---------------------------------------------------------------------------
def _cluster_horizontal(lines, tol=10):
    """Group near-horizontal segments by height -> [[total_len, y_mean, [segments]]]."""
    clusters = []
    for x1, y1, x2, y2 in sorted(lines, key=lambda l: (l[1] + l[3]) / 2):
        ym = (y1 + y2) / 2.0
        length = float(np.hypot(x2 - x1, y2 - y1))
        for c in clusters:
            if abs(c[1] - ym) <= tol:
                c[2].append((x1, y1, x2, y2))
                c[0] += length
                c[1] = float(np.mean([(a[1] + a[3]) / 2 for a in c[2]]))
                break
        else:
            clusters.append([length, ym, [(x1, y1, x2, y2)]])
    return clusters


def _fit_line(segments):
    """Least-squares y = m*x + b through segment endpoints."""
    xs = np.array([v for sg in segments for v in (sg[0], sg[2])], dtype=np.float64)
    ys = np.array([v for sg in segments for v in (sg[1], sg[3])], dtype=np.float64)
    if np.ptp(xs) < 1:
        return 0.0, float(ys.mean())
    m, b = np.polyfit(xs, ys, 1)
    return float(m), float(b)


def _intersect(l1, l2):
    (m1, b1), (m2, b2) = l1, l2
    if abs(m1 - m2) < 1e-9:
        return None
    x = (b2 - b1) / (m1 - m2)
    return float(x), float(m1 * x + b1)


def auto_calibrate_court(frames):
    """
    Detect the court from padel line geometry (works even when the near corners are off-screen):
      * near service line (white, 6.95 m from the net)  -> court Y = 16.95
      * net floor line (bottom edge of the net mesh)     -> court Y = 10
      * side-wall/floor boundaries in the far half       -> court X = 0 and X = 10
    Their 4 intersections define the homography; all 4 full-court corners are then projected.
    Returns (corners, info) or (None, reason).
    """
    try:
        frames = [f for f in frames if f is not None and getattr(f, "size", 0) > 0]
        if not frames:
            return None, "no frames"
        h, w = frames[0].shape[:2]
        frames = [f for f in frames if f.shape[:2] == (h, w)][:40]
        # Temporal median removes players, ball and compression noise.
        bg = np.median(np.stack(frames), axis=0).astype(np.uint8) if len(frames) > 2 else frames[0]
        hsv = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
        S, V = hsv[:, :, 1], hsv[:, :, 2]

        # 1) white line segments
        white = ((V > 175) & (S < 60)).astype(np.uint8) * 255
        segs = cv2.HoughLinesP(white, 1, np.pi / 360, 80, minLineLength=int(0.09 * w), maxLineGap=12)
        segs = [] if segs is None else [tuple(int(v) for v in l) for l in segs.reshape(-1, 4)]
        horiz = [l for l in segs if abs(l[3] - l[1]) <= 0.07 * max(1, abs(l[2] - l[0]))]
        vert = [l for l in segs if abs(l[2] - l[0]) <= 0.12 * max(1, abs(l[3] - l[1]))]
        clusters = _cluster_horizontal(horiz)

        low = [c for c in clusters if c[1] > 0.55 * h and c[0] > 0.25 * w]
        if not low:
            return None, "near service line not found"
        service = max(low, key=lambda c: c[0])
        service_line = _fit_line(service[2])

        mid = [c for c in clusters if 0.25 * h < c[1] < service[1] - 0.12 * h and c[0] > 0.3 * w]
        if not mid:
            return None, "net band not found"
        band_y = max(mid, key=lambda c: c[0])[1]

        centre = [l for l in vert if 0.25 * w < (l[0] + l[2]) / 2 < 0.75 * w and max(l[1], l[3]) > band_y]
        x_c = float(np.median([(l[0] + l[2]) / 2 for l in centre])) if centre else w / 2.0
        centre_fit = None
        if centre:
            near = [l for l in centre if abs((l[0] + l[2]) / 2 - x_c) < 0.02 * w]
            ys_ = np.array([v for l in near for v in (l[1], l[3])], dtype=np.float64)
            xs_ = np.array([v for l in near for v in (l[0], l[2])], dtype=np.float64)
            if len(ys_) >= 2 and np.ptp(ys_) > 0.1 * h:
                centre_fit = tuple(float(v) for v in np.polyfit(ys_, xs_, 1))   # x = a*y + c

        # 2) net floor = strongest dark->bright step below the band (net mesh -> clean floor)
        y0, y1 = int(band_y + 8), int(min(h - 2, service[1] - 0.05 * h))
        cols = slice(int(0.2 * w), int(0.8 * w))
        prof = np.array([np.median(V[y, cols]) for y in range(y0, y1)], dtype=np.float64)
        if len(prof) < 12:
            return None, "net floor search window too small"
        prof = np.convolve(prof, np.ones(3) / 3, mode="same")
        step = prof[3:] - prof[:-3]
        y_net = float(y0 + int(np.argmax(step[2:-2])) + 4)
        net_line = (0.0, y_net)

        # 3) side-wall / floor boundaries between the band and the net floor (far half).
        #    Collect candidates per side, then pick the PAIR that is geometrically consistent:
        #    vanishing point on the centre line, mirror-symmetric slopes, long segments.
        ra, rb = int(band_y + 4), int(y_net - 2)
        if rb - ra < 10:
            return None, "side-wall search window too small"
        cands = {"left": [], "right": []}
        for chan in (S, V):
            edges = cv2.Canny(cv2.GaussianBlur(chan, (5, 5), 0), 15, 45)
            for side, sign in (("left", -1), ("right", 1)):
                xa, xb = (0, int(x_c - 0.12 * w)) if side == "left" else (int(x_c + 0.12 * w), w)
                if xb - xa < 20:
                    continue
                roi = np.zeros_like(edges)
                roi[ra:rb, xa:xb] = edges[ra:rb, xa:xb]
                lines = cv2.HoughLinesP(roi, 1, np.pi / 720, 25, minLineLength=int(0.035 * w), maxLineGap=10)
                for x1_, y1_, x2_, y2_ in ([] if lines is None else lines.reshape(-1, 4)):
                    if x2_ == x1_:
                        continue
                    slope = (y2_ - y1_) / float(x2_ - x1_)
                    if not (0.15 < abs(slope) < 1.5) or np.sign(slope) != sign:
                        continue
                    line = _fit_line([(int(x1_), int(y1_), int(x2_), int(y2_))])
                    x_at_net = (y_net - line[1]) / line[0]
                    if not (-0.05 * w <= x_at_net <= 1.05 * w):     # wall must meet the net inside the image
                        continue
                    cands[side].append((float(np.hypot(x2_ - x1_, y2_ - y1_)), line))
        if not cands["left"] or not cands["right"]:
            return None, "side-wall lines not found"
        best = None
        for side in cands:
            cands[side] = sorted(cands[side], key=lambda c: -c[0])[:25]
        for len_l, line_l in cands["left"]:
            for len_r, line_r in cands["right"]:
                vp_ = _intersect(line_l, line_r)
                if vp_ is None or vp_[1] > band_y:
                    continue
                vp_err = abs(vp_[0] - x_c) / w
                sym_err = abs(abs(line_l[0]) - abs(line_r[0])) / max(abs(line_l[0]), abs(line_r[0]))
                score = (len_l + len_r) / w - 8.0 * vp_err - 2.0 * sym_err
                if best is None or score > best[0]:
                    best = (score, line_l, line_r, vp_err, sym_err)
        if best is None:
            return None, "no consistent side-wall pair"
        walls = {"left": best[1], "right": best[2]}

        vp = _intersect(walls["left"], walls["right"])
        if vp is None or abs(vp[0] - x_c) > 0.05 * w or best[4] > 0.35:
            return None, f"side walls inconsistent (vanishing point {vp}, centre line x={x_c:.0f})"

        pts = [_intersect(walls["left"], net_line), _intersect(walls["right"], net_line),
               _intersect(walls["right"], service_line), _intersect(walls["left"], service_line)]
        if any(p is None for p in pts):
            return None, "degenerate line intersections"
        court_ref = [(0, NET_Y_M), (COURT_WIDTH_M, NET_Y_M),
                     (COURT_WIDTH_M, SERVICE_LINE_FAR_M), (0, SERVICE_LINE_FAR_M)]
        img_ref = list(pts)
        if centre_fit is not None:
            # Centre service line adds 2 more correspondences -> least squares absorbs camera roll.
            a_, c_ = centre_fit
            m_s, b_s = service_line
            y_cs = (m_s * c_ + b_s) / (1.0 - m_s * a_) if abs(1.0 - m_s * a_) > 1e-9 else None
            if y_cs is not None:
                court_ref += [(COURT_WIDTH_M / 2, NET_Y_M), (COURT_WIDTH_M / 2, SERVICE_LINE_FAR_M)]
                img_ref += [(a_ * y_net + c_, y_net), (a_ * y_cs + c_, y_cs)]
        H_c2i, _ = cv2.findHomography(np.array(court_ref, dtype=np.float32), np.array(img_ref, dtype=np.float32), 0)
        if H_c2i is None:
            return None, "homography solve failed"
        reproj = cv2.perspectiveTransform(np.array([court_ref], dtype=np.float32), H_c2i)[0]
        reproj_err = float(np.max(np.linalg.norm(reproj - np.array(img_ref, dtype=np.float32), axis=1)))
        corners = cv2.perspectiveTransform(np.array([CALIBRATION_REFERENCES["full"]], dtype=np.float32), H_c2i)[0]
        if not corners_plausible(corners, bg.shape):
            return None, "computed corners implausible"
        info = {"method": "auto_court_lines",
                "service_line_y": round(float(np.mean([sg[1] for sg in service[2]])), 1),
                "net_band_y": round(float(band_y), 1), "net_floor_y": round(y_net, 1),
                "centre_line_x": round(x_c, 1), "vanishing_point": [round(vp[0], 1), round(vp[1], 1)],
                "reference": "net floor + near service line x side walls + centre line",
                "correspondences": len(img_ref), "max_reprojection_error_px": round(reproj_err, 2),
                "reference_image_points": [[round(p[0], 1), round(p[1], 1)] for p in pts],
                "frames_used": len(frames)}
        return corners.astype(np.float32), info
    except Exception as exc:
        return None, f"auto-calibration error: {exc!r}"


def draw_calibration_preview(frame, corners):
    """Project the real padel lines through a calibration so it can be checked visually."""
    out = frame.copy()
    H = compute_homography(corners)
    if H is None:
        return out
    Hi = CG.invert_homography_c2i(H)
    if Hi is None:
        return out

    def seg(a, b, color, th=2):
        pts = CG.sample_clipped_segment(Hi, a, b, out.shape)
        if pts is not None:
            cv2.polylines(out, [pts.astype(np.int32)], False, color, th, cv2.LINE_AA)

    W, L = COURT_WIDTH_M, COURT_LENGTH_M
    for a, b in (((0, 0), (W, 0)), ((W, 0), (W, L)), ((W, L), (0, L)), ((0, L), (0, 0))):
        seg(a, b, (0, 255, 0), 3)
    seg((0, NET_Y_M), (W, NET_Y_M), (255, 255, 255), 3)
    for y in (SERVICE_LINE_NEAR_M, SERVICE_LINE_FAR_M):
        seg((0, y), (W, y), (0, 255, 255), 2)
    seg((W / 2, SERVICE_LINE_NEAR_M), (W / 2, SERVICE_LINE_FAR_M), (255, 0, 255), 2)
    return out


def collect_calibration_frames(spec, count=18, max_seconds=1.6, start_frame=0):
    """Collect 15-20 contiguous/near-contiguous frames from the active scene (~1.0-1.6s),
    stopping early if a scene cut occurs so multi-angle compilations never mix two courts."""
    frames = []
    cut_det = CG.SceneCutDetector(confirm_frames=1)
    if spec.kind == "file":
        cap = cv2.VideoCapture(spec.source)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        span = min(max(count, int(round(fps * 1.6))), max(1, n - int(start_frame) - 1)) if n > 0 else count * 2
        idxs = np.linspace(int(start_frame), int(start_frame) + span, count).astype(int) if n > 0 else range(count)
        for idx in idxs:
            if n > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok and f is not None:
                triggered, _ = cut_det.update(f)
                if triggered and len(frames) >= 6:
                    break
                frames.append(f)
        cap.release()
        return frames
    cam = ThreadedLiveCamera(spec)
    try:
        end, last = time.time() + max_seconds + 6.0, -1
        t_started = None
        while time.time() < end and len(frames) < count:
            f, fid, _ = cam.read_latest()
            if f is not None and fid != last:
                if t_started is None:
                    t_started = time.time()
                last = fid
                triggered, _ = cut_det.update(f)
                if triggered and len(frames) >= 6:
                    break
                frames.append(f)
                if time.time() - t_started >= max_seconds and len(frames) >= 10:
                    break
                time.sleep(max_seconds / max(1, count))
            else:
                time.sleep(0.015)
    finally:
        cam.release()
    return frames


def interactive_live_calibration(current_frame):
    """Click 4 floor reference points. [H] cycles full court / far half / near half. [C] confirm, [R] reset, [Esc] cancel."""
    clicked = []
    win_name = "LIVE CALIBRATION: Click 4 Floor Corners (TL, TR, BR, BL) -> Press 'C' to Save"
    h, w = current_frame.shape[:2]
    s = hud_scale(current_frame)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append([float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))])

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)
    labels = ["1: Top-Left", "2: Top-Right", "3: Bottom-Right", "4: Bottom-Left"]
    result = None
    shown_once = False
    ref_modes = ["full", "far_half", "near_half"]
    ref_idx = 0

    while True:
        disp = current_frame.copy()
        msg = ("Select Corner: " + labels[len(clicked)]) if len(clicked) < 4 else "Press 'C' to Confirm or 'R' to Reset"
        draw_glass_box(disp, 10 * s, 10 * s, min(w - 10, 10 * s + 700 * s), 10 * s + 74 * s, alpha=0.75)
        put_text(disp, msg, (22 * s, 42 * s), 0.75 * s, (0, 255, 255), 2, s)
        put_text(disp, fit_text(f"[H] {CALIBRATION_REFERENCE_LABELS[ref_modes[ref_idx]]}  [Esc] cancel",
                                0.45 * s, 1, 680 * s), (22 * s, 70 * s), 0.45 * s, (200, 200, 200), 1, s)

        for i, pt in enumerate(clicked):
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(disp, (px, py), max(4, int(7 * s)), (0, 255, 0), -1, cv2.LINE_AA)
            put_text(disp, labels[i], (px + 10 * s, py - 10 * s), 0.55 * s, (0, 255, 0), 2, s)
            if i > 0:
                cv2.line(disp, (int(clicked[i - 1][0]), int(clicked[i - 1][1])), (px, py), (0, 255, 255), 2, cv2.LINE_AA)
        if len(clicked) == 4:
            cv2.line(disp, (int(clicked[3][0]), int(clicked[3][1])), (int(clicked[0][0]), int(clicked[0][1])),
                     (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow(win_name, disp)
        k = cv2.waitKey(20) & 0xFF
        if shown_once and cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break   # user closed the window
        shown_once = True
        if k in (ord('r'), ord('R')):
            clicked.clear()
        elif k in (ord('h'), ord('H')):
            ref_idx = (ref_idx + 1) % len(ref_modes)
            clicked.clear()
        elif k in (ord('c'), ord('C')) and len(clicked) == 4:
            pts = corners_from_reference(clicked, ref_modes[ref_idx])
            if pts is None:
                log("Those 4 points do not form a valid court (must be convex, in TL,TR,BR,BL order). Try again.")
                clicked.clear()
                continue
            result = pts
            break
        elif k == 27:
            break

    try:
        cv2.destroyWindow(win_name)
    except cv2.error:
        pass
    return result



# =====================================================================
# 4b. Auto-calibration review (fast 15-30 frame convergence -> 3s lock / 6-handle drag)
# =====================================================================
REVIEW_AUTO_CONFIRM_S = 3.0


def run_court_autocalibration(frames, fast_single=False):
    """True Multi-Angle AI Court Geometry & Keypoint Homography Engine (`CourtGeometryAI`).
    Works on straight-on, low-baseline, elevated, and 30-45 deg oblique/diagonal angles."""
    frames = [f for f in frames if f is not None and getattr(f, "size", 0) > 0]
    if not frames:
        return None, "no frames"
    ai = get_court_ai()
    return ai.calibrate(frames, fast_single=fast_single)


def court_handles(H_i2c, frame_shape):
    """6 named high-visibility draggable handles: `TL, TR, BR, BL, NET-L, NET-R`."""
    court_pts, img_pts, _ = CG.build_six_handles(H_i2c, frame_shape)
    return court_pts, img_pts


def aspect_changed(old_shape, new_shape, tol=0.08):
    """True when width/height ratio changed by more than tol (portrait <-> landscape, 4:3 <-> 16:9 lens)."""
    try:
        a0 = old_shape[1] / float(old_shape[0])
        a1 = new_shape[1] / float(new_shape[0])
        return abs(a1 / a0 - 1.0) > tol
    except (ZeroDivisionError, IndexError, TypeError):
        return False


class CalibrationReview:
    """idle -> detecting (<= 1.5s) -> review --(Enter/Y or 3 s idle)--> confirmed
                                             \\--(C/M)--> manual (drag 6 handles TL,TR,BR,BL,NET-L,NET-R | [N] click 4) --(Enter/C)--> confirmed
       [A] re-scans from any state."""

    def __init__(self, auto_confirm_s=REVIEW_AUTO_CONFIRM_S):
        self.auto_confirm_s = auto_confirm_s
        self.state = "idle"
        self.corners = None
        self.info = {}
        self.error = ""
        self.review_since = 0.0
        self.touched = False
        self.version = 0                    # bumps whenever self.corners changes
        self.generation = 0
        self.frame_shape = None
        self.manual_mode = "drag"           # drag | click
        self.handles_court = None
        self.handles_img = None
        self.handle_labels = list(CG.HANDLE_LABELS)
        self.drag_idx = None
        self.clicked = []
        self.ref_idx = 0
        self.confirmed_by = None
        self.lock = threading.RLock()

    # ---------------------------------------------------------------- detection
    def start(self, frames_fn, frame_shape=None, seed_frame=None):
        with self.lock:
            self.generation += 1
            gen = self.generation
            self.state, self.error, self.touched = "detecting", "", False
            self.confirmed_by = None
            if frame_shape is not None:
                self.frame_shape = tuple(frame_shape[:2])

        def work():
            t0 = time.time()
            if seed_frame is not None:
                try:
                    c_seed, info_seed = run_court_autocalibration([seed_frame], fast_single=True)
                    with self.lock:
                        if gen == self.generation and c_seed is not None and self.corners is None:
                            self.frame_shape = tuple(seed_frame.shape[:2])
                            self._set_corners(c_seed)
                            self.info = dict(info_seed, provisional=True)
                except Exception:
                    pass

            try:
                frames = frames_fn() if callable(frames_fn) else frames_fn
                corners, info = run_court_autocalibration(frames, fast_single=False)
                shape = frames[0].shape[:2] if frames else None
            except Exception as exc:
                corners, info, shape = None, f"auto-calibration error: {exc!r}", None
            with self.lock:
                if gen != self.generation:
                    return                              # superseded by a newer source / scene cut / re-run
                if corners is None:
                    self.state, self.error = "failed", str(info)
                    self.corners = None
                    log(f"Auto-calibration waiting for visible court ({info}) - press [C] to calibrate or [A] to re-scan")
                    return
                if shape is not None:
                    self.frame_shape = tuple(shape)
                info = dict(info)
                info["detection_s"] = round(time.time() - t0, 2)
                self._set_corners(corners)
                self.info = info
                self.state, self.review_since = "review", time.time()
                log(f"COURT LOCKED (Confidence: {info.get('confidence_pct')}% | Angle: {info.get('angle', 'Baseline')}) "
                    f"in {info['detection_s']}s - [ENTER] Confirm | [C] Drag Corners | [A] Re-Scan")

        threading.Thread(target=work, name="PadelIQ-autocalib", daemon=True).start()

    def reset(self):
        """Back to idle (cancels a running detection) - e.g. the phone was rotated to a new aspect ratio."""
        with self.lock:
            self.generation += 1
            self.state, self.corners, self.info, self.error = "idle", None, {}, ""
            self.handles_img = self.handles_court = None
            self.touched, self.drag_idx, self.confirmed_by = False, None, None
            self.version += 1

    def restore(self, corners, info, frame_shape):
        """Re-instate a previously confirmed calibration (F12 source swap back to this camera)."""
        with self.lock:
            self.generation += 1
            self.frame_shape = frame_shape
            self._set_corners(corners)
            self.info = dict(info or {})
            self.state, self.error = "confirmed", ""
            self.touched, self.drag_idx = False, None
            self.confirmed_by = self.info.get("confirmed_by", "restored")

    def _set_corners(self, corners):
        self.corners = np.asarray(corners, dtype=np.float32)
        self.version += 1
        H = compute_homography(self.corners)
        if H is not None and self.frame_shape is not None:
            self.handles_court, self.handles_img, self.handle_labels = CG.build_six_handles(H, self.frame_shape)

    # ----------------------------------------------------------------- decisions
    def tick(self, now=None):
        """Auto-confirm after auto_confirm_s (3s) without user interaction. -> True when it just confirmed."""
        now = time.time() if now is None else now
        with self.lock:
            if self.state == "review" and not self.touched and now - self.review_since >= self.auto_confirm_s:
                return self.confirm("auto (no input for %.0f s)" % self.auto_confirm_s)
        return False

    def confirm(self, by="user"):
        with self.lock:
            if self.corners is None or self.state not in ("review", "manual"):
                return False
            if self.state == "manual":
                self.info = dict(self.info, method="manual_adjusted",
                                 base_method=self.info.get("base_method", self.info.get("method")))
            self.state, self.confirmed_by = "confirmed", by
            self.info = dict(self.info, confirmed_by=by, confirmed_at=datetime.now().isoformat(timespec="seconds"))
            log(f"Court calibration confirmed ({by}).")
            return True

    def seconds_left(self, now=None):
        now = time.time() if now is None else now
        return max(0.0, self.auto_confirm_s - (now - self.review_since)) if self.state == "review" else None

    def enter_manual(self, fallback_corners=None, frame_shape=None):
        with self.lock:
            if frame_shape is not None:
                self.frame_shape = tuple(frame_shape[:2])
            self.generation += 1                      # cancels a running detection
            if self.corners is None and fallback_corners is not None:
                self._set_corners(fallback_corners)
            elif self.corners is not None:
                self._set_corners(self.corners)
            self.touched = True
            self.state = "manual"
            self.manual_mode = "drag" if self.handles_img is not None else "click"
            self.clicked = []
            if "base_method" not in self.info:
                self.info = dict(self.info, base_method=self.info.get("method"))

    def toggle_click_mode(self):
        with self.lock:
            if self.state == "manual":
                self.manual_mode = "click" if self.manual_mode == "drag" or self.handles_img is None else "drag"
                self.clicked = []

    def cycle_reference(self):
        with self.lock:
            self.ref_idx = (self.ref_idx + 1) % len(CALIBRATION_REFERENCES)
            self.clicked = []

    @property
    def reference(self):
        return list(CALIBRATION_REFERENCES)[self.ref_idx]

    def set_from_handles(self, court_pts, img_pts, dragged_idx=None):
        """Homography from 4-6 court<->image handle pairs (`TL, TR, BR, BL, NET-L, NET-R`)."""
        corners, court_clean, img_clean = CG.homography_from_handles(
            court_pts, img_pts, dragged_idx=dragged_idx, frame_shape=self.frame_shape
        )
        with self.lock:
            self.corners = corners
            self.version += 1
            self.handles_court, self.handles_img = court_clean, img_clean
            if len(court_clean) == len(CG.HANDLE_LABELS):
                self.handle_labels = list(CG.HANDLE_LABELS)
            else:
                self.handle_labels = [f"P{i+1}" for i in range(len(court_clean))]
            self.touched = True
            if self.state != "manual":
                self.state = "manual"
        return self.corners

    def set_manual_corners(self, corners, info=None):
        with self.lock:
            self.generation += 1
            self._set_corners(corners)
            self.info = dict(info or {}, method=(info or {}).get("method", "manual"))
            self.touched = True
            self.state = "manual"

    # ------------------------------------------------------------------- mouse
    def on_mouse(self, event, x, y):
        with self.lock:
            if self.state != "manual":
                return
            if self.manual_mode == "drag" and self.handles_img is not None:
                if event == cv2.EVENT_LBUTTONDOWN:
                    d = np.linalg.norm(self.handles_img - np.array([x, y], np.float32), axis=1)
                    k = int(np.argmin(d))
                    self.drag_idx = k if d[k] < 44 else None
                elif event == cv2.EVENT_MOUSEMOVE and self.drag_idx is not None:
                    img = self.handles_img.copy()
                    img[self.drag_idx] = (x, y)
                    try:
                        self.set_from_handles(self.handles_court, img, dragged_idx=self.drag_idx)
                    except ValueError:
                        pass
                elif event == cv2.EVENT_LBUTTONUP:
                    self.drag_idx = None
            elif self.manual_mode == "click" and event == cv2.EVENT_LBUTTONDOWN and len(self.clicked) < 4:
                self.clicked.append([float(x), float(y)])
                if len(self.clicked) == 4:
                    pts = corners_from_reference(self.clicked, self.reference)
                    if pts is None:
                        log("Those 4 points are not a valid court (convex, TL,TR,BR,BL order) - click again.")
                    else:
                        self._set_corners(pts)
                        self.info = dict(self.info, method="manual_clicked", reference=self.reference)
                        self.manual_mode = "drag" if self.handles_img is not None else "click"
                    self.clicked = []

    def banner_text(self):
        with self.lock:
            if self.state == "detecting":
                return "AI COURT RECOGNITION - scanning floor surface & vanishing lines ..."
            if self.state in ("review", "confirmed"):
                conf = self.info.get("confidence_pct")
                cs = f"{conf:.0f}%" if conf is not None else "n/a"
                ang = self.info.get("angle") or "Baseline"
                return f"COURT LOCKED (Confidence: {cs} | Angle: {ang}) — [ENTER] Confirm | [C] Drag Corners | [A] Re-Scan"
            if self.state == "manual":
                return "MANUAL FINE-TUNING — Drag TL/TR/BR/BL/NET-L/NET-R or [N] 4 Fresh Corners | [ENTER/C] Lock"
            return "Waiting for visible court floor — [A] Re-Scan | [C] Manual Corners"

    def status(self):
        with self.lock:
            return {"state": self.state, "confidence_pct": self.info.get("confidence_pct"),
                    "angle": self.info.get("angle") or "Baseline",
                    "method": self.info.get("method"), "error": self.error,
                    "banner": self.banner_text(),
                    "auto_confirm_in_s": None if self.seconds_left() is None or self.touched else round(self.seconds_left(), 1),
                    "manual_mode": self.manual_mode if self.state == "manual" else None,
                    "reference": self.reference, "confirmed_by": self.confirmed_by,
                    "handles": None if self.handles_img is None else {
                        "labels": list(self.handle_labels),
                        "court_m": np.round(self.handles_court, 3).tolist(),
                        "image_px": np.round(self.handles_img, 1).tolist()}}


# =====================================================================
# 5. Real-time tracking & analytics
# =====================================================================
class OneEuroFilter:
    """One-Euro filter (Casiez et al. 2012) on a vector: strong smoothing while still (no jitter on the big screen),
    the cutoff rises with speed so fast movement keeps a low lag."""

    def __init__(self, min_cutoff=1.2, beta=0.012, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None
        self.dx = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x = self.dx = None

    def __call__(self, x, dt):
        x = np.asarray(x, dtype=np.float32)
        if self.x is None:
            self.x, self.dx = x.copy(), np.zeros_like(x)
            return self.x.copy()
        dt = max(1e-3, float(dt))
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * (x - self.x) / dt + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * np.abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        return self.x.copy()


class LivePlayerState:
    def __init__(self, slot_id, default_pos_m):
        self.slot_id = slot_id
        self.last_pos_m = np.array(default_pos_m, dtype=np.float32)
        self.smoothed_pos_m = None
        self.bbox = None
        self.last_seen_time = 0.0
        self.last_update_time = time.time()

        self.pixel_trail = deque(maxlen=30)
        self.meter_trail = deque(maxlen=40)

        self.distance_m = 0.0
        self.instant_speed_ms = 0.0
        self.avg_speed_ms = 0.0
        self.speed_history = deque(maxlen=60)
        self.active_seconds_total = 0.0
        self.tracked_seconds_total = 0.0
        self.visited_grid_cells = set()
        self.aspect_ratios = deque(maxlen=25)
        self.stability_pct = 92.0
        self.bbox_vel = np.zeros(4, dtype=np.float32)      # px/s, for render-time prediction
        self.bbox_filter = OneEuroFilter()
        self.kf_x = None                                    # Kalman state [x, y, vx, vy] in metres
        self.kf_P = None
        self.last_seen_prev = 0.0
        self.heat = np.zeros((int(COURT_LENGTH_M / HEAT_CELL_M), int(COURT_WIDTH_M / HEAT_CELL_M)),
                             dtype=np.float32)              # seconds spent per cell
        self.max_speed_ms = 0.0
        self._recent_speed = deque(maxlen=12)

    def _kalman_step(self, z, sigma, dt):
        """Constant-velocity Kalman filter; measurement noise scales with distance from the camera."""
        R = np.diag([sigma[0] ** 2, sigma[1] ** 2])
        if self.kf_x is None:
            self.kf_x = np.array([z[0], z[1], 0.0, 0.0])
            self.kf_P = np.diag([R[0, 0], R[1, 1], 4.0, 4.0])
            return
        F = np.eye(4)
        F[0, 2] = F[1, 3] = dt
        q = KF_ACCEL_SIGMA ** 2
        G = np.array([[0.5 * dt * dt, 0], [0, 0.5 * dt * dt], [dt, 0], [0, dt]])
        Q = G @ G.T * q
        x = F @ self.kf_x
        P = F @ self.kf_P @ F.T + Q
        Hm = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
        y = np.asarray(z, dtype=np.float64) - Hm @ x
        S = Hm @ P @ Hm.T + R
        K = P @ Hm.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(4) - K @ Hm) @ P
        v = np.hypot(x[2], x[3])
        if v > MAX_PLAYER_SPEED_MS:                      # physical cap
            x[2:] *= MAX_PLAYER_SPEED_MS / v
        # Padel is enclosed by walls: positions cannot leave the 10 x 20 m court (small margin for doors).
        x[0] = np.clip(x[0], -0.3, COURT_WIDTH_M + 0.3)
        x[1] = np.clip(x[1], 0.0, COURT_LENGTH_M)
        self.kf_x, self.kf_P = x, P

    def update(self, bbox, foot_px, pos_m, now, meas_sigma=None):
        dt = max(0.016, min(0.5, now - self.last_update_time))
        self.last_update_time = now

        self.last_seen_prev = self.last_seen_time
        bbox_arr = np.array(bbox, dtype=np.float32)
        if self.bbox is None or (now - self.last_seen_time) > 1.2:
            self.bbox_filter.reset()
            self.bbox = self.bbox_filter(bbox_arr, dt)
            self.bbox_vel[:] = 0.0
            self.pixel_trail.clear()
        else:
            new_bbox = self.bbox_filter(bbox_arr, dt)
            vel = (new_bbox - self.bbox) / dt
            self.bbox_vel = 0.5 * self.bbox_vel + 0.5 * np.clip(vel, -2500.0, 2500.0)
            self.bbox = new_bbox

        self.last_seen_time = now
        self.tracked_seconds_total += dt

        sb = self.bbox
        if foot_px is not None:
            self.pixel_trail.append((int(foot_px[0]), int(foot_px[1])))
        else:
            self.pixel_trail.append((int((sb[0] + sb[2]) / 2), int(sb[3])))

        raw_m = np.array(pos_m, dtype=np.float64)
        sigma = meas_sigma if meas_sigma is not None else (0.15, 0.15)
        reacquired = self.kf_x is not None and (now - self.last_seen_prev) > 1.2
        if reacquired:
            self.kf_x = None                       # long gap: restart the filter, no teleport distance
        prev = None if self.kf_x is None else self.kf_x[:2].copy()
        self._kalman_step(raw_m, sigma, dt)
        new_pos = self.kf_x[:2].astype(np.float32)
        speed = float(np.hypot(self.kf_x[2], self.kf_x[3]))
        if prev is not None:
            self.instant_speed_ms = 0.6 * self.instant_speed_ms + 0.4 * speed
            if self.instant_speed_ms > MOVE_SPEED_MS:
                step_dist = float(np.linalg.norm(new_pos - prev))
                self.distance_m += min(step_dist, MAX_PLAYER_SPEED_MS * dt)
                self.active_seconds_total += dt
            self.speed_history.append(self.instant_speed_ms)
        self.smoothed_pos_m = new_pos

        self.last_pos_m = self.smoothed_pos_m.copy()
        self.meter_trail.append(self.smoothed_pos_m.copy())
        self._recent_speed.append(self.instant_speed_ms)
        if len(self._recent_speed) >= 6:          # top speed = peak of a ~1 s rolling median (spike-proof)
            self.max_speed_ms = max(self.max_speed_ms, float(np.median(self._recent_speed)))
        hy = int(np.clip(self.smoothed_pos_m[1] / HEAT_CELL_M, 0, self.heat.shape[0] - 1))
        hx = int(np.clip(self.smoothed_pos_m[0] / HEAT_CELL_M, 0, self.heat.shape[1] - 1))
        self.heat[hy, hx] += dt
        if self.tracked_seconds_total > 0.5:
            self.avg_speed_ms = self.distance_m / self.tracked_seconds_total

        gx = int(np.clip(self.smoothed_pos_m[0], 0, 9.99))
        half_y = self.smoothed_pos_m[1] if self.slot_id in (1, 2) else (self.smoothed_pos_m[1] - 10.0)
        gy = int(np.clip(half_y, 0, 9.99))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if 0 <= gx + dx < 10 and 0 <= gy + dy < 10:
                    self.visited_grid_cells.add((gx + dx, gy + dy))

        w = max(1.0, float(sb[2] - sb[0]))
        h = max(1.0, float(sb[3] - sb[1]))
        self.aspect_ratios.append(h / w)
        if len(self.aspect_ratios) >= 5:
            ar_std = float(np.std(self.aspect_ratios))
            raw_stab = 98.0 - (ar_std * 40.0) - (max(0.0, self.instant_speed_ms - 3.0) * 4.0)
            self.stability_pct = 0.9 * self.stability_pct + 0.1 * float(np.clip(raw_stab, 55.0, 98.0))

    @property
    def coverage_pct(self):
        return min(100.0, float(len(self.visited_grid_cells)))

    @property
    def active_pct(self):
        if self.tracked_seconds_total <= 0.1:
            return 0
        return int(round(min(100.0, (self.active_seconds_total / self.tracked_seconds_total) * 100.0)))

    def display_bbox(self, now):
        """Constant-velocity prediction between detections (max 0.25 s) -> smooth 30 fps boxes."""
        if self.bbox is None:
            return None
        ahead = float(np.clip(now - self.last_seen_time, 0.0, 0.25))
        return self.bbox + self.bbox_vel * ahead

    def is_active(self, now, window=1.5):
        return self.smoothed_pos_m is not None and (now - self.last_seen_time) < window


class LiveSlotManager:
    def __init__(self, homography_matrix):
        self.H = homography_matrix
        self.floor_mask = None
        self.floor_top_y = None
        self.reset_players()

    def reset_players(self):
        anchors = {1: (2.5, 4.5), 2: (7.5, 4.5), 3: (2.5, 15.5), 4: (7.5, 15.5)}
        self.players = {sid: LivePlayerState(sid, anchors[sid]) for sid in (1, 2, 3, 4)}

    def set_homography(self, H, floor_mask=None, floor_top_y=None):
        self.H = H
        if floor_mask is not None:
            self.floor_mask = floor_mask
        if floor_top_y is not None:
            self.floor_top_y = floor_top_y

    def pixel_to_meters(self, px_pt):
        if self.H is None:
            return np.array([np.nan, np.nan], dtype=np.float32)
        return CG.project_pts(self.H, [[float(px_pt[0]), float(px_pt[1])]])[0]

    def net_y_px(self, x_px):
        """Image row of the net floor line (court y = 10 m) at image column x_px."""
        H_c2i = CG.invert_homography_c2i(self.H)
        if H_c2i is None:
            return None
        pts = CG.project_pts(H_c2i, [[0.0, NET_Y_M], [COURT_WIDTH_M, NET_Y_M]])
        if not np.all(np.isfinite(pts)):
            return None
        a, b = pts[0], pts[1]
        if abs(b[0] - a[0]) < 1e-6:
            return float((a[1] + b[1]) / 2)
        return float(a[1] + (x_px - a[0]) * (b[1] - a[1]) / (b[0] - a[0]))

    def update_detections(self, boxes, now=None, keypoints=None, frame_shape=None):
        """Strict main-court pose-ankle player filtering + true 3D Homography net team assignment:
          - Rejects bystanders, wall posters, seated spectators, and adjacent-court players.
          - Assigns `P1, P2` (Far Side: `y_m < 10.0m`) and `P3, P4` (Near Side: `y_m >= 10.0m`)
            strictly through the 3D Homography matrix `H`, so oblique 30-45 deg angles assign teams accurately!"""
        if self.H is None or boxes is None or len(boxes) == 0:
            return
        now = time.time() if now is None else float(now)
        candidates = []

        if frame_shape is not None:
            filtered = CG.CourtGeometryAI.filter_main_court_players(
                boxes, keypoints, self.H, frame_shape,
                floor_mask=self.floor_mask, floor_top_y=self.floor_top_y,
            )
            for item in filtered:
                fx, fy = item["foot_px"]
                candidates.append((
                    item["box"],
                    (int(round(fx)), int(round(fy))),
                    item["pos_m"],
                    item["sigma_m"],
                    item["far_side"],
                ))
        else:
            for box in boxes:
                x1, y1, x2, y2 = [float(v) for v in box[:4]]
                foot_px = (int((x1 + x2) / 2), int(y2))
                pos_m = self.pixel_to_meters(foot_px)
                if not np.all(np.isfinite(pos_m)):
                    continue
                if -0.65 <= pos_m[0] <= COURT_WIDTH_M + 0.65 and -0.65 <= pos_m[1] <= COURT_LENGTH_M + 1.15:
                    dx = self.pixel_to_meters((foot_px[0] + FOOT_PIXEL_SIGMA, foot_px[1])) - pos_m
                    dy = self.pixel_to_meters((foot_px[0], foot_px[1] + FOOT_PIXEL_SIGMA)) - pos_m
                    sx = float(np.clip(np.hypot(*dx), 0.03, 1.5)) if np.all(np.isfinite(dx)) else 0.25
                    sy = float(np.clip(np.hypot(*dy), 0.03, 1.5)) if np.all(np.isfinite(dy)) else 0.25
                    far_side = bool(pos_m[1] < NET_Y_M)
                    candidates.append(((x1, y1, x2, y2), foot_px, pos_m, (sx, sy), far_side))

        if not candidates:
            return

        slot_ids = [1, 2, 3, 4]
        cost = np.zeros((4, len(candidates)), dtype=np.float32)
        for r, sid in enumerate(slot_ids):
            pl = self.players[sid]
            for c, (_, _, pos_m, _, far_side) in enumerate(candidates):
                dist = float(np.linalg.norm(pos_m - pl.last_pos_m))
                penalty = 0.0
                if (sid in (1, 2)) != far_side:          # never assign across the 3D net (`y_m = 10.0m`)
                    penalty = 1e6
                since = now - pl.last_seen_time
                if pl.smoothed_pos_m is not None and since < 1.2:
                    if dist > MAX_PLAYER_SPEED_MS * max(since, 0.05) + 1.0:
                        penalty += 1e4
                cost[r, c] = dist + penalty

        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            sid = slot_ids[r]
            pl = self.players[sid]
            max_cost = 4.2 if pl.smoothed_pos_m is None else 12.0
            if cost[r, c] < max_cost:
                box, foot_px, pos_m, sig, _ = candidates[c]
                pl.update(box, foot_px, pos_m, now, meas_sigma=sig)



# =====================================================================
# 5b. Hybrid padel ball tracker
# =====================================================================
class PadelBallTracker:
    """Multi-cue padel-ball tracker. BallWorker runs it on EVERY camera frame (30-60 fps), independent of the
    person detector:
      1. 3-frame temporal motion mask  |I_t - I_t-1| & |I_t - I_t-2|  (court lines, wall logos and fixed LED
         reflections never move -> rejected; the AND localises the ball at time t, without a 1-frame lag)
      2. wide neon yellow-green HSV (H 20-55, S 65-255, V 110-255)  OR  a bright desaturated blur streak
         (fast smashes wash the colour out), both only inside the motion mask and the court + air ROI
      3. player exclusion: shoes (ankle keypoints 15/16) and bodies, except around the wrists / racket
         (keypoints 9/10) or on a fast ongoing track
      4. cv2.KalmanFilter(4, 2) + parabola gate (x linear, y quadratic over the last measurements); candidates
         are scored by distance to the prediction and compactness; up to COAST_FRAMES predicted frames bridge
         net / player occlusions.
    Speed is measured on the court plane (homography, ground scale at the nearest player's depth), clamped to
    15-140 km/h and EMA-smoothed. Direction reversals near a wrist / player = shot; elsewhere bounce / glass."""
    HSV_LO, HSV_HI = (20, 65, 110), (55, 255, 255)
    STREAK_LO, STREAK_HI = (0, 0, 205), (180, 70, 255)
    AREA_MIN, AREA_MAX = 8, 550
    TRAIL_LEN = 18
    COAST_FRAMES = 6
    MOTION_T = 12
    SPEED_MIN_KMH, SPEED_MAX_KMH = 15.0, 140.0
    SPEED_EMA = 0.35
    EVENT_TTL = 0.45
    FAST_PX_S = 300.0
    CONFIRM_PX_S = 250.0           # a track is shown only once it has moved at ball speed
    BALL_TO_PLAYER = 0.10          # max ball diameter / height of the nearest player (ball 6.5 cm vs ~1.8 m)

    def __init__(self):
        self.reset()

    def reset(self):
        """Full reset including court geometry (new calibration / new scene)."""
        self.H = None
        self.roi = None
        self.roi_rect = None
        self.frames = 0
        self.frames_visible = 0
        self.frames_measured = 0
        self.total_shots = 0
        self.reset_track()

    def reset_track(self):
        """Forget the current ball track and motion history, keep the court geometry and session counters."""
        self.kf = None
        self._grays = deque(maxlen=2)
        self.last_ts = None
        self.hits = 0
        self.misses = 0
        self.last_seen = 0.0
        self.pos = None
        self.court_pos = None
        self.trail = deque(maxlen=self.TRAIL_LEN)
        self.court_trail = deque(maxlen=25)
        self.speed_kmh = 0.0
        self._spd = deque(maxlen=5)
        self._spd_ema = None
        self.vel = (0.0, 0.0)
        self.shots = 0
        self._last_shot = 0.0
        self.visible = False
        self.coasting = False
        self.slow_since = None
        self.suppress = []                 # (x, y, until_ts) spots where slow false tracks were dropped
        self._hist = deque(maxlen=12)      # measured (ts, x, y) of the current track
        self._last_event_t = 0.0
        self._track_end = None             # (ts, x, y, vx, vy) when the previous track was dropped
        self.events = deque(maxlen=8)      # (x, y, ts, kind) kind = hit | bounce | wall
        self.hitter = None
        self._vmax = 0.0

    # ------------------------------------------------------------------ geometry
    def set_geometry(self, H_i2c, frame_shape):
        """Court-volume ROI: the court polygon plus the space above it (lobs / smashes)."""
        self.H = H_i2c
        h, w = frame_shape[:2]
        self.roi = None
        self.roi_rect = None
        self._grays.clear()
        if H_i2c is None:
            return
        try:
            H_c2i = CG.invert_homography_c2i(H_i2c)
            if H_c2i is None:
                return
            pts_list = []
            for a, b in (((0.0, 0.0), (10.0, 0.0)), ((10.0, 0.0), (10.0, 20.0)),
                         ((10.0, 20.0), (0.0, 20.0)), ((0.0, 20.0), (0.0, 0.0))):
                seg = CG.sample_clipped_segment(H_c2i, a, b, frame_shape, steps=24)
                if seg is not None:
                    pts_list.append(seg)
            if not pts_list:
                return
            pts = np.vstack(pts_list)
            pts[:, 0] = np.clip(pts[:, 0], 0.01 * w, 0.99 * w)
            pts[:, 1] = np.clip(pts[:, 1], 0.0, 1.05 * h)
            up = pts.copy()
            up[:, 1] -= 0.38 * h
            hull = cv2.convexHull(np.vstack([pts, up]).astype(np.int32))
            roi = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(roi, hull, 255)
            x, y, bw, bh = cv2.boundingRect(hull)
            x, y = max(0, x), max(0, y)
            self.roi = roi
            self.roi_rect = (x, y, min(w, x + bw), min(h, y + bh))
        except Exception:
            self.roi = None
            self.roi_rect = None

    def _px_to_m_scale(self, x, y):
        if self.H is None:
            return None
        try:
            a, b = cv2.perspectiveTransform(np.array([[[x, y], [x + 1.0, y]]], np.float32), self.H)[0]
            return float(np.linalg.norm(b - a))
        except cv2.error:
            return None

    # ------------------------------------------------------------------ filters
    def _kf_new(self, x, y):
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kf.processNoiseCov = np.diag([2.0, 2.0, 900.0, 900.0]).astype(np.float32)
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 2.0
        kf.errorCovPost = np.diag([4.0, 4.0, 1e5, 1e5]).astype(np.float32)
        kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        kf.statePre = kf.statePost.copy()                 # correct() right after init needs a valid prior
        kf.errorCovPre = kf.errorCovPost.copy()
        return kf

    def _parabola_pred(self, ts):
        """Ballistic prediction: x(t) linear, y(t) quadratic over the last <= 8 measurements (<= 0.3 s)."""
        pts = [p for p in self._hist if ts - p[0] < 0.3][-8:]
        if len(pts) < 5:
            return None
        a = np.array(pts, np.float64)
        t = a[:, 0] - a[-1, 0]
        try:
            px = np.polyfit(t, a[:, 1], 1)
            py = np.polyfit(t, a[:, 2], 2)
        except (np.linalg.LinAlgError, ValueError):
            return None
        res = np.sqrt(np.mean((np.polyval(px, t) - a[:, 1]) ** 2 + (np.polyval(py, t) - a[:, 2]) ** 2))
        if res > 6.0:
            return None                                   # not a clean flight (bounce / hit inside the window)
        tq = ts - a[-1, 0]
        return float(np.polyval(px, tq)), float(np.polyval(py, tq))

    # ------------------------------------------------------------------ player logic
    @staticmethod
    def _kp(kps, i, j):
        if kps is None or i >= len(kps) or kps[i] is None:
            return None
        k = kps[i]
        if len(k) <= j or k[j, 2] < KCONF:
            return None
        return float(k[j, 0]), float(k[j, 1])

    def _near_wrist(self, x, y, boxes, kps, frac=0.45):
        for i, b in enumerate(boxes):
            bh = max(1.0, b[3] - b[1])
            for j in (9, 10):
                p = self._kp(kps, i, j)
                if p is not None and math.hypot(x - p[0], y - p[1]) < min(frac * bh, 140.0):
                    return True
        return False

    def _near_player(self, x, y, boxes, kps):
        """Ball at a racket: near a wrist keypoint, or (no keypoints) in the upper part of a widened player box."""
        if self._near_wrist(x, y, boxes, kps, 0.55):
            return True
        for i, b in enumerate(boxes):
            if kps is not None and i < len(kps) and kps[i] is not None:
                continue
            bw, bh = b[2] - b[0], b[3] - b[1]
            if b[0] - 0.4 * bw <= x <= b[2] + 0.4 * bw and b[1] - 0.35 * bh <= y <= b[1] + 0.75 * bh:
                return True
        return False

    def _veto(self, x, y, near, fast, strong, boxes, kps, outsiders):
        """True when a candidate sits on a player (shoe / body) and is not explained by a racket hit or a
        fast ongoing track."""
        free = near and fast
        if any(b[0] - 6 <= x <= b[2] + 6 and b[1] - 6 <= y <= b[3] + 6 for b in outsiders):
            return True                                   # spectator / neighbouring-court player
        for i, b in enumerate(boxes):
            bw, bh = max(1.0, b[2] - b[0]), max(1.0, b[3] - b[1])
            ankles = [p for p in (self._kp(kps, i, 15), self._kp(kps, i, 16)) if p is not None]
            if ankles:
                r = max(10.0, 0.11 * bh)
                if any(math.hypot(x - p[0], y - p[1]) < r for p in ankles) and not free:
                    return True                           # neon shoes
            elif b[0] - 10 <= x <= b[2] + 10 and b[3] - 0.30 * bh <= y <= b[3] + 10 and not free:
                return True
            if b[0] <= x <= b[2] and b[1] <= y <= b[3]:
                if free:
                    continue
                if strong and self._near_wrist(x, y, [b], [kps[i]] if kps is not None and i < len(kps) else None):
                    continue                              # ball entering / leaving the racket at speed
                return True                               # clothing / skin / racket frame on the body
        return False

    # ------------------------------------------------------------------ main step
    def update(self, frame, ts, player_boxes=(), yolo_balls=(), keypoints=None, det_ts=None):
        self.frames += 1
        h, w = frame.shape[:2]
        boxes = [np.asarray(b, np.float32) for b in (player_boxes if player_boxes is not None else [])]
        kps = keypoints if keypoints is not None and len(keypoints) == len(boxes) else None
        if self.roi_rect is not None and self.roi is not None and self.roi.shape == (h, w):
            x0, y0, x1, y1 = self.roi_rect
        else:
            x0, y0, x1, y1 = 0, 0, w, h
        crop = frame[y0:y1, x0:x1]
        gray = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        dt = 1 / 30.0 if self.last_ts is None else float(np.clip(ts - self.last_ts, 1 / 240.0, 0.5))
        self.last_ts = ts

        tracking = self.kf is not None
        preds = []
        if tracking:
            self.kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            p = self.kf.predict()
            preds.append((float(p[0, 0]), float(p[1, 0])))
            par = self._parabola_pred(ts)
            if par is not None:
                preds.append(par)
        gate = min(260.0, 30.0 + 1600.0 * dt) * (1.0 + 0.35 * self.misses)
        fast = self._track_speed() > self.FAST_PX_S

        def pred_dist(cx, cy):
            return min((math.hypot(cx - px, cy - py) for px, py in preds), default=1e9)

        outsiders = []
        if self.H is not None and boxes:
            try:
                feet = np.array([[[(b[0] + b[2]) / 2.0, b[3]] for b in boxes]], np.float32)
                for b, g in zip(boxes, cv2.perspectiveTransform(feet, self.H)[0]):
                    if not (-0.6 <= g[0] <= COURT_WIDTH_M + 0.6 and -0.6 <= g[1] <= COURT_LENGTH_M + 0.6):
                        outsiders.append(b)
            except cv2.error:
                pass

        cands = []
        if len(self._grays) == 2 and self._grays[0].shape == gray.shape and self._grays[1].shape == gray.shape:
            g1, g2 = self._grays[0], self._grays[1]
            _, m1 = cv2.threshold(cv2.absdiff(gray, g1), self.MOTION_T, 255, cv2.THRESH_BINARY)
            _, m2 = cv2.threshold(cv2.absdiff(gray, g2), self.MOTION_T, 255, cv2.THRESH_BINARY)
            motion3 = cv2.bitwise_and(m1, m2)
            md = cv2.dilate(motion3, None, iterations=2)
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            color = cv2.inRange(hsv, self.HSV_LO, self.HSV_HI)
            streak = cv2.bitwise_and(cv2.inRange(hsv, self.STREAK_LO, self.STREAK_HI), motion3)
            cmask = cv2.bitwise_or(cv2.bitwise_and(color, md), streak)
            roi_c = self.roi[y0:y1, x0:x1] if self.roi_rect is not None and self.roi is not None \
                and self.roi.shape == (h, w) else None
            if roi_c is not None:
                cmask = cv2.bitwise_and(cmask, roi_c)
            cands += self._blobs(cmask, color, motion3, x0, y0, pred_dist, gate, fast, boxes, kps, outsiders, ts)
            # Rescue: slow ball near the apex / half-hidden -> colour-only blob right on the predicted path
            if tracking and not any(c[5] for c in cands) and preds:
                px, py = preds[-1]
                r = int(gate * 0.6) + 6
                wx1, wy1 = int(max(0, px - x0 - r)), int(max(0, py - y0 - r))
                wx2, wy2 = int(min(x1 - x0, px - x0 + r)), int(min(y1 - y0, py - y0 + r))
                if wx2 - wx1 > 4 and wy2 - wy1 > 4:
                    win = color[wy1:wy2, wx1:wx2].copy()
                    if roi_c is not None:
                        win = cv2.bitwise_and(win, roi_c[wy1:wy2, wx1:wx2])
                    mwin = cv2.dilate(m1[wy1:wy2, wx1:wx2], None, iterations=1)
                    cands += [c[:6] + (True,) + c[7:] for c in
                              self._blobs(win, win, mwin, x0 + wx1, y0 + wy1, pred_dist, gate * 0.6, fast, boxes,
                                          kps, outsiders, ts, min_motion=0.05)]
        self._grays.appendleft(gray)
        fresh_yolo = det_ts is None or abs(ts - det_ts) < 0.02
        if fresh_yolo:
            for b in (yolo_balls if yolo_balls is not None else []):
                cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
                d = pred_dist(cx, cy)
                cands.append((cx, cy, 40.0, 0.8, 1.0, d < gate, False, False, False))

        # ---------------------------------------------------------- association
        meas = None
        if tracking:
            best = None
            for cx, cy, area, circ, mfrac, near, weak, streak, inbox in cands:
                d = pred_dist(cx, cy)
                if d >= gate:
                    continue
                score = d / gate + 0.25 * (1.0 - min(1.0, circ)) + (0.2 if weak else 0.0)
                if best is None or score < best[0]:
                    best = (score, cx, cy)
            if best:
                meas = (best[1], best[2])
        if meas is None and not (tracking and self.hits >= 3 and self.misses < self.COAST_FRAMES):
            # a new track only starts from a compact, clearly moving yellow blob away from the players
            strong = [c for c in cands if c[4] > 0.3 and not c[6] and not c[7] and not c[8] and c[3] >= 0.5]
            if strong:
                cx, cy, *_ = max(strong, key=lambda c: c[3] + 0.002 * min(c[2], 120) + 0.3 * c[4])
                if self.kf is not None and self.hits >= 3:
                    self._end_track(ts)
                self.kf, self.hits, self.misses, meas = self._kf_new(cx, cy), 0, 0, (cx, cy)
                self.trail.clear()
                self.court_trail.clear()
                self._hist.clear()
                self.slow_since = None
                self._vmax = 0.0
                tracking = True

        if meas is not None:
            st = self.kf.correct(np.array([[meas[0]], [meas[1]]], np.float32))
            self.hits += 1
            self.misses = 0
            self.coasting = False
            self.last_seen = ts
            self.pos = meas
            self.frames_measured += 1
            vx, vy = float(st[2, 0]), float(st[3, 0])
            self.vel = (vx, vy)
            self._hist.append((ts, meas[0], meas[1]))
            if self.hits >= 3:
                self._vmax = max(self._vmax, math.hypot(vx, vy))
            # a padel ball in play is never slow for long: drop tracks that crawl (shoes, debris, resting ball)
            if self.hits >= 3 and math.hypot(vx, vy) < 150:
                self.slow_since = self.slow_since or ts
            else:
                self.slow_since = None
            if self.slow_since and ts - self.slow_since > 0.4:
                self.suppress = [z for z in self.suppress if z[2] > ts][-30:] + [(meas[0], meas[1], ts + 10.0)]
                self.kf, self.hits, self.slow_since = None, 0, None
                self.trail.clear()
                self._hist.clear()
                self.visible = False
                return False
            self.trail.append((int(meas[0]), int(meas[1])))
            self._update_speed(meas, vx, vy, boxes)
            self._update_court(meas, boxes)
            if self.hits == 4:
                self._new_track_event(ts, boxes, kps)
            self._detect_event(ts, boxes, kps)
        elif self.kf is not None:
            self.misses += 1
            if self.misses > self.COAST_FRAMES or ts - self.last_seen > 0.35 or self.hits < 3:
                self._end_track(ts)
            else:                                         # coast on the prediction through a short occlusion
                st = self.kf.statePost
                self.pos = (float(st[0, 0]), float(st[1, 0]))
                self.coasting = True
                if 0 <= self.pos[0] < w and 0 <= self.pos[1] < h:
                    self.trail.append((int(self.pos[0]), int(self.pos[1])))
                    self._update_court(self.pos, boxes)
                else:
                    self._end_track(ts)
        if ts - self.last_seen > 4.0 and self.shots:
            self.shots = 0                                # ball gone for a while -> next rally
        self.visible = (self.kf is not None and self.hits >= 3 and self.misses <= self.COAST_FRAMES
                        and self.slow_since is None and self._vmax >= self.CONFIRM_PX_S)
        if self.visible:
            self.frames_visible += 1
        elif ts - self.last_seen > 0.5:
            self.trail.clear()
            self.coasting = False
        return self.visible

    def _blobs(self, mask, color, motion, ox, oy, pred_dist, gate, fast, boxes, kps, outsiders, ts, min_motion=0.25):
        out = []
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if not (self.AREA_MIN <= area <= self.AREA_MAX):
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            col = float(color[y:y + bh, x:x + bw].mean()) / 255.0
            streak = col < 0.25                          # desaturated motion-blur streak, not the yellow body
            elong = max(bw, bh) / max(1.0, min(bw, bh))
            if elong > (7.0 if streak else 3.5) + 1.0 / max(1, min(bw, bh)):
                continue
            per = cv2.arcLength(c, True)
            circ = 4 * math.pi * area / (per * per) if per > 0 else 0.0
            if not streak and area >= 25:
                if circ < 0.50:
                    continue
                hull_a = cv2.contourArea(cv2.convexHull(c))
                if hull_a > 0 and area / hull_a < 0.82:
                    continue                              # ragged shape: shoe / logo / clothing fragment
            if streak and (area < 14 or elong < 1.6):
                continue                                  # tiny white flicker / round white patch (caps, socks)
            mfrac = float(motion[y:y + bh, x:x + bw].mean()) / 255.0
            cx, cy = ox + x + bw / 2.0, oy + y + bh / 2.0
            near = pred_dist(cx, cy) < gate
            if mfrac < min_motion and not (near and mfrac > 0.08):
                continue
            if streak and not near:
                continue                                  # a blur streak can extend a track, never start one
            nb = self._nearest_box((cx, cy), boxes)
            inbox = False
            if nb is not None:
                bh = max(1.0, nb[1][3] - nb[1][1])
                if nb[0] < 0.5 * bh and not streak and math.sqrt(4.0 * area / math.pi) > max(7.0, self.BALL_TO_PLAYER * bh):
                    continue                              # too big for a ball at this player's depth (shirt / shoe)
                b = nb[1]
                bw = b[2] - b[0]
                inbox = (b[0] - 0.1 * bw <= cx <= b[2] + 0.1 * bw) and (b[1] - 0.05 * bh <= cy <= b[3] + 0.05 * bh)
            if any(math.hypot(cx - zx, cy - zy) < 28 and ts < zt for zx, zy, zt in self.suppress) \
                    and not (near and fast):
                continue                                  # learned static / slow false-positive spot
            if self._veto(cx, cy, near, fast, mfrac > 0.45, boxes, kps, outsiders):
                continue
            out.append((cx, cy, area, (0.55 if streak else circ), mfrac, near, False, streak, inbox))
        return out

    # ------------------------------------------------------------------ events / metrics
    def _end_track(self, ts):
        if self.kf is not None and self.hits >= 3 and self.pos is not None:
            self._track_end = (ts, self.pos[0], self.pos[1], self.vel[0], self.vel[1])
        self.kf, self.hits, self.misses = None, 0, 0
        self.coasting = False
        self._hist.clear()

    def _classify(self, x, y, v_in, v_out, boxes, kps):
        if self._near_player(x, y, boxes, kps):
            return "hit"
        return "bounce" if v_in[1] > 0 > v_out[1] else "wall"

    def _add_event(self, x, y, ts, kind):
        if ts - self._last_event_t < 0.25:
            return
        self._last_event_t = ts
        self.events.append((float(x), float(y), ts, kind))
        if kind == "hit":
            self._register_shot(ts)

    def _detect_event(self, ts, boxes, kps):
        """Direction reversal inside one track: dot(v_prev, v_curr) < 0 (or a > ~70 deg turn)."""
        if len(self._hist) < 7:
            return
        a = np.array(list(self._hist)[-7:], np.float64)
        dt1, dt2 = a[3, 0] - a[0, 0], a[6, 0] - a[3, 0]
        if dt1 <= 0 or dt2 <= 0:
            return
        v1, v2 = (a[3, 1:] - a[0, 1:]) / dt1, (a[6, 1:] - a[3, 1:]) / dt2
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if min(n1, n2) < 120:
            return
        if float(np.dot(v1, v2)) / (n1 * n2) > 0.35:
            return
        x, y = a[3, 1], a[3, 2]
        self._add_event(x, y, ts, self._classify(x, y, v1, v2, boxes, kps))
        for _ in range(3):                                # don't re-detect the same turn
            self._hist.popleft()

    def _new_track_event(self, ts, boxes, kps):
        """Tracks usually break at racket contact: a new track leaving the old end point the other way = impact."""
        te = self._track_end
        if te is None or ts - te[0] > 0.6 or len(self._hist) < 2:
            return
        t0, x0, y0 = self._hist[0]
        if math.hypot(x0 - te[1], y0 - te[2]) > 180:
            return
        v_in = np.array([te[3], te[4]])
        v_out = np.array(self.vel)
        n1, n2 = np.linalg.norm(v_in), np.linalg.norm(v_out)
        if min(n1, n2) < 120 or float(np.dot(v_in, v_out)) / (n1 * n2) > 0.35:
            return
        self._add_event(x0, y0, ts, self._classify(x0, y0, v_in, v_out, boxes, kps))
        self._track_end = None

    def _nearest_box(self, pt, boxes):
        best = None
        for b in boxes:
            dx = max(b[0] - pt[0], 0, pt[0] - b[2])
            dy = max(b[1] - pt[1], 0, pt[1] - b[3])
            d = math.hypot(dx, dy)
            if best is None or d < best[0]:
                best = (d, b)
        return best

    def _update_speed(self, pt, vx, vy, boxes):
        # Ball height is unknown with one camera: use the ground-plane scale (m/px) at the depth of the nearest
        # player (the ball is almost always between / near players) -> perspective-correct court speed.
        if self.hits < 3:
            return
        nb = self._nearest_box(pt, boxes)
        row = nb[1][3] if nb is not None else pt[1]
        scale = self._px_to_m_scale(pt[0], row)
        if scale is None:
            return
        kmh = math.hypot(vx, vy) * scale * 3.6
        if kmh > 200:
            return                                        # Kalman transient right after an impact
        self._spd.append(kmh)
        med = float(np.median(self._spd))
        self._spd_ema = med if self._spd_ema is None else self.SPEED_EMA * med + (1 - self.SPEED_EMA) * self._spd_ema
        self.speed_kmh = float(np.clip(self._spd_ema, self.SPEED_MIN_KMH, self.SPEED_MAX_KMH))

    def _update_court(self, pt, boxes):
        if self.H is None:
            return
        try:
            g = cv2.perspectiveTransform(np.array([[[pt[0], pt[1]]]], np.float32), self.H)[0][0]
            if not (0 <= g[0] <= COURT_WIDTH_M and 0 <= g[1] <= COURT_LENGTH_M):
                nb = self._nearest_box(pt, boxes)       # in the air: lateral position at the hitter's depth
                if nb is None:
                    return
                g = cv2.perspectiveTransform(np.array([[[pt[0], nb[1][3]]]], np.float32), self.H)[0][0]
            new = (float(np.clip(g[0], 0, COURT_WIDTH_M)), float(np.clip(g[1], 0, COURT_LENGTH_M)))
            if self.court_trail and self.court_pos is not None:   # EMA: the in-air depth guess must not zig-zag
                new = (0.5 * new[0] + 0.5 * self.court_pos[0], 0.5 * new[1] + 0.5 * self.court_pos[1])
            self.court_pos = new
            self.court_trail.append(self.court_pos)
        except cv2.error:
            pass

    def _track_speed(self):
        return math.hypot(*self.vel) if self.kf is not None else 0.0

    def _register_shot(self, ts):
        if ts - self._last_shot > 0.4:
            self.shots += 1
            self.total_shots += 1
            self._last_shot = ts

    def recent_events(self, now):
        return [e for e in self.events if 0 <= now - e[2] < self.EVENT_TTL]

    def state(self, now=None):
        return {"visible": bool(self.visible), "coasting": bool(self.coasting and self.visible),
                "image_px": None if self.pos is None or not self.visible else [round(self.pos[0], 1), round(self.pos[1], 1)],
                "court_m": None if self.court_pos is None or not self.visible else [round(self.court_pos[0], 2), round(self.court_pos[1], 2)],
                "court_trail_m": [[round(x, 2), round(y, 2)] for x, y in list(self.court_trail)[-15:]] if self.visible else [],
                "speed_kmh": round(self.speed_kmh, 1), "rally_shots": self.shots, "total_shots": self.total_shots,
                "visibility_pct": round(100.0 * self.frames_visible / max(1, self.frames), 1)}


class BallWorker:
    """Runs the ball tracker on every captured frame (up to max_fps) in its own thread, independent of the
    slower person detector, so fast balls keep a dense trajectory. boxes_getter() -> (boxes, yolo_balls[,
    keypoints[, detection_capture_ts]])."""

    def __init__(self, cam_getter, ball, boxes_getter, min_height=720, max_fps=60.0, rotation_getter=None):
        self.cam_getter, self.ball, self.boxes_getter = cam_getter, ball, boxes_getter
        self.min_height, self.period = min_height, 1.0 / max_fps
        self.rotation_getter = rotation_getter or (lambda: 0)
        self.running = True
        self.rate = 0.0
        self.errors = 0
        self.thread = threading.Thread(target=self._loop, name="PadelIQ-ball", daemon=True)
        self.thread.start()

    def _loop(self):
        last_fid, last_cam, n, t0 = -1, None, 0, time.time()
        while self.running:
            try:
                cam = self.cam_getter()
                if cam is None:
                    time.sleep(0.05)
                    continue
                if cam is not last_cam:
                    last_cam, last_fid = cam, -1
                    self.ball.reset_track()               # keep the court geometry (restored on F12 swaps)
                frame, fid, ts = cam_wait_latest(cam, last_fid, 0.05)
                if frame is None or fid == last_fid:
                    continue
                last_fid = fid
                canvas = prepare_canvas(frame, self.min_height, self.rotation_getter())
                got = self.boxes_getter()
                boxes, yolo_balls = got[0], got[1]
                kps = got[2] if len(got) > 2 else None
                det_ts = got[3] if len(got) > 3 else None
                self.ball.update(canvas, ts, boxes, yolo_balls, keypoints=kps, det_ts=det_ts)
                n += 1
                if time.time() - t0 >= 2.0:
                    self.rate, n, t0 = n / (time.time() - t0), 0, time.time()
                time.sleep(max(0.0, self.period - 0.002))
            except Exception as exc:
                self.errors += 1
                if self.errors <= 3:
                    log(f"Ball tracker error (recovered): {exc!r}")
                time.sleep(0.05)

    def stop(self):
        self.running = False


# =====================================================================
# 6. AI coach — 2 rotating tactical suggestions per active player
# =====================================================================
COACH_ROTATE_S = 3.0


def coach_tip_pool(p):
    """Ordered (most important first) list of tips derived from live metrics + court position."""
    tips = []
    x, y = (float(p.smoothed_pos_m[0]), float(p.smoothed_pos_m[1])) if p.smoothed_pos_m is not None else (5.0, 5.0)
    dist_to_net = abs(y - NET_Y_M)

    if p.stability_pct < 75:
        tips.append("Widen base & split-step before changing direction")
    elif p.stability_pct < 85:
        tips.append("Flex knees more - stay low for balance")
    else:
        tips.append("Solid balance - keep compact ready stance")

    if p.instant_speed_ms > 4.0:
        tips.append("Explosive move - reset to ready position fast")
    elif p.instant_speed_ms < 0.6:
        tips.append("Stay on toes - add small adjustment steps")
    else:
        tips.append("Good tempo - shadow the ball with your partner")

    if p.coverage_pct < 25:
        tips.append("Expand lateral coverage - cheat toward middle")
    elif p.coverage_pct > 60:
        tips.append("High coverage - save energy between points")

    if dist_to_net < 3.5:
        tips.append("At net: paddle up, punch volleys deep")
    elif dist_to_net > 7.5:
        tips.append("Back court: use the glass, lob to retake net")
    else:
        tips.append("No-man's-land: close to net or drop to glass")

    if x < 1.5 or x > COURT_WIDTH_M - 1.5:
        tips.append("Too wide - recover toward the centre line")
    else:
        tips.append("Call middle balls early with your partner")
    return tips


def coach_suggestions(p, now):
    pool = coach_tip_pool(p)
    k = int(now // COACH_ROTATE_S) + p.slot_id
    n = len(pool)
    return [pool[(2 * k) % n], pool[(2 * k + 1) % n]]


# =====================================================================
# 7. Live HUD rendering (resolution independent)
# =====================================================================
def hud_scale(frame):
    h, w = frame.shape[:2]
    return max(0.5, min(h / BASE_HUD_HEIGHT, w / 960.0))


def put_text(frame, text, org, scale, color, thickness=1, s=1.0):
    th = max(1, int(round(thickness * max(1.0, s * 0.9))))
    cv2.putText(frame, text, (int(org[0]), int(org[1])), FONT, scale, color, th, cv2.LINE_AA)


def text_width(text, scale, thickness=1):
    return cv2.getTextSize(text, FONT, scale, thickness)[0][0]


def fit_text(text, scale, thickness, max_w):
    if text_width(text, scale, thickness) <= max_w:
        return text
    while len(text) > 4 and text_width(text + "...", scale, thickness) > max_w:
        text = text[:-1]
    return text.rstrip() + "..."


# ---------------------------------------------------------------- HUD compositing primitives
# Translucency is only ever blended on the small ROI slice under a panel / glow (never a full 1280x720 copy),
# and panel chrome + text are pre-rendered patches: re-rendered only when their (10 Hz-throttled) content changes,
# while boxes, skeleton trails, minimap dots and the ball comet are drawn fresh on every frame.
HUD_TEXT_HZ = 10.0
_HUD_SLOW = {}                      # key -> (t, value)
_HUD_PATCH = {}                     # key -> (signature, img, mask, bg_img)


def hud_slow(key, now, fn, hz=HUD_TEXT_HZ):
    """fn() re-evaluated at most `hz` times per second (readable numbers, far fewer text renders)."""
    ent = _HUD_SLOW.get(key)
    if ent is None or now - ent[0] >= 1.0 / hz or now < ent[0]:
        ent = (now, fn())
        _HUD_SLOW[key] = ent
    return ent[1]


def hud_patch(key, sig, size, draw, bg, border=None):
    """Pre-rendered opaque panel content: draw(img) paints on a solid `bg` canvas of `size` (w, h).
    Cached until `sig` / size / colours change. Returns (img, mask_of_content, solid_bg)."""
    w, h = max(1, int(size[0])), max(1, int(size[1]))
    full_sig = (sig, w, h, tuple(bg), None if border is None else tuple(border))
    ent = _HUD_PATCH.get(key)
    if ent is not None and ent[0] == full_sig:
        return ent[1:]
    img = np.empty((h, w, 3), np.uint8)
    img[:] = bg
    draw(img)
    if border is not None:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), border, 1, cv2.LINE_AA)
    bg_img = np.empty_like(img)
    bg_img[:] = bg
    mask = (cv2.absdiff(img, bg_img).max(axis=2) > 0)[:, :, None]
    _HUD_PATCH[key] = (full_sig, img, mask, bg_img)
    return img, mask, bg_img


def composite_patch(frame, x1, y1, patch, alpha):
    """roi = alpha * bg + (1 - alpha) * roi on the panel slice only, then the opaque text / line pixels."""
    img, mask, bg_img = patch
    ph, pw = img.shape[:2]
    H, W = frame.shape[:2]
    x1, y1 = int(x1), int(y1)
    fx1, fy1, fx2, fy2 = max(0, x1), max(0, y1), min(W, x1 + pw), min(H, y1 + ph)
    if fx2 <= fx1 or fy2 <= fy1:
        return
    sx1, sy1 = fx1 - x1, fy1 - y1
    sx2, sy2 = sx1 + (fx2 - fx1), sy1 + (fy2 - fy1)
    roi = frame[fy1:fy2, fx1:fx2]
    cv2.addWeighted(roi, 1.0 - alpha, bg_img[sy1:sy2, sx1:sx2], alpha, 0, dst=roi)
    np.copyto(roi, img[sy1:sy2, sx1:sx2], where=mask[sy1:sy2, sx1:sx2])


def blend_roi(frame, rect, alpha, draw):
    """Translucent effect on a cropped slice: draw(layer, ox, oy) paints on a copy of frame[rect] using
    frame coordinates minus (ox, oy); the slice is then blended back in place."""
    H, W = frame.shape[:2]
    x1, y1 = max(0, int(rect[0])), max(0, int(rect[1]))
    x2, y2 = min(W, int(rect[2])), min(H, int(rect[3]))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    layer = roi.copy()
    draw(layer, x1, y1)
    cv2.addWeighted(layer, alpha, roi, 1.0 - alpha, 0, dst=roi)


def draw_glass_box(frame, x1, y1, x2, y2, bg=(18, 18, 18), alpha=0.68, border=None):
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = np.empty_like(roi)
    overlay[:] = bg
    cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0, dst=roi)
    if border is not None:
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), border, 1, cv2.LINE_AA)


def _rects_overlap(a, b):
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _overlap_area(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def draw_court_overlay(frame, corners_px, H, s):
    if corners_px is None:
        return
    H_use = H if H is not None else compute_homography(corners_px)
    H_c2i = CG.invert_homography_c2i(H_use) if H_use is not None else None
    if H_c2i is None:
        return

    W, L = COURT_WIDTH_M, COURT_LENGTH_M
    segs = [
        ((0, 0), (W, 0), (80, 255, 80), 2),
        ((W, 0), (W, L), (80, 255, 80), 2),
        ((W, L), (0, L), (80, 255, 80), 2),
        ((0, L), (0, 0), (80, 255, 80), 2),
        ((0, NET_Y_M), (W, NET_Y_M), (255, 255, 255), 2),
        ((0, SERVICE_LINE_NEAR_M), (W, SERVICE_LINE_NEAR_M), (80, 255, 80), 1),
        ((0, SERVICE_LINE_FAR_M), (W, SERVICE_LINE_FAR_M), (80, 255, 80), 1),
        ((W / 2, SERVICE_LINE_NEAR_M), (W / 2, SERVICE_LINE_FAR_M), (80, 255, 80), 1),
    ]
    for a, b, color, th in segs:
        pts = CG.sample_clipped_segment(H_c2i, a, b, frame.shape)
        if pts is not None:
            cv2.polylines(frame, [pts.astype(np.int32)], False, color, max(1, int(round(th * s))), cv2.LINE_AA)


# ---------------------------------------------------------------- ball
BALL_TRAIL_CYAN = np.array((255, 255, 0), np.float32)     # BGR, tail
BALL_TRAIL_GOLD = np.array((0, 200, 255), np.float32)     # BGR, head


def _draw_ball_icon(img, cx, cy, r):
    """Tiny padel-ball pictogram (OpenCV fonts have no emoji)."""
    cv2.circle(img, (cx, cy), r, (40, 240, 230), -1, cv2.LINE_AA)
    cv2.ellipse(img, (cx - r, cy), (r, r), 0, -55, 55, (245, 245, 245), 1, cv2.LINE_AA)
    cv2.ellipse(img, (cx + r, cy), (r, r), 0, 125, 235, (245, 245, 245), 1, cv2.LINE_AA)


def draw_ball_impacts(frame, ball, now, s):
    """Expanding impact rings on hits (white / gold) and bounces / glass (cyan)."""
    if ball is None or not hasattr(ball, "recent_events"):
        return
    for x, y, t, kind in ball.recent_events(now):
        a = (now - t) / ball.EVENT_TTL
        r = int((10 + 34 * a) * s)
        col = (80, 230, 255) if kind == "hit" else (255, 230, 60)
        th = max(1, int(round((3.0 - 2.0 * a) * s)))
        cx, cy = int(x), int(y)
        blend_roi(frame, (cx - r - th - 2, cy - r - th - 2, cx + r + th + 3, cy + r + th + 3), max(0.15, 1.0 - a),
                  lambda L, ox, oy, cx=cx, cy=cy, r=r, col=col, th=th:
                  cv2.circle(L, (cx - ox, cy - oy), r, col, th, cv2.LINE_AA))


def draw_ball_hud(frame, ball, now, s, obstacles=()):
    """Neon target ring, 18-frame tapering cyan->gold comet trail, impact rings and a non-overlapping
    'BALL: xx km/h | Shots: N' tag. Drawn after every other layer."""
    draw_ball_impacts(frame, ball, now, s)
    if ball is None or not ball.visible or ball.pos is None:
        return
    tr = list(ball.trail)
    n = len(tr)
    for i in range(1, n):
        a = i / max(1, n - 1)                                     # 0 = oldest -> 1 = newest
        col = tuple(int(v) for v in BALL_TRAIL_CYAN * (1 - a) + BALL_TRAIL_GOLD * a)
        cv2.line(frame, tr[i - 1], tr[i], col, max(1, int(round((1 + 5 * a * a) * s))), cv2.LINE_AA)
    cx, cy = int(ball.pos[0]), int(ball.pos[1])
    r = max(8, int(11 * s))
    coasting = bool(getattr(ball, "coasting", False))
    gr, gth = int(r * 1.9), max(2, int(5 * s))
    blend_roi(frame, (cx - gr - gth - 2, cy - gr - gth - 2, cx + gr + gth + 3, cy + gr + gth + 3),
              0.25 if coasting else 0.40,
              lambda L, ox, oy: cv2.circle(L, (cx - ox, cy - oy), gr, (0, 255, 255), gth, cv2.LINE_AA))
    pulse = 1.0 + 0.15 * math.sin(now * 10)
    cv2.circle(frame, (cx, cy), int(r * pulse), (0, 255, 255), max(2, int(2 * s)), cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), int(r * 1.45), (255, 255, 0), max(1, int(1 * s)), cv2.LINE_AA)
    for ang in (0, 90, 180, 270):                                 # target ticks
        ux, uy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        cv2.line(frame, (int(cx + ux * r * 1.5), int(cy + uy * r * 1.5)), (int(cx + ux * r * 2.1), int(cy + uy * r * 2.1)),
                 (255, 255, 0), max(1, int(2 * s)), cv2.LINE_AA)
    label = hud_slow("ball_label", now, lambda: f"BALL: {ball.speed_kmh:.0f} km/h | Shots: {ball.shots}")
    fs = 0.48 * s
    (tw, th), _ = cv2.getTextSize(label, FONT, fs, 1)
    icon = int(th * 0.75) + 1
    h, w = frame.shape[:2]
    tw2, th2 = tw + 2 * icon + int(20 * s), th + int(12 * s)
    off = int(r * 2.3)
    best = None
    for ox, oy in ((off, -off - th2), (-off - tw2, -off - th2), (off, off), (-off - tw2, off)):
        bx = int(np.clip(cx + ox, 2, w - tw2 - 2))
        by = int(np.clip(cy + oy, 2, h - th2 - 2))
        rect = (bx, by, bx + tw2, by + th2)
        cost = sum(_overlap_area(rect, o) for o in obstacles) + 0.5 * _overlap_area(rect, (cx - r, cy - r, cx + r, cy + r))
        if best is None or cost < best[0]:
            best = (cost, bx, by)
        if cost == 0:
            break
    _, bx, by = best

    def paint(im):
        _draw_ball_icon(im, int(7 * s) + icon, th2 // 2, icon)
        put_text(im, label, (int(12 * s) + 2 * icon, th + int(6 * s)), fs, (0, 255, 255), 1, s)
    composite_patch(frame, bx, by, hud_patch("ball_label", label, (tw2, th2), paint, (10, 10, 10), (0, 255, 255)), 0.82)


# ---------------------------------------------------------------- calibration review
def draw_calibration_review(frame, review, now, s, web=False):
    """Glowing court model + 6 draggable handles (TL, TR, BR, BL, NET-L, NET-R) + lock banner."""
    if review is None or review.state in ("idle", "confirmed"):
        return
    h, w = frame.shape[:2]
    with review.lock:
        state, corners = review.state, None if review.corners is None else review.corners.copy()
        conf = review.info.get("confidence_pct")
        ang = review.info.get("angle") or "Baseline"
        left = review.seconds_left(now)
        touched, mode = review.touched, review.manual_mode
        handles = None if review.handles_img is None else review.handles_img.copy()
        hlabels = list(getattr(review, "handle_labels", CG.HANDLE_LABELS))
        clicked = list(review.clicked)
        ref = review.reference
    H = compute_homography(corners) if corners is not None else None
    H_c2i = CG.invert_homography_c2i(H) if H is not None else None
    if H_c2i is not None and state in ("review", "manual"):
        segs = [[(0, 0), (10, 0)], [(10, 0), (10, 20)], [(10, 20), (0, 20)], [(0, 20), (0, 0)],
                [(0, NET_Y_M), (10, NET_Y_M)], [(0, SERVICE_LINE_NEAR_M), (10, SERVICE_LINE_NEAR_M)],
                [(0, SERVICE_LINE_FAR_M), (10, SERVICE_LINE_FAR_M)],
                [(5, SERVICE_LINE_NEAR_M), (5, SERVICE_LINE_FAR_M)]]
        polys = []
        for idx_s, (a, b) in enumerate(segs):
            pts = CG.sample_clipped_segment(H_c2i, a, b, frame.shape, steps=44)
            if pts is not None:
                polys.append((idx_s, pts.astype(np.int32)))
        if polys:
            allp = np.vstack([p for _, p in polys])
            gw = max(4, int(9 * s))
            (gx1, gy1), (gx2, gy2) = allp.min(axis=0) - gw, allp.max(axis=0) + gw + 1

            def glow(L, ox, oy):
                off = np.array([ox, oy], np.int32)
                for _, ipts in polys:
                    cv2.polylines(L, [ipts - off], False, (255, 255, 0), gw, cv2.LINE_AA)
            blend_roi(frame, (gx1, gy1, gx2, gy2), 0.35, glow)
        for idx_s, ipts in polys:
            col = (255, 255, 255) if idx_s == 4 else ((80, 255, 80) if idx_s < 4 else (255, 255, 0))
            cv2.polylines(frame, [ipts], False, col, max(1, int(2 * s)), cv2.LINE_AA)
    if state == "manual":
        if mode == "drag" and handles is not None:
            for i, (x, y) in enumerate(handles):
                lbl = hlabels[i] if i < len(hlabels) else str(i + 1)
                is_net = lbl.startswith("NET")
                hcol = (255, 255, 255) if is_net else (0, 255, 255)
                rad = max(9, int(12 * s))
                cv2.circle(frame, (int(x), int(y)), rad, (12, 12, 12), -1, cv2.LINE_AA)
                cv2.circle(frame, (int(x), int(y)), rad, hcol, max(2, int(2.5 * s)), cv2.LINE_AA)
                tx = int(np.clip(x + rad + 5 * s, 6, w - 72 * s))
                ty = int(np.clip(y + 5 * s, 20 * s, h - 8))
                draw_glass_box(frame, tx - int(4 * s), ty - int(15 * s), tx + int(56 * s), ty + int(5 * s),
                               bg=(8, 8, 8), alpha=0.82, border=hcol)
                put_text(frame, lbl, (tx, ty), 0.44 * s, hcol, 1, s)
        for i, (x, y) in enumerate(clicked):
            cv2.circle(frame, (int(x), int(y)), max(5, int(7 * s)), (0, 255, 0), -1, cv2.LINE_AA)

    if state == "detecting":
        l1, l2, col = "Detecting court floor & vanishing geometry ...", "", (0, 255, 255)
    elif state == "review":
        cs = f"{conf:.0f}%" if conf is not None else "n/a"
        l1 = f"COURT LOCKED (Confidence: {cs} | Angle: {ang}) -- [ENTER] Confirm | [C] Drag Corners | [A] Re-Scan"
        l2 = "Confirm or drag handles in the dashboard" if web else "Live perspective lock verified on floor plane"
        if left is not None and not touched:
            l2 += f"   (auto-confirm in {left:.1f}s)"
        col = (80, 255, 140)
    elif state == "manual":
        l1 = "MANUAL CALIBRATION -- " + (
            "Drag TL, TR, BR, BL, NET-L, NET-R handles (live grid preview)"
            if mode == "drag" else f"Click 4 fresh corners TL, TR, BR, BL ({CALIBRATION_REFERENCE_LABELS[ref]})"
        )
        l2 = ("Drag handles on video, [N] 4 fresh clicks, or Confirm" if web else
              "[ENTER / C] Lock Court   [N] Click 4 Fresh Corners / Drag   [H] Reference   [A] Re-Scan   [Esc] Back")
        col = (0, 215, 255)
    else:
        l1 = "Waiting for visible Padel court floor in camera view ..."
        l2 = ("Scene-cut detector active -- or click 'Calibrate' to set corners" if web else
              "Scene-cut detector active -- press [C] for manual handles or [A] to Re-Scan")
        col = (60, 160, 255)
    fs1, fs2 = 0.56 * s, 0.44 * s
    bw = w - int(28 * s)
    l1 = fit_text(l1, fs1, 2, bw - int(32 * s))
    l2 = fit_text(l2, fs2, 1, bw - int(32 * s))
    bh = int((62 if l2 else 40) * s)

    def paint(im):
        put_text(im, l1, (int(14 * s), int(26 * s)), fs1, col, 2, s)
        if l2:
            put_text(im, l2, (int(14 * s), int(50 * s)), fs2, (235, 235, 235), 1, s)
    composite_patch(frame, int(14 * s), int(10 * s),
                    hud_patch("calib_banner", (l1, l2, col), (bw, bh), paint, (10, 10, 10), col), 0.88)


# ---------------------------------------------------------------- minimap
def draw_minimap(frame, slot_mgr, now, s, ball=None):
    h, w = frame.shape[:2]
    m = int(16 * s)
    iw = int(140 * s)
    ih = 2 * iw                          # 10 m x 20 m keeps a true 1:2 aspect
    pad = int(14 * s)
    title_h = int(24 * s)
    mw, mh = iw + 2 * pad, ih + 2 * pad + title_h
    mx1, my1 = w - mw - m, m
    rect = (mx1, my1, mx1 + mw, my1 + mh)
    lx1, ly1 = pad, pad + title_h        # inner court origin, panel-local
    line_th = max(1, int(round(1 * s)))

    def lpx(xm, ym, ox=0, oy=0):
        return (int(ox + lx1 + (np.clip(xm, 0, COURT_WIDTH_M) / COURT_WIDTH_M) * iw),
                int(oy + ly1 + (np.clip(ym, 0, COURT_LENGTH_M) / COURT_LENGTH_M) * ih))

    def chrome(im):                      # static court map: rendered once per size, then cached
        put_text(im, "2D COURT MAP", (pad, int(18 * s)), 0.45 * s, (255, 255, 255), 1, s)
        cv2.rectangle(im, (lx1, ly1), (lx1 + iw, ly1 + ih), (245, 245, 245), max(1, int(round(2 * s))), cv2.LINE_AA)
        for ym in (SERVICE_LINE_NEAR_M, SERVICE_LINE_FAR_M):
            cv2.line(im, lpx(0, ym), lpx(COURT_WIDTH_M, ym), (230, 230, 230), line_th, cv2.LINE_AA)
        cv2.line(im, lpx(COURT_WIDTH_M / 2, SERVICE_LINE_NEAR_M), lpx(COURT_WIDTH_M / 2, SERVICE_LINE_FAR_M),
                 (230, 230, 230), line_th, cv2.LINE_AA)
        cv2.line(im, lpx(0, NET_Y_M), lpx(COURT_WIDTH_M, NET_Y_M), (255, 255, 255),
                 max(2, int(round(3 * s))), cv2.LINE_AA)
    composite_patch(frame, mx1, my1, hud_patch("minimap", round(s, 3), (mw, mh), chrome, (160, 85, 30), (255, 255, 255)),
                    0.80)

    ix1, iy1 = mx1 + lx1, my1 + ly1

    def to_px(xm, ym):
        return lpx(xm, ym, mx1, my1)

    dot_r = max(3, int(6 * s))
    fs = 0.42 * s
    visible = []
    for sid, p in slot_mgr.players.items():
        if not p.meter_trail or (now - p.last_seen_time) > 1.5:
            continue
        color = SLOT_COLORS[sid]
        m_pts = [to_px(mm[0], mm[1]) for mm in p.meter_trail]
        for i in range(1, len(m_pts)):
            cv2.line(frame, m_pts[i - 1], m_pts[i], color, max(1, int(round(2 * s))), cv2.LINE_AA)
        visible.append((sid, color, m_pts[-1]))

    if visible:                           # one glow blend over the court area for all dots
        gr = int(dot_r * 2.2)

        def glows(L, ox, oy):
            for _, color, (cx, cy) in visible:
                cv2.circle(L, (cx - ox, cy - oy), gr, color, -1, cv2.LINE_AA)
        blend_roi(frame, (ix1 - gr - 2, iy1 - gr - 2, ix1 + iw + gr + 3, iy1 + ih + gr + 3), 0.35, glows)
    for sid, color, (cx, cy) in visible:
        cv2.circle(frame, (cx, cy), dot_r, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), dot_r + 1, (255, 255, 255), 1, cv2.LINE_AA)

    dot_rects = [(cx - dot_r - 1, cy - dot_r - 1, cx + dot_r + 1, cy + dot_r + 1) for _, _, (cx, cy) in visible]
    label_rects = []
    bounds = (ix1, iy1, ix1 + iw, iy1 + ih)
    for sid, color, (cx, cy) in visible:
        label = f"P{sid}"
        (lw, lh), _ = cv2.getTextSize(label, FONT, fs, 1)
        off = dot_r + int(4 * s)
        options = [(cx + off, cy + lh // 2), (cx - off - lw, cy + lh // 2),
                   (cx - lw // 2, cy - off), (cx - lw // 2, cy + off + lh),
                   (cx - off - lw, cy - off), (cx - off - lw, cy + off + lh),
                   (cx + off, cy - off), (cx + off, cy + off + lh)]
        best, best_cost = None, None
        for lx, ly in options:
            r = (lx, ly - lh, lx + lw, ly + 2)
            cost = sum(_overlap_area(r, o) for o in dot_rects + label_rects)
            cost += (lw * lh) * (r[0] < bounds[0] or r[2] > bounds[2] or r[1] < bounds[1] or r[3] > bounds[3])
            if best is None or cost < best_cost:
                best, best_cost = (lx, ly, r), cost
            if cost == 0:
                break
        lx, ly, r = best
        label_rects.append(r)
        put_text(frame, label, (lx, ly), fs, (255, 255, 255), 2, s)

    if ball is not None and ball.visible and ball.court_pos is not None:
        tr = [to_px(x, y) for x, y in list(ball.court_trail)[-12:]]
        for i in range(1, len(tr)):
            a = i / max(1, len(tr) - 1)
            cv2.line(frame, tr[i - 1], tr[i], (int(120 + 135 * a), 255, 255), max(1, int(round(1.5 * s))), cv2.LINE_AA)
        bx, by = to_px(*ball.court_pos)
        if len(tr) >= 3 and math.hypot(bx - tr[-3][0], by - tr[-3][1]) > 3:      # flight direction vector
            cv2.arrowedLine(frame, tr[-3], (bx, by), (0, 255, 255), max(1, int(round(1.5 * s))), cv2.LINE_AA,
                            tipLength=0.45)
        pr = int(max(3, 5 * s) * (1.0 + 0.35 * abs(math.sin(now * 6))))
        blend_roi(frame, (bx - 2 * pr - 2, by - 2 * pr - 2, bx + 2 * pr + 3, by + 2 * pr + 3), 0.55,
                  lambda L, ox, oy: cv2.circle(L, (bx - ox, by - oy), pr * 2, (0, 255, 255), -1, cv2.LINE_AA))
        cv2.circle(frame, (bx, by), pr, (225, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (bx, by), pr + 1, (0, 255, 255), max(1, int(round(1.5 * s))), cv2.LINE_AA)
    return rect


# ---------------------------------------------------------------- panels
def draw_coach_panel(frame, slot_mgr, now, s, fps_val, source_label, recording, rec_seconds, max_w):
    m = int(16 * s)
    x1, y1 = m, m
    pw = int(min(520 * s, max_w))
    lh = int(21 * s)
    inner_w = pw - int(24 * s)

    def content():
        active = [p for p in slot_mgr.players.values() if p.is_active(now)] if slot_mgr else []
        rows = tuple((p.slot_id, tuple(coach_suggestions(p, now))) for p in sorted(active, key=lambda q: q.slot_id))
        mm, ss = divmod(int(rec_seconds), 60)
        return f"PadelIQ LIVE | {fps_val:.1f} FPS | {source_label}", (f"REC {mm:02d}:{ss:02d}" if recording else ""), rows
    header, rec, rows = hud_slow("coach", now, content)
    nrows = sum(max(1, len(t)) for _, t in rows) if rows else 1
    ph = int(32 * s) + lh * 3 + int(8 * s) + nrows * lh + int(10 * s)
    rec_w = int(100 * s) if rec else 0
    tx = int(12 * s)

    def paint(im):
        y = int(26 * s)
        put_text(im, fit_text(header, 0.55 * s, 2, inner_w - rec_w), (tx, y), 0.55 * s, (0, 255, 180), 2, s)
        if rec:
            r = max(4, int(7 * s))
            put_text(im, rec, (pw - rec_w + 2 * r + int(5 * s), y), 0.48 * s, (60, 60, 255), 2, s)
        y += lh + int(2 * s)
        put_text(im, fit_text("[C] Calib  [L] Lock  [R] Rotate  [E] Shutter  [F] Full  [F12] Failover  [M] QR  [S] Rec  [Q] Quit",
                              0.42 * s, 1, inner_w), (tx, y), 0.42 * s, (200, 200, 200), 1, s)
        y += lh + int(6 * s)
        put_text(im, "AI Coach Suggestions:", (tx, y), 0.52 * s, (255, 255, 255), 2, s)
        y += lh
        if not rows:
            put_text(im, fit_text("Waiting for players inside calibrated zone...", 0.44 * s, 1, inner_w), (tx, y),
                     0.44 * s, (180, 180, 180), 1, s)
            return
        label_w = int(40 * s)
        for sid, tips in rows:
            color = SLOT_COLORS[sid]
            put_text(im, f"P{sid}", (tx, y), 0.46 * s, color, 2, s)
            for tip in tips or ("",):
                bx = tx + label_w
                if tip:
                    cv2.circle(im, (bx + int(3 * s), y - int(5 * s)), max(2, int(3 * s)), color, -1, cv2.LINE_AA)
                    put_text(im, fit_text(tip, 0.43 * s, 1, inner_w - label_w - int(14 * s)),
                             (bx + int(12 * s), y), 0.43 * s, (235, 235, 235), 1, s)
                y += lh
    composite_patch(frame, x1, y1, hud_patch("coach", (header, rec, rows), (pw, ph), paint, (18, 18, 18), (90, 90, 90)),
                    0.86)
    if rec:                                                    # pulsing REC dot animates every frame
        pulse = 0.5 + 0.5 * math.sin(now * 2 * math.pi * 1.2)
        r = max(4, int(7 * s))
        cv2.circle(frame, (x1 + pw - rec_w + r, y1 + int(21 * s)), int(r * (0.8 + 0.25 * pulse)),
                   (0, 0, int(140 + 115 * pulse)), -1, cv2.LINE_AA)
    return (x1, y1, x1 + pw, y1 + ph)


def draw_player_tracks(frame, slot_mgr, now, s):
    """Foot trails + bounding boxes (drawn under the HUD panels)."""
    for sid, p in slot_mgr.players.items():
        color = SLOT_COLORS[sid]
        if len(p.pixel_trail) > 1 and (now - p.last_seen_time) < 1.5:
            t_pts = list(p.pixel_trail)
            for i in range(1, len(t_pts)):
                cv2.line(frame, t_pts[i - 1], t_pts[i], color, max(1, int((i / len(t_pts)) * 3 * s)), cv2.LINE_AA)
        if p.bbox is not None and (now - p.last_seen_time) < 0.8:
            x1, y1, x2, y2 = map(int, p.display_bbox(now))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(1, int(round(2 * s))), cv2.LINE_AA)


def draw_player_cards(frame, slot_mgr, now, s, obstacles):
    h, w = frame.shape[:2]
    cw, ch = int(192 * s), int(100 * s)
    gap = int(8 * s)
    placed = []
    boxes = {sid: tuple(map(int, p.display_bbox(now))) for sid, p in slot_mgr.players.items()
             if p.bbox is not None and (now - p.last_seen_time) < 0.8}
    for sid, p in slot_mgr.players.items():
        color = SLOT_COLORS[sid]
        if sid not in boxes:
            continue
        x1, y1, x2, y2 = boxes[sid]
        others = [b for k, b in boxes.items() if k != sid]
        cx = (x1 + x2) // 2
        options = [
            (cx - cw // 2, y1 - ch - gap),      # above
            (cx - cw // 2, y2 + gap),           # below
            (x2 + gap, (y1 + y2) // 2 - ch // 2),   # right
            (x1 - cw - gap, (y1 + y2) // 2 - ch // 2),  # left
        ]
        best, best_cost = None, None
        for ox, oy in options:
            ox = int(np.clip(ox, 2, max(2, w - cw - 2)))
            oy = int(np.clip(oy, 2, max(2, h - ch - 2)))
            r = (ox, oy, ox + cw, oy + ch)
            cost = (sum(_overlap_area(r, o) for o in obstacles + placed)
                    + 0.6 * sum(_overlap_area(r, o) for o in others)
                    + 0.8 * _overlap_area(r, (x1, y1, x2, y2)))
            if best is None or cost < best_cost:
                best, best_cost = r, cost
            if cost == 0:
                break
        placed.append(best)
        lines = hud_slow(("card", sid), now, lambda p=p, sid=sid: (
            f"P{sid} (Coverage: {p.coverage_pct:.1f}%)",
            f"Distance: {p.distance_m:.1f}m",
            f"AvgSpeed: {p.avg_speed_ms:.2f}m/s",
            f"Active: {int(p.active_seconds_total)}s ({p.active_pct}%)",
            f"Stability: {int(p.stability_pct)}%",
        ))

        def paint(im, lines=lines, color=color):
            for idx, txt in enumerate(lines):
                ty = int(18 * s) + idx * int(17.5 * s)
                put_text(im, fit_text(txt, 0.42 * s, 1, cw - int(14 * s)), (int(8 * s), ty), 0.42 * s,
                         color if idx == 0 else (240, 240, 240), 2 if idx == 0 else 1, s)
        composite_patch(frame, best[0], best[1], hud_patch(("card", sid), lines, (cw, ch), paint, (15, 15, 15), color), 0.82)


def draw_compact_tags(frame, slot_mgr, now, s):
    """Small 'P1 | 3.2 m/s' tags above each box (web mode: full stats live in the dashboard)."""
    h, w = frame.shape[:2]
    placed = []
    fs = 0.5 * s
    for sid, p in slot_mgr.players.items():
        if p.bbox is None or (now - p.last_seen_time) >= 0.8:
            continue
        x1, y1, x2, y2 = map(int, p.display_bbox(now))
        color = SLOT_COLORS[sid]
        label = hud_slow(("tag", sid), now, lambda p=p, sid=sid: f"P{sid}  {p.instant_speed_ms:.1f} m/s")
        (tw, th), _ = cv2.getTextSize(label, FONT, fs, 1)
        bw, bh = tw + int(14 * s), th + int(12 * s)
        bx = int(np.clip((x1 + x2) // 2 - bw // 2, 2, max(2, w - bw - 2)))
        by = y1 - bh - int(6 * s)
        if by < 2:
            by = y2 + int(6 * s)
        by = int(np.clip(by, 2, max(2, h - bh - 2)))
        for _ in range(4):                                   # nudge away from other tags
            r = (bx, by, bx + bw, by + bh)
            if not any(_rects_overlap(r, o) for o in placed):
                break
            by = int(np.clip(by - bh - 2, 2, max(2, h - bh - 2))) if by > bh + 4 else by + bh + 2
        placed.append((bx, by, bx + bw, by + bh))

        def paint(im, label=label, color=color, bh=bh):
            cv2.rectangle(im, (0, 0), (max(3, int(4 * s)), bh - 1), color, -1)
            put_text(im, label, (int(9 * s), bh - int(7 * s)), fs, (245, 245, 245), 1, s)
        composite_patch(frame, bx, by, hud_patch(("tag", sid), label, (bw, bh), paint, (12, 12, 12), color), 0.82)


def draw_compact_header(frame, s, fps_val, source_label, now=None):
    now = time.time() if now is None else now
    text = hud_slow("compact_header", now, lambda: fit_text(
        f"PadelIQ LIVE  |  {fps_val:.0f} FPS AI  |  {source_label}", 0.5 * s, 1, frame.shape[1] * 0.6))
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.5 * s, 1)
    bw, bh = tw + int(30 * s), th + int(16 * s)
    x1, y1 = frame.shape[1] - bw - int(12 * s), int(12 * s)

    def paint(im):
        cv2.circle(im, (int(11 * s), bh // 2), max(3, int(4 * s)), (60, 220, 60), -1, cv2.LINE_AA)
        put_text(im, text, (int(22 * s), th + int(8 * s)), 0.5 * s, (235, 235, 235), 1, s)
    composite_patch(frame, x1, y1, hud_patch("compact_header", text, (bw, bh), paint, (10, 10, 10)), 0.82)


def draw_reconnect_banner(frame, now, s, detail=""):
    """No frame to hold yet (first connect): centred banner with a spinner."""
    h, w = frame.shape[:2]
    bw, bh = int(min(w - 20, 600 * s)), int(84 * s)
    x1, y1 = (w - bw) // 2, int(h * 0.42)
    draw_glass_box(frame, x1, y1, x1 + bw, y1 + bh, bg=(10, 10, 10), alpha=0.62, border=(0, 190, 255))
    r = int(16 * s)
    c = (x1 + int(32 * s), y1 + bh // 2)
    start = int((now * 360) % 360)
    cv2.ellipse(frame, c, (r, r), 0, start, start + 270, (0, 190, 255), max(2, int(3 * s)), cv2.LINE_AA)
    dots = "." * (1 + int(now * 2) % 3)
    tx = x1 + int(64 * s)
    put_text(frame, fit_text("Connecting to Live Stream" + dots, 0.75 * s, 2, bw - int(76 * s)),
             (tx, y1 + int(38 * s)), 0.75 * s, (0, 215, 255), 2, s)
    if detail:
        put_text(frame, fit_text(detail, 0.45 * s, 1, bw - int(76 * s)), (tx, y1 + int(66 * s)), 0.45 * s,
                 (210, 210, 210), 1, s)


def draw_relock_badge(frame, now, s, detail=""):
    """Subtle bottom-centre badge while the last clean frame is held and the camera re-initialises."""
    h, w = frame.shape[:2]
    title = "Re-locking Camera" + "." * (1 + int(now * 2) % 3)
    text = hud_slow("relock", now, lambda: fit_text(f"{title}   {detail}" if detail else title, 0.5 * s, 1, w * 0.6))
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.5 * s, 1)
    bw, bh = tw + int(44 * s), th + int(18 * s)
    x1, y1 = (w - bw) // 2, h - bh - int(18 * s)
    composite_patch(frame, x1, y1, hud_patch("relock", text, (bw, bh), lambda im: put_text(
        im, text, (int(34 * s), th + int(9 * s)), 0.5 * s, (0, 215, 255), 1, s), (10, 10, 10), (0, 190, 255)), 0.72)
    r = max(5, int(8 * s))
    start = int((now * 360) % 360)
    cv2.ellipse(frame, (x1 + int(17 * s), y1 + bh // 2), (r, r), 0, start, start + 270, (0, 190, 255),
                max(1, int(2 * s)), cv2.LINE_AA)


def draw_lock_badge(frame, s):
    """'CALIBRATION LOCKED' with a padlock pictogram, bottom-left."""
    h, w = frame.shape[:2]
    text = "CALIBRATION LOCKED"
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.46 * s, 1)
    bw, bh = tw + int(40 * s), th + int(16 * s)

    def paint(im):
        bx, by = int(10 * s), bh // 2
        cv2.ellipse(im, (bx + int(6 * s), by - int(2 * s)), (int(4 * s), int(5 * s)), 0, 180, 360, (80, 255, 140),
                    max(1, int(2 * s)), cv2.LINE_AA)
        cv2.rectangle(im, (bx, by - int(2 * s)), (bx + int(12 * s), by + int(7 * s)), (80, 255, 140), -1)
        put_text(im, text, (int(30 * s), th + int(8 * s)), 0.46 * s, (80, 255, 140), 1, s)
    composite_patch(frame, int(16 * s), h - bh - int(16 * s),
                    hud_patch("lock_badge", round(s, 3), (bw, bh), paint, (10, 10, 10), (80, 255, 140)), 0.78)


def draw_toast(frame, s, text):
    """Short operator feedback (exposure mode, lock, source swap) above the bottom edge."""
    if not text:
        return
    h, w = frame.shape[:2]
    text = fit_text(text, 0.6 * s, 2, w * 0.7)
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.6 * s, 2)
    bw, bh = tw + int(32 * s), th + int(22 * s)
    composite_patch(frame, (w - bw) // 2, h - bh - int(70 * s), hud_patch("toast", text, (bw, bh), lambda im: put_text(
        im, text, (int(16 * s), th + int(11 * s)), 0.6 * s, (255, 255, 255), 2, s), (20, 20, 20), (0, 255, 180)), 0.85)


def render_live_hud(frame, slot_mgr, corners_px, fps_val, H=None, source_label="LIVE",
                    recording=False, rec_seconds=0.0, reconnecting=False, reconnect_detail="",
                    hud_mode="full", ball=None, review=None, web=False, now=None,
                    held=False, locked=False, toast=""):
    """
    Draws the PadelIQ HUD in place. Works for any resolution (scales from 720p).
    hud_mode="full": coach panel + minimap + stat cards (desktop).
    hud_mode="compact": court, boxes and small speed tags (web dashboard shows the rest).
    held=True: the frame is the last clean one, held while the camera re-locks (subtle badge, no big banner).
    """
    now = time.time() if now is None else float(now)
    s = hud_scale(frame)
    h, w = frame.shape[:2]

    if slot_mgr is not None and corners_px is not None:
        draw_court_overlay(frame, corners_px, H if H is not None else slot_mgr.H, s)
    if slot_mgr is not None:
        draw_player_tracks(frame, slot_mgr, now, s)
    if hud_mode == "compact":
        if slot_mgr is not None:
            draw_compact_tags(frame, slot_mgr, now, s)
        draw_compact_header(frame, s, fps_val, source_label, now)
        draw_calibration_review(frame, review, now, s, web=web)
        draw_ball_hud(frame, ball, now, s)                    # ball on top of every other layer
        rects = {"coach": (0, 0, 0, 0), "minimap": (w, 0, w, 0)}
    else:
        if slot_mgr is not None:
            mini_rect = draw_minimap(frame, slot_mgr, now, s, ball=ball)
        else:
            mini_rect = (w, 0, w, 0)
        coach_rect = draw_coach_panel(frame, slot_mgr, now, s, fps_val, source_label, recording, rec_seconds,
                                      max_w=mini_rect[0] - int(32 * s))
        if slot_mgr is not None:
            # Stat cards are placed so they never cover the panels or each other.
            draw_player_cards(frame, slot_mgr, now, s, [coach_rect, mini_rect])
        draw_calibration_review(frame, review, now, s, web=web)
        draw_ball_hud(frame, ball, now, s, obstacles=[coach_rect, mini_rect])      # on top of everything
        rects = {"coach": coach_rect, "minimap": mini_rect}
    if locked:
        draw_lock_badge(frame, s)
    if reconnecting:
        if held:
            draw_relock_badge(frame, now, s, reconnect_detail)
        else:
            draw_reconnect_banner(frame, now, s, reconnect_detail)
    draw_toast(frame, s, toast)
    return rects


# =====================================================================
# 8. Recording
# =====================================================================
class LiveRecorder:
    def __init__(self, out_dir=APP_DIR):
        self.out_dir = out_dir
        self.writer = None
        self.path = None
        self.size = None
        self.start_time = 0.0
        self.frames = 0

    @property
    def active(self):
        return self.writer is not None

    @property
    def seconds(self):
        return time.time() - self.start_time if self.active else 0.0

    def start(self, frame_size, fps):
        w, h = int(frame_size[0]), int(frame_size[1])
        w, h = w - (w % 2), h - (h % 2)            # encoders prefer even sizes
        fps = float(np.clip(fps, 5.0, 30.0))
        path = os.path.join(self.out_dir, f"padeliq_live_recording_{datetime.now():%Y%m%d_%H%M%S}.mp4")
        for codec in ("mp4v", "avc1"):
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, (w, h))
            if writer.isOpened():
                self.writer, self.path, self.size = writer, path, (w, h)
                self.start_time, self.frames = time.time(), 0
                log(f"REC started -> {os.path.basename(path)} ({w}x{h} @ {fps:.1f} fps, {codec})")
                return True
            writer.release()
        log("WARNING: could not start recording (no MP4 encoder available).")
        return False

    def write(self, frame):
        if not self.active:
            return
        try:
            if (frame.shape[1], frame.shape[0]) != self.size:
                frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
            self.writer.write(frame)
            self.frames += 1
        except Exception as exc:
            log(f"Recording error ({exc!r}) - stopping recording.")
            self.stop()

    def stop(self):
        if not self.active:
            return None
        self.writer.release()
        path, frames = self.path, self.frames
        self.writer = None
        log(f"REC stopped -> {os.path.basename(path)} ({frames} frames)")
        return path


# =====================================================================
# 9. Detector wrapper (never takes the live loop down)
# =====================================================================
class PersonDetector:
    """Hybrid YOLOv8n + YOLOv8n-pose wrapper: provides person boxes, 17 COCO pose keypoints
    (`self.last_keypoints`) for ankle-floor localization, and sports-ball boxes (`self.last_balls`)."""

    def __init__(self, model_path="yolov8n.pt", imgsz=640, conf=0.28, backend="auto", pose_stride=2, use_roi=True):
        self.imgsz, self.conf = imgsz, conf
        self._errors = 0
        self.use_roi = use_roi
        self.last_balls = np.zeros((0, 4), dtype=np.float32)
        self.last_keypoints = []
        self.pose_det = CG.PoseCourtDetector(app_dir=APP_DIR, model_path=model_path, imgsz=imgsz, conf=conf,
                                             backend=backend)
        self.pose_det.POSE_STRIDE = max(1, int(pose_stride))
        self.pose_det.adaptive_pose = self.pose_det.POSE_STRIDE > 1
        self.model = self.pose_det.det_model or self.pose_det.pose_model
        get_court_ai(self.pose_det)
        if self.pose_det.ready:
            log(f"AI engine ready: detection {self.pose_det.det_backend} @ {self.pose_det.det_imgsz} | pose "
                f"{self.pose_det.pose_backend} (stride {self.pose_det.POSE_STRIDE}) | court ROI "
                f"{'on' if use_roi else 'off'}")
        else:
            log("WARNING: YOLO/Pose models unavailable. HUD will run without player detection.")

    @property
    def backend_info(self):
        pd = self.pose_det
        st = pd.stats
        tot = st["pose_crops"] + st["pose_reused"]
        return {"det_backend": pd.det_backend, "det_imgsz": pd.det_imgsz, "pose_backend": pd.pose_backend,
                "pose_stride": pd.POSE_STRIDE, "roi": list(pd.roi) if pd.roi else None,
                "pose_reuse_pct": round(100.0 * st["pose_reused"] / tot, 1) if tot else None}

    def set_roi(self, roi):
        self.pose_det.set_roi(roi if self.use_roi else None)

    def set_roi_from_corners(self, corners, frame_shape):
        self.set_roi(CG.PoseCourtDetector.court_roi(corners, frame_shape))

    def __call__(self, frame):
        """Returns `(N, 4)` person boxes; populates `self.last_keypoints` and `self.last_balls`."""
        self.last_balls = np.zeros((0, 4), dtype=np.float32)
        self.last_keypoints = []
        if not self.pose_det.ready:
            return np.zeros((0, 4), dtype=np.float32)
        try:
            boxes, kps, _ = self.pose_det.detect(frame, conf=self.conf)
            self.last_balls = self.pose_det.last_balls
            self.last_keypoints = kps
            return boxes
        except Exception as exc:
            self._errors += 1
            if self._errors <= 3:
                log(f"Detection error (frame skipped): {exc!r}")
            return np.zeros((0, 4), dtype=np.float32)


# =====================================================================
# 10. Main live loop
# =====================================================================
def prepare_canvas(raw, min_height, rotation=0):
    """Rotate (0/90/180/270) BEFORE any resize, never squash a portrait buffer (pillarboxed 16:9 instead), then
    upscale low-res streams (ESP32 640x480 / 800x600) so HUD text stays crisp."""
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    elif raw.shape[2] == 4:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    if rotation or raw.shape[0] > raw.shape[1]:
        raw = ORI.landscape(raw, rotation)
    h, w = raw.shape[:2]
    if min_height and h < min_height:
        f = min_height / float(h)
        return cv2.resize(raw, (int(round(w * f)), int(min_height)), interpolation=cv2.INTER_LINEAR)
    return raw


def fit_to_window(img, win, cache):
    """The Win32 HighGUI backend stretches imshow() images to the window (WINDOW_KEEPRATIO is ignored), so the HUD
    is letterboxed to the window's aspect here: fonts and the minimap are never squashed, in any window size."""
    now = time.time()
    if now - cache.get("t", 0) > 0.25:
        try:
            cache["rect"] = cv2.getWindowImageRect(win)
        except Exception:
            cache["rect"] = None
        cache["t"] = now
    rect = cache.get("rect")
    if not rect or rect[2] <= 0 or rect[3] <= 0:
        return img
    h, w = img.shape[:2]
    ar_win, ar = rect[2] / float(rect[3]), w / float(h)
    if abs(ar_win / ar - 1.0) < 0.01:
        return img
    if ar_win > ar:
        pad = int(round(h * ar_win)) - w
        return cv2.copyMakeBorder(img, 0, 0, pad // 2, pad - pad // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    pad = int(round(w / ar_win)) - h
    return cv2.copyMakeBorder(img, pad // 2, pad - pad // 2, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def placeholder_frame(size=(1280, 720)):
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)
    grad = np.linspace(25, 55, h, dtype=np.uint8)[:, None]
    img[:, :, 0] = grad
    img[:, :, 1] = (grad * 0.6).astype(np.uint8)
    img[:, :, 2] = (grad * 0.3).astype(np.uint8)
    return img


class LiveSession:
    def __init__(self, spec, args, cam_getter=None):
        self.spec = spec
        self.args = args
        self.cam_getter = cam_getter
        self.save_path, self.load_paths = calibration_files_for(spec.kind)
        self.canvas_shape = None
        self.corners = None
        self.H = None
        self.slot_mgr = LiveSlotManager(None)
        self.recorder = LiveRecorder()
        self.fps = 0.0
        self.quit = False
        self.last_recording = None
        self.review = CalibrationReview()
        self.scene_cut = CG.SceneCutDetector()
        self.applied_version = -1
        self.calib_frames = deque(maxlen=24)
        self._last_calib_grab = 0.0
        self.ball = PadelBallTracker()
        self.last_boxes = np.zeros((0, 4), np.float32)
        self.last_yolo_balls = np.zeros((0, 4), np.float32)
        self.last_kps = []
        self.last_det_ts = None
        self.lock = threading.RLock()       # display loop <-> async AI worker (slot manager / geometry)
        self.calib_locked = False           # [L]: geometry keys / scene-cut re-calibration disabled on stage
        self.roi_setter = None              # detector.set_roi (court-ROI guided inference)
        self._roi_key = None
        self._stash = {}                    # per-source confirmed calibration (F12 swaps)
        self.epoch = 0                      # bumps on source swaps: stale AI results are dropped
        self._toast = ("", 0.0)
        self.rotation = 0                   # 0/90/180/270 applied to every camera frame before anything else
        self.pending_autolock = False       # confirm + lock the court as soon as auto-calibration lands
        self.last_raw = None                # newest un-rotated camera frame (orientation check)

    # --------------------------------------------------------- orientation
    def set_rotation(self, deg, auto_lock=False, why=""):
        """Lock held. New rotation -> fresh geometry + court re-calibration on the next (rotated) frame."""
        self.rotation = ORI.norm_deg(deg)
        self.epoch += 1
        self.corners = self.H = None
        self.canvas_shape = None            # ensure_geometry() restarts auto-calibration on the next frame
        self.review.reset()
        self.slot_mgr.set_homography(None)
        self.slot_mgr.reset_players()
        self.ball.reset()
        self.scene_cut.reset()
        self.calib_frames.clear()
        self.last_boxes, self.last_yolo_balls, self.last_kps = np.zeros((0, 4), np.float32), np.zeros((0, 4), np.float32), []
        self.pending_autolock = bool(auto_lock)
        self.toast(f"Camera rotation {self.rotation} deg{(' - ' + why) if why else ''} - re-calibrating court", 2.5)

    def auto_orient(self, raw, auto_lock=False, first=False):
        """Lock held. Score the current rotation; below 55 % test 90 CW / 90 CCW / 180 and take the most upright."""
        deg, score, scores = ORI.detect_orientation(raw, None if first else self.rotation)
        log(f"Orientation check {raw.shape[1]}x{raw.shape[0]}: " +
            ", ".join(f"{d} deg {v:.0f}%" for d, v in scores.items()) + f" -> {deg} deg")
        self.set_rotation(deg, auto_lock=auto_lock, why=f"auto-orientation {score:.0f}%")
        return deg

    def _autolock_tick(self):
        rv = self.review
        if not self.pending_autolock:
            return
        if rv.state == "review" and rv.corners is not None and self.corners is not None:
            self.pending_autolock = False
            if rv.confirm("auto (orientation lock)") and self.canvas_shape is not None:
                self.save_confirmed(self.canvas_shape)
                self.calib_locked = True
                self.toast(f"Court locked at {self.rotation} deg ({rv.info.get('confidence_pct')}%) - analytics live", 3.0)
        elif rv.state == "failed":
            self.pending_autolock = False
            self.toast("Court not found - press [R] to rotate or [C] to drag corners", 3.5)

    # --------------------------------------------------------- stage helpers
    def toast(self, msg, secs=2.2):
        self._toast = (msg, time.time() + secs)
        log(msg)

    def toast_text(self, now=None):
        msg, until = self._toast
        return msg if msg and (now or time.time()) < until else ""

    def set_lock(self, locked):
        rv = self.review
        if locked:
            if rv.state in ("review", "manual") and rv.confirm("operator (lock)") and self.canvas_shape is not None:
                self.save_confirmed(self.canvas_shape)
            if rv.state != "confirmed":
                self.toast("Confirm the court calibration before locking it")
                return False
        self.calib_locked = bool(locked)
        self.toast("Calibration Locked - geometry keys disabled" if locked else "Calibration Unlocked")
        return True

    def _update_roi(self):
        if self.roi_setter is None:
            return
        rv = self.review
        key = (rv.state == "confirmed", rv.version, self.canvas_shape, self.applied_version)
        if key == self._roi_key:
            return
        self._roi_key = key
        roi = None
        if key[0] and self.corners is not None and self.canvas_shape is not None:
            roi = CG.PoseCourtDetector.court_roi(self.corners, self.canvas_shape)
        self.roi_setter(roi)
        if roi:
            log(f"Court-ROI guided inference: detector crop {roi[2] - roi[0]}x{roi[3] - roi[1]} at ({roi[0]}, {roi[1]})")

    @staticmethod
    def _src_key(spec):
        return f"{spec.kind}:{spec.source}"

    def switch_source(self, new_spec):
        """F12 hot-swap: stash this source's confirmed calibration, restore the target's (or start a fresh one)."""
        if self.spec is not None and self.review.state == "confirmed" and self.corners is not None \
                and self.canvas_shape is not None:
            self._stash[self._src_key(self.spec)] = {
                "corners": self.corners.copy(), "info": dict(self.review.info), "shape": self.canvas_shape,
                "locked": self.calib_locked, "floor_mask": self.slot_mgr.floor_mask,
                "floor_top_y": self.slot_mgr.floor_top_y, "rotation": self.rotation}
        self.spec = new_spec
        self.pending_autolock = False
        self.save_path, self.load_paths = calibration_files_for(new_spec.kind)
        self.epoch += 1
        self.slot_mgr.reset_players()
        self.ball.reset()
        self.scene_cut.reset()
        self.calib_frames.clear()
        self.last_boxes, self.last_yolo_balls, self.last_kps = np.zeros((0, 4), np.float32), np.zeros((0, 4), np.float32), []
        st = self._stash.get(self._src_key(new_spec))
        if st:
            self.canvas_shape = st["shape"]
            self.corners = st["corners"].copy()
            self.H = compute_homography(self.corners)
            self.review.restore(self.corners, st["info"], st["shape"])
            self.applied_version = self.review.version
            self.slot_mgr.set_homography(self.H)
            self.slot_mgr.floor_mask, self.slot_mgr.floor_top_y = st["floor_mask"], st["floor_top_y"]
            self.ball.set_geometry(self.H, self.canvas_shape)
            self.calib_locked = st["locked"]
            self.rotation = st.get("rotation", 0)
        else:
            self.canvas_shape = None
            self.corners = self.H = None
            self.review.reset()
            self.calib_locked = False
            self.rotation = 0
        self._roi_key = None
        self._update_roi()

    # --------------------------------------------------------- auto-calibration review
    def _frames_for_calibration(self):
        # Fast 15-frame collection directly from live camera (~0.75-1.1s) so convergence finishes in 1.0-1.8s!
        cam = self.cam_getter() if callable(self.cam_getter) else None
        if cam is not None:
            collected = list(self.calib_frames)[-6:]
            last_id = -1
            t_end = time.time() + 0.75
            while time.time() < t_end and len(collected) < 15:
                raw, fid, _ = cam.read_latest()
                if raw is not None and fid != last_id:
                    last_id = fid
                    collected.append(prepare_canvas(raw, self.args.hud_min_height, self.rotation))
                else:
                    time.sleep(0.012)
            if len(collected) >= 5:
                return collected
        if self.spec.kind == "file":
            return [prepare_canvas(f, self.args.hud_min_height, self.rotation)
                    for f in collect_calibration_frames(self.spec, count=16)]
        end = time.time() + 1.4
        while time.time() < end and len(self.calib_frames) < 15:
            time.sleep(0.04)
        return list(self.calib_frames)

    def start_auto_calibration(self, seed_frame=None):
        self.calib_frames.clear()
        if seed_frame is not None:
            self.calib_frames.append(seed_frame.copy())
            self.scene_cut.set_reference(seed_frame)
        self.review.start(self._frames_for_calibration, self.canvas_shape, seed_frame=seed_frame)

    def note_frame(self, canvas):
        now = time.time()
        if self.review.state == "detecting" and now - self._last_calib_grab > 0.045:
            self.calib_frames.append(canvas.copy())
            self._last_calib_grab = now
        elif self.review.state in ("review", "confirmed", "failed") and not self.calib_locked:
            # Automatic Scene-Cut Detector: re-calibrate immediately if stream cuts to a new court/angle!
            cut, metrics = self.scene_cut.update(canvas, now=now)
            if cut:
                log(f"Scene cut detected (HSV diff={metrics.get('hist_diff')}, SSIM={metrics.get('ssim')}) "
                    f"-> triggering fast 15-frame court re-calibration!")
                self.corners = None
                self.H = None
                self.slot_mgr.set_homography(None)
                self.slot_mgr.reset_players()
                self.ball.reset()
                self.start_auto_calibration(seed_frame=canvas)

    def sync_review(self, canvas_shape):
        """Apply review corners + primary floor mask to tracker; save on confirmation."""
        if self.review.tick():
            self.save_confirmed(canvas_shape)
        rv = self.review
        if rv.version != self.applied_version and rv.corners is not None:
            H = compute_homography(rv.corners)
            if H is not None:
                ai = get_court_ai()
                self.corners, self.H = rv.corners.copy(), H
                self.slot_mgr.set_homography(
                    H, floor_mask=ai.last_floor_mask, floor_top_y=rv.info.get("floor_top_y", ai.last_floor_top_y)
                )
                if rv.drag_idx is None and not rv.info.get("provisional"):
                    self.slot_mgr.reset_players()
                self.ball.set_geometry(H, (canvas_shape[0], canvas_shape[1]))
            self.applied_version = rv.version
        self._autolock_tick()
        self._update_roi()

    def save_confirmed(self, canvas_shape):
        if self.corners is None:
            return
        extra = dict(self.review.info, source_label=self.spec.label)
        extra = {k: v for k, v in extra.items() if k not in ("corners", "frame_size")}
        try:
            save_calibration(self.save_path, self.corners, (canvas_shape[0], canvas_shape[1], 3), extra=extra)
        except Exception as exc:
            log(f"Could not save calibration: {exc}")

    def ensure_geometry(self, canvas):
        shape = canvas.shape[:2]
        if shape == self.canvas_shape:
            return
        if self.corners is not None and self.canvas_shape is not None and aspect_changed(self.canvas_shape, shape):
            log(f"Frame aspect changed {self.canvas_shape[1]}x{self.canvas_shape[0]} -> {shape[1]}x{shape[0]} "
                f"(phone rotated / lens switched) -> fresh court calibration.")
            self.corners = self.H = None
            self.review.reset()
        if self.corners is not None and self.canvas_shape is not None:
            oh, ow = self.canvas_shape
            self.corners = self.corners * np.array([shape[1] / ow, shape[0] / oh], dtype=np.float32)
            self.H = compute_homography(self.corners)
            log(f"Stream resolution changed {ow}x{oh} -> {shape[1]}x{shape[0]}; court corners rescaled.")
        else:
            # NO fake static trapezoid fallback! Start cleanly with True AI CourtGeometryAI lock.
            self.corners = None
            self.H = None
        self.canvas_shape = shape
        self.slot_mgr.set_homography(self.H)
        self.slot_mgr.reset_players()
        self.ball.set_geometry(self.H, shape)
        if self.review.state == "idle":
            self.start_auto_calibration(seed_frame=canvas)

    def handle_key(self, key, clean_canvas=None, hud_frame=None):
        if key < 0 or key == 255:
            return
        ch = chr(key).lower() if key < 256 else ""
        rv = self.review
        cam = self.cam_getter() if callable(self.cam_getter) else None
        if ch == "l":
            self.set_lock(not self.calib_locked)
            return
        if ch == "e":
            if cam is None or cam.kind != "webcam":
                self.toast("Exposure control applies to USB webcams")
            else:
                cam.toggle_exposure()
                self.toast(f"Camera: {cam.exposure_label}")
            return
        if ch == "k":
            if cam is not None and cam.kind == "webcam":
                cam.open_driver_settings()
                self.toast("Camera driver settings (Power Line Frequency 50 Hz)")
            return
        if self.calib_locked and (ch in ("c", "a", "n", "h") or (rv.state == "manual" and key == 27)):
            self.toast("Calibration is locked - press [L] to unlock")
            return
        if rv.state == "manual" and key == 27:
            with rv.lock:
                rv.state, rv.review_since = "review", time.time()
            return
        if key in (13, 10) or ch == "y" or (rv.state == "manual" and ch == "c"):
            if rv.confirm("operator") and self.canvas_shape is not None:
                self.save_confirmed(self.canvas_shape)
            return
        if ch == "c":
            if clean_canvas is None:
                log("Calibration needs a live frame - waiting for the stream.")
                return
            rv.enter_manual(self.corners, clean_canvas.shape)
            log("Manual calibration: drag TL/TR/BR/BL/NET-L/NET-R handles ([N] = click 4 fresh corners, [Enter/C] lock)")
            return
        if ch == "a":
            log("Re-scanning court geometry ...")
            if self.last_raw is not None:
                self.auto_orient(self.last_raw, auto_lock=self.spec.kind == "mobile")
            else:
                self.start_auto_calibration(seed_frame=clean_canvas)
            return
        if ch == "r":                       # always allowed: rotating invalidates the geometry -> re-lock after
            self.calib_locked = False
            new = ORI.norm_deg(self.rotation + 90)
            upright = self.last_raw is None or                 ORI.orientation_score(ORI.landscape(self.last_raw, new))[0] >= ORI.MIN_CONF
            self.set_rotation(new, auto_lock=self.spec.kind == "mobile" and upright, why="[R]")
            if not upright:                  # never auto-lock a court that looks sideways / upside down
                self.toast(f"Rotated to {new} deg - court does not look upright: press [R] again or [A] to auto-orient", 3.5)
            return
        if rv.state == "manual" and ch == "n":
            rv.toggle_click_mode()
            return
        if rv.state == "manual" and ch == "h":
            rv.cycle_reference()
            return
        if key == 27 or ch == "q":
            self.quit = True
        elif ch == "x":
            self.slot_mgr.reset_players()
            log("All player stats reset.")
        elif ch == "s":
            if self.recorder.active:
                self.last_recording = self.recorder.stop()
            elif hud_frame is not None:
                self.recorder.start((hud_frame.shape[1], hud_frame.shape[0]), self.fps or 15.0)
            else:
                log("Recording needs a live frame - waiting for the stream.")


class AsyncAIWorker:
    """Runs YOLO + pose on the newest submitted canvas in a background thread (single slot, drop-oldest), so the
    desktop window shows every camera frame at native FPS while boxes are velocity-predicted between AI ticks."""

    def __init__(self, detector, session):
        self.detector, self.session = detector, session
        self.cond = threading.Condition()
        self.pending = None
        self.inferred = 0
        self.latency_ms = 0.0
        self.rate = 0.0
        self._times = deque(maxlen=20)
        self.running = True
        self.thread = threading.Thread(target=self._loop, name="PadelIQ-AI", daemon=True)
        self.thread.start()

    def submit(self, canvas, ts):
        with self.cond:
            self.pending = (canvas, ts, self.session.epoch)   # overwrite: never process a stale frame
            self.cond.notify()

    def _loop(self):
        s = self.session
        while self.running:
            with self.cond:
                if self.pending is None:
                    self.cond.wait(0.1)
                job, self.pending = self.pending, None
            if job is None:
                continue
            canvas, ts, epoch = job
            try:
                t0 = time.perf_counter()
                boxes = self.detector(canvas)
                self.latency_ms = 0.8 * self.latency_ms + 0.2 * (time.perf_counter() - t0) * 1000 if self.inferred \
                    else (time.perf_counter() - t0) * 1000
                with s.lock:
                    if s.canvas_shape == canvas.shape[:2] and s.epoch == epoch:
                        s.last_boxes, s.last_yolo_balls = boxes, self.detector.last_balls
                        s.last_kps, s.last_det_ts = self.detector.last_keypoints, ts
                        s.slot_mgr.update_detections(boxes, now=ts, keypoints=self.detector.last_keypoints,
                                                     frame_shape=canvas.shape)
                self.inferred += 1
                now = time.time()
                self._times.append(now)
                if len(self._times) > 1:
                    self.rate = (len(self._times) - 1) / max(1e-3, self._times[-1] - self._times[0])
            except Exception as exc:
                log(f"AI worker error (frame skipped): {exc!r}")
                time.sleep(0.05)

    def stop(self):
        self.running = False
        with self.cond:
            self.cond.notify_all()


def tune_runtime_threads():
    """Latency-oriented thread tuning shared by the desktop engine and the web server:
    * GIL switch interval 5 ms -> 2 ms: capture / render / MJPEG threads get the interpreter back sooner while the
      AI thread runs Python-heavy post-processing.
    * torch intra-op threads = cores - 2, leaving cores for capture decode, HUD render and JPEG encode
      (OpenCV and torch release the GIL inside their kernels)."""
    sys.setswitchinterval(0.002)
    try:
        import torch
        torch.set_num_threads(max(2, (os.cpu_count() or 4) - 2))
    except Exception:
        pass


class SessionMobileController:
    """Lets the paired phone drive the desktop LiveSession calibration (padeliq_mobile_bridge.MobileController).
    Called from the HTTPS bridge thread; every geometry change honours the stage calibration lock."""

    def __init__(self, session):
        self.s = session
        self.last_canvas = None
        self.phone_ready = False            # set on the first phone frame -> the display loop swaps sources
        self.reorient_at = 0.0              # phone rotated -> display loop re-runs the orientation check

    def _active(self):
        return self.s.spec is not None and self.s.spec.kind == "mobile"

    def _unlocked(self):
        if self.s.calib_locked:
            raise ValueError("calibration is locked on stage - press [L] on the PC to unlock")

    def calibration_snapshot(self):
        s, now = self.s, time.time()
        if not self._active():
            return {"state": "idle", "corners": None, "frame_shape": None, "locked": s.calib_locked,
                    "rotation": s.rotation}
        rv = s.review
        st = rv.status()
        with rv.lock:
            corners = None if rv.corners is None else rv.corners.copy()
            version = rv.version
        with s.lock:
            players = sum(1 for p in list(s.slot_mgr.players.values()) if p.is_active(now))
            shape = s.canvas_shape
        return dict(st, corners=corners, frame_shape=shape, version=version, players=players,
                    inference_fps=round(s.fps, 1), locked=s.calib_locked, rotation=s.rotation)

    def auto_detect(self):
        self._unlocked()
        canvas = self.last_canvas
        if not self._active() or canvas is None:
            raise ValueError("no phone frame yet")
        with self.s.lock:
            self.s.start_auto_calibration(seed_frame=canvas)
        end = time.time() + 30.0
        while time.time() < end and self.s.review.state == "detecting":
            time.sleep(0.1)
        rv = self.s.review
        if rv.state == "failed":
            raise ValueError(f"court not found ({rv.error}) - drag the 4 corners manually")
        if rv.corners is None:
            raise ValueError("court detection still running")
        return (f"Court detected: {rv.info.get('confidence_pct') or 0:.0f}% ({rv.info.get('angle') or 'Baseline'}) "
                f"- drag corners to fine-tune, then Lock")

    def preview_corners(self, corners_px):
        self._unlocked()
        s = self.s
        with s.lock:
            if s.canvas_shape is None:
                raise ValueError("no phone frame yet")
            if compute_homography(corners_px) is None:
                raise ValueError("corners must form a convex court in order TL, TR, BR, BL")
            s.review.frame_shape = tuple(s.canvas_shape)
            s.review.set_manual_corners(corners_px, dict(s.review.info, method="phone_touch"))

    def lock(self, corners_px=None):
        s = self.s
        if corners_px is not None:
            self.preview_corners(corners_px)
        with s.lock:
            rv = s.review
            if rv.state != "confirmed" and not rv.confirm("operator (phone)"):
                raise ValueError(f"nothing to lock yet (calibration state: {rv.state})")
            s.sync_review(s.canvas_shape)
            s.save_confirmed(s.canvas_shape)
            if not s.calib_locked:
                s.set_lock(True)
        return "Court locked - live analytics started"

    def on_phone_connected(self, meta):
        self.phone_ready = True

    def on_phone_rotated(self, meta):
        self.reorient_at = time.time() + 0.4    # let frames in flight with the old rotation drain first

    def hud_jpeg(self):
        return None


def start_desktop_mobile_bridge(args, session):
    """QR phone portal for the desktop window. The HTTPS server (cert generation, bind) starts in the background so
    the stage window never freezes. Returns (MB module, hub, controller)."""
    import padeliq_mobile_bridge as MB
    hub = MB.MobileHub(https_port=args.mobile_port, lan_ip=args.lan_ip, token=args.mobile_token or None)
    ctrl = SessionMobileController(session)
    hub.controller = ctrl
    if args.no_https:
        hub.https_error = "disabled with --no-https"
    else:
        threading.Thread(target=lambda: MB.start_https_server(MB.build_https_app(hub), hub,     # FastAPI import +
                                                             os.path.join(APP_DIR, "certs")),  # cert gen off-thread
                         name="PadelIQ-https-start", daemon=True).start()
    log(f"Mobile QR portal: scan the QR code in the window or open {hub.url.split('?')[0]}")
    return MB, hub, ctrl


def draw_qr_overlay(frame, MB, hub, cache):
    """[M] pairing card (QR + URL + link state) in the bottom-right corner, pre-rendered once per state."""
    h, w = frame.shape[:2]
    s = h / 720.0
    state = "connected" if hub.connected else ("ready" if hub.https_enabled else (hub.https_error or "starting"))
    key = (w, h, hub.url, state)
    if cache.get("key") != key:
        q = int(250 * s)
        pad = int(14 * s)
        cw, ch = q + 2 * pad, q + int(96 * s) + 2 * pad
        card = np.full((ch, cw, 3), (24, 18, 12), np.uint8)
        card[pad:pad + q, pad:pad + q] = cv2.resize(hub.qr_image(q), (q, q), interpolation=cv2.INTER_NEAREST)
        y = pad + q + int(24 * s)
        put_text(card, "MOBILE QR PORTAL", (pad, y), 0.6 * s, (0, 255, 180), 2, s)
        put_text(card, fit_text(hub.url.split("?")[0], 0.42 * s, 1, cw - 2 * pad), (pad, y + int(24 * s)),
                 0.42 * s, (235, 235, 235), 1, s)
        msg = {"connected": "Phone connected - starting stream ...", "ready": "Scan with the phone camera  [M] hide"}
        put_text(card, fit_text(msg.get(state, f"HTTPS: {state}"), 0.42 * s, 1, cw - 2 * pad),
                 (pad, y + int(48 * s)), 0.42 * s, (0, 200, 255) if state != "ready" else (180, 180, 180), 1, s)
        cv2.rectangle(card, (0, 0), (cw - 1, ch - 1), (0, 255, 180), max(1, int(2 * s)))
        cache.update(key=key, card=card)
    card = cache["card"]
    ch, cw = card.shape[:2]
    m = int(18 * h / 720.0)
    x0, y0 = max(0, w - cw - m), max(0, h - ch - m - int(40 * h / 720.0))
    roi = frame[y0:y0 + ch, x0:x0 + cw]
    roi[:] = card[:roi.shape[0], :roi.shape[1]]


def run_live_padeliq(args):
    tune_runtime_threads()
    spec = resolve_source(args.source, esp32_ip=args.esp32_ip, video=args.video, backend=args.backend)
    if spec is None:
        log(f"ERROR: no usable video source for --source {args.source!r}. "
            "Use webcam (best camera), 0/1, an http(s)/rtsp URL, an ESP32 IP, a video file, 'sim' or 'auto'.")
        return 2

    test_mode = args.test_run > 0
    display = not (test_mode or args.no_display)
    log(f"Source: {spec.label}")

    detector = PersonDetector(args.model, imgsz=args.imgsz, conf=args.conf, backend=args.ai_backend,
                              pose_stride=args.pose_stride, use_roi=not args.no_roi)

    session = LiveSession(spec, args, cam_getter=lambda: cam)
    mobile = {"MB": None, "hub": None, "ctrl": None, "card": {}, "waiting": (None, 0.0, "")}

    def ensure_mobile():
        if mobile["hub"] is None:
            mobile["MB"], mobile["hub"], mobile["ctrl"] = start_desktop_mobile_bridge(args, session)
        return mobile["hub"]

    def make_cam(sp, exposure=args.exposure):
        if sp.kind == "mobile":
            return ensure_mobile().create_camera(sp, stall_timeout=args.stall_timeout)
        return ThreadedLiveCamera(sp, stall_timeout=args.stall_timeout, cam_width=args.cam_width,
                                  cam_height=args.cam_height, cam_fps=args.cam_fps, exposure=exposure,
                                  sport_exposure=args.sport_exposure)

    cam = make_cam(spec)
    qr_overlay = spec.kind == "mobile"
    session.roi_setter = detector.set_roi
    ai = AsyncAIWorker(detector, session)
    # [F12] failover: live camera <-> clean match replay (or --failover-source). The target is opened in the
    # background and swapped in only once it delivers frames, so the window never shows a gap.
    swap_targets = [args.source, args.failover_source or ("sim" if spec.kind != "file" else "webcam")]
    spec_cache = {0: spec}
    swap_idx = 0
    pending = None
    max_gap_ms = 0.0                    # longest interval between displayed camera frames (swap smoothness)
    big_gaps = []
    swaps_done = 0
    fullscreen = False
    e2e_ms = deque(maxlen=600)          # capture/arrival -> frame shown (per displayed camera frame)
    ball_worker = BallWorker(lambda: cam, session.ball,
                             lambda: (session.last_boxes, session.last_yolo_balls, session.last_kps, session.last_det_ts),
                             args.hud_min_height, rotation_getter=lambda: session.rotation)
    oriented_cam = None                 # phone cameras get an orientation check + instant lock on their first frame
    win_cache = {}
    win_name = "PadelIQ Live Camera Engine"
    if display:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, 1280, 720)
        cv2.setMouseCallback(win_name, lambda ev, x, y, fl, pa: session.review.on_mouse(ev, x, y))
        log("Live engine running:  [Enter] confirm court  [C] drag corners  [A] re-scan  [L] lock calibration  "
            "[E] auto exposure / sport shutter  [K] camera driver settings  [F] fullscreen  [F12] failover  "
            "[M] mobile QR portal  [R] rotate 90  [X] reset stats  [S] record  [Q] quit")

    # Scripted key presses exercise the same handlers headlessly during --test-run.
    scripted_keys = {}
    if test_mode:
        n = args.test_run
        scripted_keys = {max(2, n // 10): ord('s'), max(3, n // 4): ord('x'), max(4, (n * 3) // 5): ord('s')}
        if args.test_stage_keys:
            scripted_keys.update({max(5, (n * 7) // 10): ord('l'), max(6, (n * 3) // 4): ord('a'),
                                  max(7, (n * 4) // 5): ord('e')})
        if args.test_failover:
            scripted_keys[max(8, n // 3)] = F12_KEYS[0]
            scripted_keys[max(9, (n * 2) // 3)] = F12_KEYS[0]
        if args.test_qr:
            scripted_keys[max(5, n // 5)] = ord('m')
        if args.test_rotate:                   # unlock, rotate 90 (pillarboxed, un-stretched), auto-orient back
            scripted_keys[max(10, (n * 2) // 5)] = ord('r')
            scripted_keys[max(11, n // 2)] = ord('a')

    processed = 0                       # frames processed by the AI worker
    displayed = 0                       # camera frames shown in the window
    last_fid = -1
    last_canvas = None
    last_cap_ts = 0.0
    fps_t0 = time.time()
    frame_intervals = deque(maxlen=30)
    start = time.time()
    shown = False
    exit_code = 0
    recordings_made = []
    captured = False
    captured_ball = False
    rot_shot_due = None

    try:
        while not session.quit:
            # ---- F12 failover: swap once the target camera has delivered a frame
            if pending is not None:
                pc = pending["cam"]
                if pending["err"] or (time.time() - pending["t0"] > 20 and (pc is None or pc.frame_id == 0)):
                    session.toast(f"Failover failed: {pending['err'] or 'no frames from ' + swap_targets[pending['idx']]}", 3.5)
                    if pc is not None:
                        threading.Thread(target=pc.release, daemon=True).start()
                    pending = None
                elif pc is not None and pc.frame_id > 0:
                    if pending.get("mobile"):
                        swap_targets[pending["idx"]], spec_cache[pending["idx"]] = "mobile", pending["spec"]
                        qr_overlay = False
                    old = cam
                    with session.lock:
                        session.switch_source(pending["spec"])
                    cam, spec, swap_idx = pc, pending["spec"], pending["idx"]
                    last_fid = -1
                    pending = None
                    swaps_done += 1
                    threading.Thread(target=old.release, daemon=True).start()
                    session.toast(f"Source: {spec.label}", 2.5)
            # ---- QR portal: a scanned phone streams its first frame -> seamless swap to the phone camera
            mc = mobile["ctrl"]
            if mc is not None and mc.phone_ready:
                mc.phone_ready = False
                if spec.kind != "mobile" and pending is None and not args.no_mobile_autoswitch:
                    msp = SourceSpec("mobile", "mobile", "MOBILE QR CAMERA")
                    pending = {"idx": swap_idx, "t0": time.time(), "cam": make_cam(msp), "spec": msp, "err": "",
                               "mobile": True}
                    session.toast("Phone connected - switching to the phone camera", 2.5)
                elif spec.kind == "mobile":
                    qr_overlay = False
            # Wake the instant the camera publishes; never queue (newest frame only).
            raw, fid, cap_ts = cam_wait_latest(cam, last_fid, 1.0 / 60.0)
            ok = raw is not None
            new_frame = ok and fid != last_fid
            prev_processed = processed
            processed = ai.inferred
            if new_frame:
                last_fid, last_cap_ts = fid, cap_ts
                mc = mobile["ctrl"]
                if mc is not None and mc.reorient_at and time.time() >= mc.reorient_at and cam.kind == "mobile":
                    mc.reorient_at = 0.0                   # phone rotated: explicit operator action -> re-lock
                    with session.lock:
                        session.calib_locked = False
                        session.auto_orient(raw, auto_lock=True, first=True)
                elif cam.kind == "mobile" and oriented_cam is not cam and session.review.state != "confirmed":
                    oriented_cam = cam
                    with session.lock:
                        session.auto_orient(raw, auto_lock=True, first=True)
                rot = session.rotation
                canvas = prepare_canvas(raw, args.hud_min_height, rot)
                with session.lock:
                    session.last_raw = raw
                    if rot != session.rotation:            # rotated meanwhile: drop this stale-orientation frame
                        continue
                    session.ensure_geometry(canvas)
                    session.note_frame(canvas)
                    session.sync_review(canvas.shape)
                ai.submit(canvas, cap_ts)                # async YOLO + pose on the newest frame
                last_canvas = canvas
                if mobile["ctrl"] is not None and cam.kind == "mobile":
                    mobile["ctrl"].last_canvas = canvas
                displayed += 1
                t = time.time()
                if displayed > 1:
                    frame_intervals.append(t - fps_t0)
                    if displayed > 30:
                        max_gap_ms = max(max_gap_ms, (t - fps_t0) * 1000.0)
                        if (t - fps_t0) > 0.05 and len(big_gaps) < 40:
                            big_gaps.append((round(t - start, 2), round((t - fps_t0) * 1000), session.review.state))
                    session.fps = 1.0 / max(1e-3, float(np.median(frame_intervals)))   # display FPS (recorder)
                fps_t0 = t

            with session.lock:
                if last_canvas is not None:
                    session.sync_review(last_canvas.shape)           # live drag / auto-confirm between frames
                    hud = last_canvas.copy()
                    slot_mgr, corners = session.slot_mgr, session.corners
                else:
                    hud = placeholder_frame()
                    slot_mgr, corners = None, None

                degraded = cam.is_degraded
                if degraded:
                    if cam.frame_id == 0:
                        detail = f"Connecting to {spec.label} ..."
                    else:
                        detail = f"No frames for {cam.seconds_since_frame:.1f}s | attempt #{max(1, cam.reconnect_count)}"
                else:
                    detail = ""
                render_live_hud(hud, slot_mgr, corners, ai.rate, H=session.H, source_label=spec.label,
                                recording=session.recorder.active, rec_seconds=session.recorder.seconds,
                                reconnecting=degraded, reconnect_detail=detail,
                                ball=session.ball if last_canvas is not None else None,
                                review=session.review if last_canvas is not None else None,
                                held=last_canvas is not None, locked=session.calib_locked,
                                toast=session.toast_text())
            if cam.kind == "mobile" and cam.waiting_for_phone:
                # QR pairing screen until the phone streams (PIL text -> rebuilt at 2 Hz, then reused)
                MBm, hub_m = mobile["MB"], mobile["hub"]
                wdetail = MBm.waiting_detail(cam, hub_m)
                wimg, wt, wd = mobile["waiting"]
                if wimg is None or wd != wdetail or time.time() - wt > 0.5:
                    wimg = MBm.render_waiting_screen(hub_m, (hud.shape[1], hud.shape[0]), wdetail)
                    mobile["waiting"] = (wimg, time.time(), wdetail)
                hud = wimg.copy()
                draw_toast(hud, hud.shape[0] / 720.0, session.toast_text())
            elif qr_overlay and mobile["hub"] is not None:
                draw_qr_overlay(hud, mobile["MB"], mobile["hub"], mobile["card"])
            if new_frame or degraded:
                session.recorder.write(hud)
            if args.capture_verified and not captured and last_canvas is not None and                     session.review.state == "confirmed" and session.ball.visible:
                now_c = time.time()
                act = {sid: p for sid, p in session.slot_mgr.players.items()
                       if p.is_active(now_c, 0.5) and p.smoothed_pos_m is not None}
                sides_ok = len(act) == 4 and all(
                    (sid in (1, 2)) == (float(p.smoothed_pos_m[1]) < NET_Y_M)
                    for sid, p in act.items())
                if sides_ok:
                    cv2.imwrite(args.capture_verified, hud)
                    captured = True
                    log(f"[verify] saved {os.path.basename(args.capture_verified)}: 4 players on correct sides, "
                        f"ball {session.ball.speed_kmh:.0f} km/h, calibration {session.review.info.get('method')} "
                        f"({session.review.info.get('confidence_pct')}%, confirmed by {session.review.confirmed_by})")

            if args.capture_ball and not captured_ball and last_canvas is not None and \
                    session.review.state == "confirmed" and session.ball.visible and not session.ball.coasting and \
                    len(session.ball.trail) >= 12 and session.ball.court_pos is not None and \
                    session.ball.speed_kmh > session.ball.SPEED_MIN_KMH and                     sum(1 for p in session.slot_mgr.players.values() if p.is_active(time.time(), 0.5)) >= 4:
                cv2.imwrite(args.capture_ball, hud)
                captured_ball = True
                log(f"[verify] saved {os.path.basename(args.capture_ball)}: ball {session.ball.speed_kmh:.0f} km/h, "
                    f"trail {len(session.ball.trail)} pts, shots {session.ball.shots}, court {session.ball.court_pos}")
            if rot_shot_due is not None and processed >= rot_shot_due:
                rot_shot_due = None
                cv2.imwrite(os.path.join(APP_DIR, "rotate_test_verified.jpg"), hud)
                log(f"[test] [R] frame: rotation {session.rotation} deg, HUD {hud.shape[1]}x{hud.shape[0]} "
                    f"-> rotate_test_verified.jpg")
            if new_frame:
                e2e_ms.append((time.time() - last_cap_ts) * 1000.0)
            key = -1
            if display:
                cv2.imshow(win_name, fit_to_window(hud, win_name, win_cache))
                key = cv2.waitKeyEx(1)
                if shown and cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    log("Window closed.")
                    break
                shown = True
            else:
                due = [k for k in scripted_keys if k <= processed and k > prev_processed] if processed != prev_processed else []
                if due:
                    key = scripted_keys.pop(due[0])
                    if key == ord('r'):
                        rot_shot_due = processed + 25
                    log(f"[test] simulating key '{'F12' if key in F12_KEYS else chr(key).upper()}' at frame {processed}")

            was_recording = session.recorder.active
            if key in F12_KEYS:
                if pending is None:
                    idx = 1 - swap_idx
                    holder = {"idx": idx, "t0": time.time(), "cam": None, "spec": None, "err": ""}

                    def open_target(holder=holder, idx=idx, exposure=cam.exposure_mode):
                        try:
                            sp = spec_cache.get(idx) or resolve_source(swap_targets[idx], esp32_ip=args.esp32_ip,
                                                                       video=args.video, backend=args.backend)
                            if sp is None:
                                raise ValueError(f"no usable source for {swap_targets[idx]!r}")
                            spec_cache[idx] = sp
                            holder["spec"] = sp
                            holder["cam"] = make_cam(sp, exposure)
                        except Exception as exc:
                            holder["err"] = str(exc)
                    threading.Thread(target=open_target, daemon=True).start()
                    pending = holder
                    session.toast(f"[F12] Failover -> {swap_targets[idx]} (current view stays live until it is ready)", 3.0)
                key = -1
            elif key in (ord('m'), ord('M')):
                ensure_mobile()
                qr_overlay = not qr_overlay
                session.toast("Mobile QR portal: scan to stream the phone camera" if qr_overlay else "QR portal hidden")
                key = -1
            elif display and key in (ord('f'), ord('F')):
                fullscreen = not fullscreen
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
                key = -1
            elif key >= 256:
                key = -1
            with session.lock:
                session.handle_key(key, clean_canvas=last_canvas, hud_frame=hud)
            if was_recording and not session.recorder.active and session.last_recording:
                recordings_made.append(session.last_recording)

            if test_mode:
                if processed >= args.test_run:
                    cv2.imwrite(VERIFY_SCREENSHOT, hud)
                    el = time.time() - start
                    lat = np.array(e2e_ms) if e2e_ms else np.array([np.nan])
                    log(f"[test] AI processed {processed} frames in {el:.1f}s (~{ai.rate:.1f} AI FPS, "
                        f"{ai.latency_ms:.0f} ms/inference) | window showed {displayed} camera frames "
                        f"(~{session.fps:.1f} display FPS) | capture->display latency median {np.median(lat):.1f} ms, "
                        f"p95 {np.percentile(lat, 95):.1f} ms -> screenshot {os.path.basename(VERIFY_SCREENSHOT)}")
                    if args.test_rotate:
                        up = ORI.orientation_score(ORI.landscape(session.last_raw, session.rotation))[0]                             if session.last_raw is not None else 0.0
                        log(f"[test] final orientation score {up:.0f}% at {session.rotation} deg")
                        if up < ORI.MIN_CONF or not session.calib_locked:
                            log("[test] FAILED: [A] did not restore an upright, locked court after [R]")
                            exit_code = 1
                    if args.test_qr and spec.kind != "mobile":
                        log("[test] FAILED: [M] QR portal did not switch to the phone camera")
                        exit_code = 1
                    log(f"[test] orientation: rotation {session.rotation} deg | canvas "
                        f"{last_canvas.shape[1] if last_canvas is not None else 0}x"
                        f"{last_canvas.shape[0] if last_canvas is not None else 0}")
                    log(f"[test] stage: calibration locked={session.calib_locked} | F12 swaps={swaps_done} | "
                        f"longest gap between displayed frames {max_gap_ms:.0f} ms | AI {detector.backend_info}")
                    if big_gaps:
                        log(f"[test] display gaps > 50 ms (t s, ms, calibration state): {big_gaps}")
                    if args.test_failover and swaps_done < 2:
                        log("[test] FAILED: F12 failover did not complete twice")
                        exit_code = 1
                    break
                if time.time() - start > args.test_timeout:
                    cv2.imwrite(VERIFY_SCREENSHOT, hud)
                    log(f"[test] FAILED: only {processed}/{args.test_run} frames within {args.test_timeout}s.")
                    exit_code = 1
                    break
    except KeyboardInterrupt:
        log("Interrupted by user.")
    finally:
        ai.stop()
        if pending is not None and pending.get("cam") is not None:
            pending["cam"].release()
        if session.recorder.active:
            p = session.recorder.stop()
            if p:
                recordings_made.append(p)
        ball_worker.stop()
        cam.release()
        if display:
            cv2.destroyAllWindows()

    if test_mode and exit_code == 0:
        exit_code = verify_test_outputs(recordings_made, keep=args.keep_test_recording)
        active = [p.slot_id for p in session.slot_mgr.players.values() if p.smoothed_pos_m is not None]
        log(f"[test] players tracked: {active or 'none'} | source: {spec.label} | calibration: "
            f"{session.review.state} ({session.review.info.get('method')}, {session.review.info.get('confidence_pct')}%) "
            f"| ball visible {session.ball.state()['visibility_pct']}% of ball frames, {session.ball.total_shots} shots")
        log("[test] SMOKE TEST PASSED" if exit_code == 0 else "[test] SMOKE TEST FAILED")
    return exit_code


def verify_test_outputs(recordings, keep=False):
    ok = True
    img = cv2.imread(VERIFY_SCREENSHOT)
    if img is None or img.size == 0:
        log("[test] screenshot missing or unreadable!")
        ok = False
    if not recordings:
        log("[test] the scripted [S] recording toggle produced no file!")
        ok = False
    for path in recordings:
        cap = cv2.VideoCapture(path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        readable = cap.isOpened() and cap.read()[0]
        cap.release()
        log(f"[test] recording {os.path.basename(path)}: {n} frames, readable={bool(readable)}")
        ok = ok and bool(readable) and n > 0
        if not keep:
            try:
                os.remove(path)
            except OSError:
                pass
    return 0 if ok else 1


def build_arg_parser():
    p = argparse.ArgumentParser(description="PadelIQ Real-Time Live Camera Engine (ESP32-CAM / RTSP / Webcam)")
    p.add_argument("--source", default="auto",
                   help="auto (default) | sim | webcam (best colour webcam, Logitech BRIO preferred) | "
                        "mobile (QR phone camera) | 0/1 (webcam index) | "
                        "http://<esp32-ip>:81/stream | rtsp://... | ESP32 IP | YouTube URL | video file")
    p.add_argument("--esp32-ip", default=None, help="ESP32-CAM IP or URL tried first in auto mode (env PADELIQ_STREAM_URL also works)")
    p.add_argument("--video", default=None, help="video file for simulated live mode (default: padel_input.mp4)")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO model (yolov8n.pt is fastest for live use)")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="YOLO inference size (512 default; 640 = slower)")
    p.add_argument("--conf", type=float, default=0.30, help="YOLO person confidence threshold")
    p.add_argument("--ai-backend", default="auto", choices=["auto", "openvino", "pytorch"],
                   help="auto = OpenVINO (yolov8n_openvino_model + yolov8n-pose_openvino_model) with PyTorch fallback")
    p.add_argument("--pose-stride", type=int, default=2, help="fresh pose every N AI ticks per player (1 = every tick)")
    p.add_argument("--no-roi", action="store_true", help="always run detection on the full frame")
    p.add_argument("--exposure", default="auto", choices=list(EXPOSURE_MODES),
                   help="webcam exposure at start: auto | sport (fast shutter, max FPS); toggle live with [E]")
    p.add_argument("--sport-exposure", type=float, default=SPORT_EXPOSURE,
                   help="sport-shutter exposure value (DirectShow log2 s: -5 = 1/32 s, -6 = 1/64 s)")
    p.add_argument("--failover-source", default=None,
                   help="[F12] target (default: sim replay when live, best webcam when replaying)")
    p.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "any"], help="webcam backend")
    p.add_argument("--mobile-port", type=int, default=8443,
                   help="HTTPS port of the QR phone portal ([M] or --source mobile; certs/ generated on first run)")
    p.add_argument("--lan-ip", default=None, help="LAN IP encoded in the QR code (auto-detected by default)")
    p.add_argument("--no-https", action="store_true", help="do not start the HTTPS phone portal")
    p.add_argument("--no-mobile-autoswitch", action="store_true",
                   help="[M] portal: do not switch to the phone automatically when it starts streaming")
    p.add_argument("--mobile-token", default="", help=argparse.SUPPRESS)      # fixed pairing token (tests)
    p.add_argument("--cam-width", type=int, default=None, help="webcam width (default 1280, MJPG)")
    p.add_argument("--cam-height", type=int, default=None, help="webcam height (default 720, MJPG)")
    p.add_argument("--cam-fps", type=int, default=None, help="webcam FPS request (default 60; driver clamps)")
    p.add_argument("--stall-timeout", type=float, default=STALL_TIMEOUT_S, help="seconds without frames before reconnecting")
    p.add_argument("--hud-min-height", type=int, default=720, help="upscale smaller streams to this height for a crisp HUD (0 = off)")
    p.add_argument("--no-display", action="store_true", help="run without a window")
    p.add_argument("--test-run", type=int, default=0, metavar="N",
                   help="headless smoke test: process N frames, save live_hud_verified.jpg, exit 0 on success")
    p.add_argument("--test-timeout", type=float, default=180.0, help="max seconds for --test-run")
    p.add_argument("--keep-test-recording", action="store_true", help="keep the MP4 produced by --test-run")
    p.add_argument("--capture-ball", default="", metavar="PATH",
                   help="save the first full-HUD frame with a locked ball (trail, km/h tag, minimap dot)")
    p.add_argument("--test-stage-keys", action="store_true", help="--test-run also presses [L], [A] (refused), [E]")
    p.add_argument("--test-failover", action="store_true", help="--test-run also presses [F12] twice")
    p.add_argument("--test-qr", action="store_true", help="--test-run also presses [M]; a phone must then stream")
    p.add_argument("--test-rotate", action="store_true", help="--test-run also presses [R] (rotate 90) then [A] (auto-orient)")
    p.add_argument("--capture-verified", default="", metavar="PATH",
                   help="save the first frame with confirmed calibration, 4 players on correct sides and the ball visible")
    p.add_argument("--auto-calibrate", action="store_true",
                   help="detect the court from its lines, save the calibration for this source type and exit")
    return p


def run_auto_calibration(args):
    spec = resolve_source(args.source, esp32_ip=args.esp32_ip, video=args.video, backend=args.backend)
    if spec is None:
        log(f"ERROR: no usable video source for --source {args.source!r}")
        return 2
    log(f"Auto-calibrating court from: {spec.label}")
    frames = collect_calibration_frames(spec)
    if not frames:
        log("ERROR: could not read frames for calibration.")
        return 1
    frames = [prepare_canvas(f, args.hud_min_height) for f in frames]
    corners, info = run_court_autocalibration(frames)
    if corners is None:
        log(f"Auto-calibration FAILED: {info}. Use [C] in the live window to calibrate manually.")
        return 1
    save_path, _ = calibration_files_for(spec.kind)
    extra = dict(info)
    if spec.kind == "file":
        extra["video"] = os.path.basename(str(spec.source))
    save_calibration(save_path, corners, frames[0].shape, extra=extra)
    preview = os.path.join(APP_DIR, "auto_calibration_preview.jpg")
    cv2.imwrite(preview, draw_calibration_preview(frames[len(frames) // 2], corners))
    log(f"Court corners (px): {np.round(corners, 1).tolist()}")
    log(f"Max reprojection error: {info.get('max_reprojection_error_px')} px | preview -> {os.path.basename(preview)}")
    return 0


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    try:
        if args.auto_calibrate:
            return run_auto_calibration(args)
        return run_live_padeliq(args)
    except Exception:
        log("FATAL error:\n" + traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
