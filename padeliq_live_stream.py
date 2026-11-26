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
# Ranked name hints for the external colour webcam (DirectShow friendly names). IR / depth sensors are skipped.
PREFERRED_CAMERA_HINTS = ("brio", "logitech", "c920", "c922", "c930", "streamcam", "elgato", "obsbot", "insta360")
IR_CAMERA_HINTS = (" ir ", "infrared", "ir camera", "depth", "windows hello")
LEGACY_VIDEO_CALIBRATION_FILE = os.path.join(APP_DIR, "court_calibration.json")
VERIFY_SCREENSHOT = os.path.join(APP_DIR, "live_hud_verified.jpg")
PREFERRED_SIM_VIDEOS = ("clean_padel_match.mp4", "padel_input.mp4")

STALL_TIMEOUT_S = 2.5           # watchdog: no frames for this long -> reconnect
WEBCAM_PROFILE_FILE = os.path.join(APP_DIR, "webcam_profiles.json")   # negotiated mode per webcam index
DEFAULT_CAM_SIZE = (1280, 720)
DEFAULT_CAM_FPS = 60            # request the max; the driver clamps to what the sensor/USB mode delivers
DEFAULT_IMGSZ = 512             # YOLO input: ~30% faster than 640 on CPU with the same 4-player recall
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
    kind: str                      # "webcam" | "stream" | "file" | "youtube"
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
                 cam_width=None, cam_height=None, cam_fps=None):
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
        self.open_timeout = float(open_timeout)
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
            return cap, frame
        return None, None

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
            while self.running and self._gen == gen:
                ok, frame = cap.read()
                if self._gen != gen or not self.running:
                    break
                if ok and frame is not None and frame.size > 0:
                    fails = 0
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
            self.bbox = bbox_arr
            self.bbox_vel[:] = 0.0
            self.pixel_trail.clear()
        else:
            new_bbox = 0.55 * bbox_arr + 0.45 * self.bbox
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
    """Neon ball = HSV colour AND (frame-difference motion OR Kalman-predicted trajectory), inside the
    court volume ROI, blob area/circularity/aspect filtered, YOLO 'sports ball' (class 32) as fallback.
    Static yellow logos and shoes are rejected (no motion, not on the predicted path, foot zones)."""
    HSV_LO, HSV_HI = (22, 90, 120), (48, 255, 255)
    AREA_MIN, AREA_MAX = 12, 450
    TRAIL_LEN = 15

    def __init__(self):
        self.reset()

    def reset(self):
        self.kf = None
        self.prev_gray = None
        self.last_ts = None
        self.hits = 0
        self.last_seen = 0.0
        self.pos = None
        self.court_pos = None
        self.trail = deque(maxlen=self.TRAIL_LEN)
        self.court_trail = deque(maxlen=25)
        self.speed_kmh = 0.0
        self._spd = deque(maxlen=5)
        self.vel = (0.0, 0.0)
        self.shots = 0
        self.total_shots = 0
        self._vy = deque(maxlen=4)
        self._last_shot = 0.0
        self.visible = False
        self.roi = None
        self.H = None
        self.frames = 0
        self.frames_visible = 0
        self.hitter = None
        self.slow_since = None
        self.suppress = []                 # (x, y, until_ts) spots where slow false tracks were dropped
        self._last_dir = 0
        self._last_dir_t = 0.0

    def set_geometry(self, H_i2c, frame_shape):
        """Court-volume ROI: the court polygon plus the space above it (lobs / smashes)."""
        self.H = H_i2c
        h, w = frame_shape[:2]
        self.roi = None
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
            self.roi = roi
        except Exception:
            self.roi = None

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

    def _px_to_m_scale(self, x, y):
        if self.H is None:
            return None
        try:
            a, b = cv2.perspectiveTransform(np.array([[[x, y], [x + 1.0, y]]], np.float32), self.H)[0]
            return float(np.linalg.norm(b - a))
        except cv2.error:
            return None

    def update(self, frame, ts, player_boxes=(), yolo_balls=()):
        self.frames += 1
        h, w = frame.shape[:2]
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        dt = 1 / 30.0 if self.last_ts is None else float(np.clip(ts - self.last_ts, 1 / 240.0, 0.5))
        self.last_ts = ts
        pred = None
        tracking = self.kf is not None and ts - self.last_seen < 0.45
        if tracking:
            self.kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            p = self.kf.predict()
            pred = (float(p[0, 0]), float(p[1, 0]))
        gate = min(260.0, 30.0 + 1600.0 * dt)

        # people standing outside the court (spectators, adjacent courts) wear neon too: ignore their boxes
        outsiders = []
        if self.H is not None and len(player_boxes):
            try:
                feet = np.array([[[(b[0] + b[2]) / 2.0, b[3]] for b in player_boxes]], np.float32)
                for b, g in zip(player_boxes, cv2.perspectiveTransform(feet, self.H)[0]):
                    if not (-0.6 <= g[0] <= COURT_WIDTH_M + 0.6 and -0.6 <= g[1] <= COURT_LENGTH_M + 0.6):
                        outsiders.append(b)
            except cv2.error:
                pass
        cands = []
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            motion = cv2.dilate((cv2.absdiff(gray, self.prev_gray) > 15).astype(np.uint8), None, iterations=2)
            color = cv2.inRange(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV), self.HSV_LO, self.HSV_HI)
            if self.roi is not None and self.roi.shape == color.shape:
                color = cv2.bitwise_and(color, self.roi)
            cnts, _ = cv2.findContours(color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                area = cv2.contourArea(c)
                if not (self.AREA_MIN <= area <= self.AREA_MAX):
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                if max(bw, bh) > 3.5 * max(1, min(bw, bh)) + 1:
                    continue
                per = cv2.arcLength(c, True)
                circ = 4 * math.pi * area / (per * per) if per > 0 else 0
                if area >= 25 and circ < 0.45:
                    continue
                cx, cy = x + bw / 2.0, y + bh / 2.0
                mfrac = float(motion[y:y + bh, x:x + bw].mean())
                near_pred = pred is not None and math.hypot(cx - pred[0], cy - pred[1]) < gate
                moving = mfrac > 0.3
                if not (moving or (near_pred and mfrac > 0.1)):
                    continue                              # static logo / paint / resting ball / still clothing
                in_foot = any(b[0] - 10 <= cx <= b[2] + 10 and b[3] - 0.30 * (b[3] - b[1]) <= cy <= b[3] + 10
                              for b in player_boxes)
                if in_foot and not (near_pred and self._track_speed() > 350):
                    continue                              # neon shoes (unless a fast ball is passing)
                if any(b[0] - 6 <= cx <= b[2] + 6 and b[1] - 6 <= cy <= b[3] + 6 for b in outsiders):
                    continue                              # spectator / neighbouring-court player
                if any(math.hypot(cx - zx, cy - zy) < 28 and ts < zt for zx, zy, zt in self.suppress) and                         not (near_pred and self._track_speed() > 350):
                    continue                              # learned static/slow false-positive spot
                in_body = any(b[0] <= cx <= b[2] and b[1] + 0.15 * (b[3] - b[1]) <= cy <= b[3] for b in player_boxes)
                if in_body and not near_pred and area > 160:
                    continue                              # big neon clothing patch
                cands.append((cx, cy, area, circ, moving))
        for b in yolo_balls:                              # YOLO class 32 fallback
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            cands.append((cx, cy, 40.0, 0.8, True))
        self.prev_gray = gray

        meas = None
        if pred is not None:
            best = None
            for cx, cy, area, circ, moving in cands:
                d = math.hypot(cx - pred[0], cy - pred[1])
                if d < gate and (best is None or d < best[0]):
                    best = (d, cx, cy)
            if best:
                meas = (best[1], best[2])
        if meas is None and cands and not (tracking and self.hits >= 3):
            moving = [c for c in cands if c[4]]
            if moving:
                cx, cy, *_ = max(moving, key=lambda c: c[3] + 0.002 * min(c[2], 120))
                self.kf, self.hits, meas = self._kf_new(cx, cy), 0, (cx, cy)
                self.trail.clear()
                self._vy.clear()
        if meas is not None:
            st = self.kf.correct(np.array([[meas[0]], [meas[1]]], np.float32))
            self.hits += 1
            self.last_seen = ts
            self.pos = meas
            vx, vy = float(st[2, 0]), float(st[3, 0])
            self.vel = (vx, vy)
            # a padel ball in play is never slow for long: drop tracks that crawl (shoes, debris, resting ball)
            if self.hits >= 3 and math.hypot(vx, vy) < 150:
                self.slow_since = self.slow_since or ts
            else:
                self.slow_since = None
            if self.slow_since and ts - self.slow_since > 0.4:
                self.suppress = [z for z in self.suppress if z[2] > ts][-30:] + [(meas[0], meas[1], ts + 10.0)]
                self.kf, self.hits, self.slow_since = None, 0, None
                self.trail.clear()
                self.visible = False
                return False
            if self.hits >= 1:
                self.trail.append((int(meas[0]), int(meas[1])))
                self._update_speed(meas, vx, vy, player_boxes)
                self._update_court(meas, player_boxes)
                self._update_shots(vy, ts, player_boxes)
        if ts - self.last_seen > 4.0 and self.shots:
            self.shots = 0                                # ball gone for a while -> next rally
        self.visible = (self.kf is not None and self.hits >= 3 and ts - self.last_seen < 0.2
                        and self.slow_since is None)
        if self.visible:
            self.frames_visible += 1
        elif ts - self.last_seen > 0.5:
            self.trail.clear()
        return self.visible

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
        # Ball height is unknown in a single camera: use the ground scale at the depth of the nearest
        # player (the ball is usually between / near players) -> approximate speed.
        nb = self._nearest_box(pt, boxes)
        row = nb[1][3] if nb is not None else pt[1]
        scale = self._px_to_m_scale(pt[0], row)
        if scale is None:
            return
        kmh = float(np.clip(math.hypot(vx, vy) * scale * 3.6, 0, 200))
        self._spd.append(kmh)
        self.speed_kmh = float(np.median(self._spd))

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
            self.court_pos = (float(np.clip(g[0], 0, COURT_WIDTH_M)), float(np.clip(g[1], 0, COURT_LENGTH_M)))
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

    def _update_shots(self, vy, ts, boxes):
        # (a) new fast segment heading the other way soon after the last one = a hit (track drops at contact)
        if self.hits == 4 and abs(vy) > 120:
            d = 1 if vy > 0 else -1
            if self._last_dir and d != self._last_dir and ts - self._last_dir_t < 2.0:
                self._register_shot(ts)
        if self.hits >= 4 and abs(vy) > 120:
            self._last_dir, self._last_dir_t = (1 if vy > 0 else -1), ts
        # (b) direction reversal inside one continuous track
        self._vy.append(vy)
        if len(self._vy) < 4:
            return
        a, b = np.mean(list(self._vy)[:2]), np.mean(list(self._vy)[2:])
        if a * b < 0 and min(abs(a), abs(b)) > 60:
            self._register_shot(ts)
            self._vy.clear()

    def state(self, now=None):
        return {"visible": bool(self.visible),
                "image_px": None if self.pos is None or not self.visible else [round(self.pos[0], 1), round(self.pos[1], 1)],
                "court_m": None if self.court_pos is None or not self.visible else [round(self.court_pos[0], 2), round(self.court_pos[1], 2)],
                "court_trail_m": [[round(x, 2), round(y, 2)] for x, y in list(self.court_trail)[-15:]] if self.visible else [],
                "speed_kmh": round(self.speed_kmh, 1), "rally_shots": self.shots, "total_shots": self.total_shots,
                "visibility_pct": round(100.0 * self.frames_visible / max(1, self.frames), 1)}


class BallWorker:
    """Runs the ball tracker on every captured frame (up to max_fps) in its own thread, independent of
    the slower person detector, so fast balls keep a dense trajectory."""

    def __init__(self, cam_getter, ball, boxes_getter, min_height=720, max_fps=60.0):
        self.cam_getter, self.ball, self.boxes_getter = cam_getter, ball, boxes_getter
        self.min_height, self.period = min_height, 1.0 / max_fps
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
                    self.ball.reset()
                frame, fid, ts = cam_wait_latest(cam, last_fid, 0.05)
                if frame is None or fid == last_fid:
                    continue
                last_fid = fid
                canvas = prepare_canvas(frame, self.min_height)
                boxes, yolo_balls = self.boxes_getter()
                self.ball.update(canvas, ts, boxes, yolo_balls)
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


def draw_glass_box(frame, x1, y1, x2, y2, bg=(18, 18, 18), alpha=0.68, border=None):
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = np.full(roi.shape, bg, dtype=np.uint8)
    frame[y1:y2, x1:x2] = cv2.addWeighted(overlay, alpha, roi, 1.0 - alpha, 0)
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


def draw_ball_hud(frame, ball, now, s, obstacles=()):
    """Glowing target ring, 15-frame comet trail and a 'BALL xx km/h | Rally shots N' tag."""
    if ball is None or not ball.visible or ball.pos is None:
        return
    tr = list(ball.trail)
    n = len(tr)
    for i in range(1, n):
        a = i / max(1, n - 1)                                     # 0 = oldest -> 1 = newest
        col = (int(255 * a), 255, int(255 * (1 - a) + 40 * a))    # cyan tail -> neon yellow head (BGR)
        cv2.line(frame, tr[i - 1], tr[i], col, max(1, int(round((1 + 4 * a) * s))), cv2.LINE_AA)
    cx, cy = int(ball.pos[0]), int(ball.pos[1])
    r = max(8, int(11 * s))
    glow = frame.copy()
    cv2.circle(glow, (cx, cy), int(r * 1.9), (0, 255, 255), max(2, int(5 * s)), cv2.LINE_AA)
    cv2.addWeighted(glow, 0.35, frame, 0.65, 0, frame)
    pulse = 1.0 + 0.15 * math.sin(now * 10)
    cv2.circle(frame, (cx, cy), int(r * pulse), (0, 255, 255), max(2, int(2 * s)), cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), int(r * 1.45), (255, 255, 0), max(1, int(1 * s)), cv2.LINE_AA)
    for ang in (0, 90, 180, 270):                                 # target ticks
        ux, uy = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        cv2.line(frame, (int(cx + ux * r * 1.5), int(cy + uy * r * 1.5)), (int(cx + ux * r * 2.1), int(cy + uy * r * 2.1)),
                 (255, 255, 0), max(1, int(2 * s)), cv2.LINE_AA)
    label = f"BALL {ball.speed_kmh:.0f} km/h | Rally shots {ball.shots}"
    fs = 0.45 * s
    (tw, th), _ = cv2.getTextSize(label, FONT, fs, 1)
    h, w = frame.shape[:2]
    tw2, th2 = tw + int(14 * s), th + int(12 * s)
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
    draw_glass_box(frame, bx, by, bx + tw + int(14 * s), by + th + int(12 * s), bg=(10, 10, 10), alpha=0.82,
                   border=(0, 255, 255))
    put_text(frame, label, (bx + int(7 * s), by + th + int(6 * s)), fs, (0, 255, 255), 1, s)


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
        glow = frame.copy()
        polys = []
        for idx_s, (a, b) in enumerate(segs):
            pts = CG.sample_clipped_segment(H_c2i, a, b, frame.shape, steps=44)
            if pts is not None:
                ipts = pts.astype(np.int32)
                polys.append((idx_s, ipts))
                cv2.polylines(glow, [ipts], False, (255, 255, 0), max(4, int(9 * s)), cv2.LINE_AA)
        cv2.addWeighted(glow, 0.35, frame, 0.65, 0, frame)
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
    l1 = fit_text(l1, fs1, 2, w - 60 * s)
    l2 = fit_text(l2, fs2, 1, w - 60 * s)
    bh = int((62 if l2 else 40) * s)
    y0 = int(10 * s)
    draw_glass_box(frame, int(14 * s), y0, w - int(14 * s), y0 + bh, bg=(10, 10, 10), alpha=0.88, border=col)
    put_text(frame, l1, (int(28 * s), y0 + int(26 * s)), fs1, col, 2, s)
    if l2:
        put_text(frame, l2, (int(28 * s), y0 + int(50 * s)), fs2, (235, 235, 235), 1, s)


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
    draw_glass_box(frame, *rect, bg=(160, 85, 30), alpha=0.80, border=(255, 255, 255))
    put_text(frame, "2D COURT MAP", (mx1 + pad, my1 + int(18 * s)), 0.45 * s, (255, 255, 255), 1, s)

    ix1, iy1 = mx1 + pad, my1 + pad + title_h

    def to_px(xm, ym):
        return (int(ix1 + (np.clip(xm, 0, COURT_WIDTH_M) / COURT_WIDTH_M) * iw),
                int(iy1 + (np.clip(ym, 0, COURT_LENGTH_M) / COURT_LENGTH_M) * ih))

    line_th = max(1, int(round(1 * s)))
    cv2.rectangle(frame, (ix1, iy1), (ix1 + iw, iy1 + ih), (245, 245, 245), max(1, int(round(2 * s))), cv2.LINE_AA)
    # Service lines (6.95 m from the net) and centre service divider
    for ym in (SERVICE_LINE_NEAR_M, SERVICE_LINE_FAR_M):
        cv2.line(frame, to_px(0, ym), to_px(COURT_WIDTH_M, ym), (230, 230, 230), line_th, cv2.LINE_AA)
    cv2.line(frame, to_px(COURT_WIDTH_M / 2, SERVICE_LINE_NEAR_M), to_px(COURT_WIDTH_M / 2, SERVICE_LINE_FAR_M),
             (230, 230, 230), line_th, cv2.LINE_AA)
    # Net
    cv2.line(frame, to_px(0, NET_Y_M), to_px(COURT_WIDTH_M, NET_Y_M), (255, 255, 255),
             max(2, int(round(3 * s))), cv2.LINE_AA)

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

    # Dots first, then labels placed so they avoid every dot and every other label.
    for sid, color, (cx, cy) in visible:
        glow = frame.copy()
        cv2.circle(glow, (cx, cy), int(dot_r * 2.2), color, -1, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.35, frame, 0.65, 0, frame)
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
            # penalise leaving the court rectangle
            cost += (lw * lh) * (r[0] < bounds[0] or r[2] > bounds[2] or r[1] < bounds[1] or r[3] > bounds[3])
            if best is None or cost < best_cost:
                best, best_cost = (lx, ly, r), cost
            if cost == 0:
                break
        lx, ly, r = best
        label_rects.append(r)
        put_text(frame, label, (lx, ly), fs, (255, 255, 255), 2, s)

    if ball is not None and ball.visible and ball.court_pos is not None:
        tr = [to_px(x, y) for x, y in ball.court_trail]
        for i in range(1, len(tr)):
            a = i / max(1, len(tr) - 1)
            cv2.line(frame, tr[i - 1], tr[i], (int(120 + 135 * a), 255, 255), max(1, int(round(1.5 * s))), cv2.LINE_AA)
        bx, by = to_px(*ball.court_pos)
        pr = int(max(3, 5 * s) * (1.0 + 0.35 * abs(math.sin(now * 6))))
        glow = frame.copy()
        cv2.circle(glow, (bx, by), pr * 2, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.45, frame, 0.55, 0, frame)
        cv2.circle(frame, (bx, by), pr, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (bx, by), pr + 1, (0, 255, 255), max(1, int(round(1.5 * s))), cv2.LINE_AA)
    return rect


def draw_coach_panel(frame, slot_mgr, now, s, fps_val, source_label, recording, rec_seconds, max_w):
    m = int(16 * s)
    x1, y1 = m, m
    pw = int(min(520 * s, max_w))
    lh = int(21 * s)
    tx = x1 + int(12 * s)
    inner_w = pw - int(24 * s)

    active = [p for p in slot_mgr.players.values() if p.is_active(now)] if slot_mgr else []
    rows = 2 * len(active) if active else 1
    ph = int(32 * s) + lh * 3 + int(8 * s) + rows * lh + int(10 * s)
    draw_glass_box(frame, x1, y1, x1 + pw, y1 + ph, bg=(18, 18, 18), alpha=0.86, border=(90, 90, 90))

    # Header + pulsing REC indicator
    rec_w = int(100 * s) if recording else 0
    header = fit_text(f"PadelIQ LIVE | {fps_val:.1f} FPS | {source_label}", 0.55 * s, 2, inner_w - rec_w)
    y = y1 + int(26 * s)
    put_text(frame, header, (tx, y), 0.55 * s, (0, 255, 180), 2, s)
    if recording:
        pulse = 0.5 + 0.5 * math.sin(now * 2 * math.pi * 1.2)
        rx = x1 + pw - rec_w
        r = max(4, int(7 * s))
        col = (0, 0, int(140 + 115 * pulse))
        cv2.circle(frame, (rx + r, y - int(5 * s)), int(r * (0.8 + 0.25 * pulse)), col, -1, cv2.LINE_AA)
        mm, ss = divmod(int(rec_seconds), 60)
        put_text(frame, f"REC {mm:02d}:{ss:02d}", (rx + 2 * r + int(5 * s), y), 0.48 * s, (60, 60, 255), 2, s)

    y += lh + int(2 * s)
    ctl = fit_text("[C] Calibrate  [R] Reset Stats  [S] Record  [Q] Quit", 0.42 * s, 1, inner_w)
    put_text(frame, ctl, (tx, y), 0.42 * s, (200, 200, 200), 1, s)
    y += lh + int(6 * s)
    put_text(frame, "AI Coach Suggestions:", (tx, y), 0.52 * s, (255, 255, 255), 2, s)
    y += lh

    if not active:
        msg = "Waiting for players inside calibrated zone..."
        put_text(frame, fit_text(msg, 0.44 * s, 1, inner_w), (tx, y), 0.44 * s, (180, 180, 180), 1, s)
    else:
        label_w = int(40 * s)
        for p in sorted(active, key=lambda q: q.slot_id):
            color = SLOT_COLORS[p.slot_id]
            put_text(frame, f"P{p.slot_id}", (tx, y), 0.46 * s, color, 2, s)
            for tip in coach_suggestions(p, now):
                bx = tx + label_w
                cv2.circle(frame, (bx + int(3 * s), y - int(5 * s)), max(2, int(3 * s)), color, -1, cv2.LINE_AA)
                put_text(frame, fit_text(tip, 0.43 * s, 1, inner_w - label_w - int(14 * s)),
                         (bx + int(12 * s), y), 0.43 * s, (235, 235, 235), 1, s)
                y += lh
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
        bx1, by1, bx2, by2 = best
        draw_glass_box(frame, bx1, by1, bx2, by2, bg=(15, 15, 15), alpha=0.82, border=color)
        lines = [
            f"P{sid} (Coverage: {p.coverage_pct:.1f}%)",
            f"Distance: {p.distance_m:.1f}m",
            f"AvgSpeed: {p.avg_speed_ms:.2f}m/s",
            f"Active: {int(p.active_seconds_total)}s ({p.active_pct}%)",
            f"Stability: {int(p.stability_pct)}%",
        ]
        for idx, txt in enumerate(lines):
            ty = by1 + int(18 * s) + idx * int(17.5 * s)
            put_text(frame, fit_text(txt, 0.42 * s, 1, cw - int(14 * s)), (bx1 + int(8 * s), ty), 0.42 * s,
                     color if idx == 0 else (240, 240, 240), 2 if idx == 0 else 1, s)


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
        label = f"P{sid}  {p.instant_speed_ms:.1f} m/s"
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
        draw_glass_box(frame, bx, by, bx + bw, by + bh, bg=(12, 12, 12), alpha=0.82, border=color)
        cv2.rectangle(frame, (bx, by), (bx + max(3, int(4 * s)), by + bh - 1), color, -1)
        put_text(frame, label, (bx + int(9 * s), by + bh - int(7 * s)), fs, (245, 245, 245), 1, s)


def draw_compact_header(frame, s, fps_val, source_label):
    text = fit_text(f"PadelIQ LIVE  |  {fps_val:.0f} FPS AI  |  {source_label}", 0.5 * s, 1, frame.shape[1] * 0.6)
    (tw, th), _ = cv2.getTextSize(text, FONT, 0.5 * s, 1)
    x1, y1 = frame.shape[1] - tw - int(30 * s) - int(12 * s), int(12 * s)
    draw_glass_box(frame, x1, y1, x1 + tw + int(30 * s), y1 + th + int(16 * s), bg=(10, 10, 10), alpha=0.82)
    cv2.circle(frame, (x1 + int(11 * s), y1 + (th + int(16 * s)) // 2), max(3, int(4 * s)), (60, 220, 60), -1, cv2.LINE_AA)
    put_text(frame, text, (x1 + int(22 * s), y1 + th + int(8 * s)), 0.5 * s, (235, 235, 235), 1, s)


def draw_reconnect_banner(frame, now, s, detail=""):
    """Non-blocking translucent banner with a spinner."""
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
    put_text(frame, fit_text("Reconnecting to Live Stream" + dots, 0.75 * s, 2, bw - int(76 * s)),
             (tx, y1 + int(38 * s)), 0.75 * s, (0, 215, 255), 2, s)
    if detail:
        put_text(frame, fit_text(detail, 0.45 * s, 1, bw - int(76 * s)), (tx, y1 + int(66 * s)), 0.45 * s,
                 (210, 210, 210), 1, s)


def render_live_hud(frame, slot_mgr, corners_px, fps_val, H=None, source_label="LIVE",
                    recording=False, rec_seconds=0.0, reconnecting=False, reconnect_detail="",
                    hud_mode="full", ball=None, review=None, web=False, now=None):
    """
    Draws the PadelIQ HUD in place. Works for any resolution (scales from 720p).
    hud_mode="full": coach panel + minimap + stat cards (desktop).
    hud_mode="compact": court, boxes and small speed tags (web dashboard shows the rest).
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
        draw_compact_header(frame, s, fps_val, source_label)
        draw_ball_hud(frame, ball, now, s)
        draw_calibration_review(frame, review, now, s, web=web)
        if reconnecting:
            draw_reconnect_banner(frame, now, s, reconnect_detail)
        return {"coach": (0, 0, 0, 0), "minimap": (w, 0, w, 0)}

    if slot_mgr is not None:
        mini_rect = draw_minimap(frame, slot_mgr, now, s, ball=ball)
    else:
        mini_rect = (w, 0, w, 0)
    coach_rect = draw_coach_panel(frame, slot_mgr, now, s, fps_val, source_label, recording, rec_seconds,
                                  max_w=mini_rect[0] - int(32 * s))
    if slot_mgr is not None:
        # Stat cards are placed so they never cover the panels or each other.
        draw_player_cards(frame, slot_mgr, now, s, [coach_rect, mini_rect])
    draw_ball_hud(frame, ball, now, s, obstacles=[coach_rect, mini_rect])      # on top of everything
    draw_calibration_review(frame, review, now, s, web=web)
    if reconnecting:
        draw_reconnect_banner(frame, now, s, reconnect_detail)
    return {"coach": coach_rect, "minimap": mini_rect}


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

    def __init__(self, model_path="yolov8n.pt", imgsz=640, conf=0.28):
        self.imgsz, self.conf = imgsz, conf
        self._errors = 0
        self.last_balls = np.zeros((0, 4), dtype=np.float32)
        self.last_keypoints = []
        self.pose_det = CG.PoseCourtDetector(app_dir=APP_DIR, model_path=model_path, imgsz=imgsz, conf=conf)
        self.model = self.pose_det.det_model or self.pose_det.pose_model
        get_court_ai(self.pose_det)
        if self.pose_det.ready:
            log(f"AI Detector + Pose Engine ready (pose backend: {self.pose_det.pose_backend})")
        else:
            log("WARNING: YOLO/Pose models unavailable. HUD will run without player detection.")

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
def prepare_canvas(raw, min_height):
    """Upscale low-res streams (ESP32 640x480 / 800x600) so HUD text stays crisp."""
    h, w = raw.shape[:2]
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
    elif raw.shape[2] == 4:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
    if min_height and h < min_height:
        f = min_height / float(h)
        return cv2.resize(raw, (int(round(w * f)), int(min_height)), interpolation=cv2.INTER_LINEAR)
    return raw


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
        self.lock = threading.RLock()       # display loop <-> async AI worker (slot manager / geometry)

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
                    collected.append(prepare_canvas(raw, self.args.hud_min_height))
                else:
                    time.sleep(0.012)
            if len(collected) >= 5:
                return collected
        if self.spec.kind == "file":
            return [prepare_canvas(f, self.args.hud_min_height) for f in collect_calibration_frames(self.spec, count=16)]
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
        elif self.review.state in ("review", "confirmed", "failed"):
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
        if rv.state == "manual" and key == 27:
            with rv.lock:
                rv.state, rv.review_since = "review", time.time()
            return
        if key in (13, 10) or ch == "y" or (rv.state == "manual" and ch == "c"):
            if rv.confirm("operator") and self.canvas_shape is not None:
                self.save_confirmed(self.canvas_shape)
            return
        if ch in ("m", "c"):
            if clean_canvas is None:
                log("Calibration needs a live frame - waiting for the stream.")
                return
            rv.enter_manual(self.corners, clean_canvas.shape)
            log("Manual calibration: drag TL/TR/BR/BL/NET-L/NET-R handles ([N] = click 4 fresh corners, [Enter/C] lock)")
            return
        if ch == "a":
            log("Re-scanning court geometry ...")
            self.start_auto_calibration(seed_frame=clean_canvas)
            return
        if rv.state == "manual" and ch == "n":
            rv.toggle_click_mode()
            return
        if rv.state == "manual" and ch == "h":
            rv.cycle_reference()
            return
        if key == 27 or ch == "q":
            self.quit = True
        elif ch == "r":
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
            self.pending = (canvas, ts)          # overwrite: never process a stale frame
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
            canvas, ts = job
            try:
                t0 = time.perf_counter()
                boxes = self.detector(canvas)
                self.latency_ms = 0.8 * self.latency_ms + 0.2 * (time.perf_counter() - t0) * 1000 if self.inferred \
                    else (time.perf_counter() - t0) * 1000
                with s.lock:
                    if s.canvas_shape == canvas.shape[:2]:
                        s.last_boxes, s.last_yolo_balls = boxes, self.detector.last_balls
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

    detector = PersonDetector(args.model, imgsz=args.imgsz, conf=args.conf)
    cam = ThreadedLiveCamera(spec, stall_timeout=args.stall_timeout,
                             cam_width=args.cam_width, cam_height=args.cam_height, cam_fps=args.cam_fps)
    session = LiveSession(spec, args, cam_getter=lambda: cam)
    ai = AsyncAIWorker(detector, session)
    e2e_ms = deque(maxlen=600)          # capture/arrival -> frame shown (per displayed camera frame)
    ball_worker = BallWorker(lambda: cam, session.ball,
                             lambda: (session.last_boxes, session.last_yolo_balls), args.hud_min_height)
    win_name = "PadelIQ Live Camera Engine"
    if display:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(win_name, lambda ev, x, y, fl, pa: session.review.on_mouse(ev, x, y))
        log("Live engine running:  [Enter/Y] confirm court  [M] adjust court  [A] re-run auto-calibration  "
            "[R] reset  [S] record  [Q] quit")

    # Scripted key presses exercise the same handlers headlessly during --test-run.
    scripted_keys = {}
    if test_mode:
        n = args.test_run
        scripted_keys = {max(2, n // 10): ord('s'), max(3, n // 4): ord('r'), max(4, (n * 3) // 5): ord('s')}

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

    try:
        while not session.quit:
            # Wake the instant the camera publishes; never queue (newest frame only).
            raw, fid, cap_ts = cam_wait_latest(cam, last_fid, 1.0 / 60.0)
            ok = raw is not None
            new_frame = ok and fid != last_fid
            prev_processed = processed
            processed = ai.inferred
            if new_frame:
                last_fid, last_cap_ts = fid, cap_ts
                canvas = prepare_canvas(raw, args.hud_min_height)
                with session.lock:
                    session.ensure_geometry(canvas)
                    session.note_frame(canvas)
                    session.sync_review(canvas.shape)
                ai.submit(canvas, cap_ts)                # async YOLO + pose on the newest frame
                last_canvas = canvas
                displayed += 1
                t = time.time()
                if displayed > 1:
                    frame_intervals.append(t - fps_t0)
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
                                review=session.review if last_canvas is not None else None)
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

            if new_frame:
                e2e_ms.append((time.time() - last_cap_ts) * 1000.0)
            key = -1
            if display:
                cv2.imshow(win_name, hud)
                key = cv2.waitKey(1) & 0xFF
                if shown and cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                    log("Window closed.")
                    break
                shown = True
            else:
                due = [k for k in scripted_keys if k <= processed and k > prev_processed] if processed != prev_processed else []
                if due:
                    key = scripted_keys.pop(due[0])
                    log(f"[test] simulating key '{chr(key).upper()}' at frame {processed}")

            was_recording = session.recorder.active
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
                   help="auto (default) | sim | webcam (best colour webcam, Logitech BRIO preferred) | 0/1 (webcam index) | "
                        "http://<esp32-ip>:81/stream | rtsp://... | ESP32 IP | YouTube URL | video file")
    p.add_argument("--esp32-ip", default=None, help="ESP32-CAM IP or URL tried first in auto mode (env PADELIQ_STREAM_URL also works)")
    p.add_argument("--video", default=None, help="video file for simulated live mode (default: padel_input.mp4)")
    p.add_argument("--model", default="yolov8n.pt", help="YOLO model (yolov8n.pt is fastest for live use)")
    p.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="YOLO inference size (512 default; 640 = slower)")
    p.add_argument("--conf", type=float, default=0.30, help="YOLO person confidence threshold")
    p.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "any"], help="webcam backend")
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
