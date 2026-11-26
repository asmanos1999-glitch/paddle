# -*- coding: utf-8 -*-
"""
PadelIQ Mobile Bridge - zero-app QR phone-camera pairing + court calibration
============================================================================

Turns any phone into the PadelIQ camera: scan the on-screen QR code, accept the local
certificate once, and the phone's rear camera streams straight into the vision engine.

    LAN detection    get_lan_ip()                       -> 192.168.x.x of the active Wi-Fi/Ethernet adapter
    HTTPS            ensure_self_signed_cert()          -> certs/cert.pem + certs/key.pem (SAN = LAN IP, localhost)
                                                           (getUserMedia needs a secure context on LAN IPs)
    QR               qr_png_bytes() / qr_bgr()          -> https://<LAN_IP>:8443/mobile-camera?k=<pairing token>
    Camera source    MobileHub + MobileCamera           -> drop-in for ThreadedLiveCamera (read_latest, status, ...)
    Phone web app    build_mobile_router()              -> GET /mobile-camera, WS /ws/mobile-cam, GET /api/qr.png ...
    Waiting screen   render_waiting_screen()            -> big scannable QR inside the live video feed
    Scale fiducial   FiducialScaleDetector              -> cv2.QRCodeDetector: corners, px/cm, planar tilt, and a
                                                           metric cross-check of the court homography (10 x 20 m)

This module intentionally does not import padeliq_live_stream / the enterprise server, so both the
desktop engine and the web command center can use it without circular imports. The host plugs in
through a small "controller" object (see MobileController).

WebSocket protocol (/ws/mobile-cam?k=<token>)
    phone -> server   binary            one JPEG frame
                      {"type":"meta", "w", "h", "orientation", "lens", "zoom", "lenses", "ua"}
                      {"type":"auto_detect"}
                      {"type":"corners", "pts": [[nx, ny] x4]}      normalised 0..1 (TL, TR, BR, BL)
                      {"type":"lock"}
                      {"type":"ping", "t"}
    server -> phone   {"type":"welcome", ...}
                      {"type":"ack", "n"}                           one per decoded frame (flow control)
                      {"type":"calib", "state", "corners", ...}     ~5 Hz
                      {"type":"result", "action", "ok", "message"}
                      {"type":"pong", "t"}
"""
import os
import io
import re
import json
import time
import socket
import asyncio
import secrets
import datetime as _dt
import ipaddress
import threading
from collections import deque

import numpy as np
import cv2

MOBILE_HTTPS_PORT = 8443
DEFAULT_FIDUCIAL_SIZE_CM = 20.0
COURT_W_M, COURT_L_M = 10.0, 20.0
FIDUCIAL_PAYLOAD_RE = re.compile(r"PADELIQ[-_ ]?SCALE[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*(cm|mm|m)?", re.I)


def _log(msg):
    print(f"[PadelIQ {_dt.datetime.now():%H:%M:%S}] [mobile] {msg}", flush=True)


# =====================================================================
# 1. LAN IP + self-signed certificate
# =====================================================================
def _is_private_v4(ip):
    try:
        a = ipaddress.ip_address(ip)
        return a.version == 4 and a.is_private and not a.is_loopback and not a.is_link_local
    except ValueError:
        return False


VIRTUAL_NIC_RE = re.compile(
    r"vmware|vmnet|virtualbox|vbox|hyper-?v|vethernet|wsl|docker|default switch|loopback|pseudo|"
    r"\btap\b|tap-|\btun\b|wireguard|openvpn|vpn|zerotier|tailscale|hamachi|npcap|bluetooth|teredo|isatap", re.I)
WIFI_NIC_RE = re.compile(r"wi-?fi|wlan|wireless|802\.11", re.I)
LAN_ADAPTER = ""                                # name of the adapter chosen for the QR (for the log)


def _route_ip():
    """Source IP the OS would use for the default route (UDP connect to 8.8.8.8:80; no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return ""


def _lan_adapters():
    """[(name, ipv4)] of adapters that are up, private, not loopback / link-local (169.254)."""
    try:
        import psutil
        stats = psutil.net_if_stats()
        out = []
        for name, addrs in psutil.net_if_addrs().items():
            st = stats.get(name)
            if st is not None and not st.isup:
                continue
            for a in addrs:
                if a.family == socket.AF_INET and _is_private_v4(a.address):
                    out.append((name, a.address))
        return out
    except Exception:
        return []


def get_lan_ip(override=None):
    """IPv4 the phone should use to reach this PC.
    1. --lan-ip / PADELIQ_LAN_IP override.
    2. The default-route adapter (UDP connect to 8.8.8.8:80) - unless it is a virtual NIC.
    3. Otherwise the best real adapter: VMware / Hyper-V vEthernet / WSL / VirtualBox / VPN / link-local (169.254)
       / loopback are skipped; Wi-Fi beats Ethernet, then 192.168.* > 10.* > 172.16-31.*.
    Adapters are filtered by NAME, not by subnet: many venue / campus Wi-Fi networks use 172.16-31.x, while VMware's
    host-only adapters use 192.168.x.1."""
    global LAN_ADAPTER
    override = override or os.environ.get("PADELIQ_LAN_IP", "").strip()
    if override:
        LAN_ADAPTER = "override"
        return override
    route = _route_ip()
    adapters = _lan_adapters()
    real = [(n, ip) for n, ip in adapters if not VIRTUAL_NIC_RE.search(n)]
    if adapters:
        for n, ip in real:
            if ip == route:
                LAN_ADAPTER = n
                return ip

        def rank(item):
            n, ip = item
            net = 0 if ip.startswith("192.168.") else (1 if ip.startswith("10.") else 2)
            return (0 if WIFI_NIC_RE.search(n) else 1, net, ip.endswith(".1"))
        if real:
            LAN_ADAPTER, ip = sorted(real, key=rank)[0]
            return ip
    if _is_private_v4(route):                   # psutil unavailable: trust the default route
        LAN_ADAPTER = "default route"
        return route
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ip = info[4][0]
            if _is_private_v4(ip) and not ip.endswith(".1"):
                LAN_ADAPTER = "hostname"
                return ip
    except OSError:
        pass
    LAN_ADAPTER = "loopback"
    return "127.0.0.1"


def _cert_covers(cert_path, ip):
    try:
        from cryptography import x509
        with open(cert_path, "rb") as f:
            cert = x509.load_pem_x509_certificate(f.read())
        san = cert.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
        ips = {str(a) for a in san.get_values_for_type(x509.IPAddress)}
        not_after = getattr(cert, "not_valid_after_utc", None) or cert.not_valid_after.replace(tzinfo=_dt.timezone.utc)
        return ip in ips and not_after > _dt.datetime.now(_dt.timezone.utc) + _dt.timedelta(days=7)
    except Exception:
        return False


def ensure_self_signed_cert(cert_dir, lan_ip):
    """Create (or refresh when the LAN IP changed) certs/cert.pem + certs/key.pem. Returns (cert, key)."""
    os.makedirs(cert_dir, exist_ok=True)
    cert_path, key_path = os.path.join(cert_dir, "cert.pem"), os.path.join(cert_dir, "key.pem")
    if os.path.isfile(cert_path) and os.path.isfile(key_path) and _cert_covers(cert_path, lan_ip):
        return cert_path, key_path
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError:
        return _openssl_cert(cert_path, key_path, lan_ip)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, f"PadelIQ local ({lan_ip})"),
                      x509.NameAttribute(NameOID.ORGANIZATION_NAME, "PadelIQ")])
    alt = [x509.IPAddress(ipaddress.ip_address(lan_ip)), x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
           x509.DNSName("localhost")]
    host = socket.gethostname()
    if host and re.fullmatch(r"[A-Za-z0-9.-]+", host):
        alt.append(x509.DNSName(host))
    now = _dt.datetime.now(_dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name).public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - _dt.timedelta(minutes=5)).not_valid_after(now + _dt.timedelta(days=825))
            .add_extension(x509.SubjectAlternativeName(alt), critical=False)
            .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
            .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), critical=False)
            .sign(key, hashes.SHA256()))
    with open(key_path, "wb") as f:
        f.write(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.TraditionalOpenSSL,
                                  serialization.NoEncryption()))
    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    _log(f"Generated self-signed HTTPS certificate for {lan_ip} -> {cert_path}")
    return cert_path, key_path


def _openssl_cert(cert_path, key_path, lan_ip):
    import shutil
    import subprocess
    exe = shutil.which("openssl")
    if not exe:
        raise RuntimeError("neither the 'cryptography' package nor openssl is available (pip install cryptography)")
    subprocess.run([exe, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825",
                    "-keyout", key_path, "-out", cert_path, "-subj", f"/CN=PadelIQ local ({lan_ip})",
                    "-addext", f"subjectAltName=IP:{lan_ip},IP:127.0.0.1,DNS:localhost"],
                   check=True, capture_output=True)
    _log(f"Generated self-signed HTTPS certificate with openssl -> {cert_path}")
    return cert_path, key_path


# =====================================================================
# 2. QR code images
# =====================================================================
def _qr_matrix(text, border=4):
    import qrcode
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_M, border=border, box_size=1)
    qr.add_data(text)
    qr.make(fit=True)
    return np.array(qr.get_matrix(), dtype=bool)


def qr_bgr(text, size=480, border=4):
    """Crisp (nearest-neighbour, integer module size) black-on-white QR as a BGR image of ~size px."""
    m = _qr_matrix(text, border)
    mod = max(1, size // m.shape[0])
    img = np.where(m, 0, 255).astype(np.uint8)
    img = cv2.resize(img, (m.shape[1] * mod, m.shape[0] * mod), interpolation=cv2.INTER_NEAREST)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def qr_png_bytes(text, size=512, border=4):
    ok, buf = cv2.imencode(".png", qr_bgr(text, size, border))
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()


# =====================================================================
# 3. Phone camera source (drop-in for ThreadedLiveCamera)
# =====================================================================
class MobileCamera:
    """Same public surface the engines use on ThreadedLiveCamera: read_latest(), read(), frame_id,
    status, is_degraded, seconds_since_frame, reconnect_count, native_fps, last_error, release()."""

    kind = "mobile"
    exposure_mode = "auto"
    exposure_label = "Phone auto exposure"
    exposure_info = {"mode": "auto", "label": "Phone auto exposure", "applies": False}

    def set_exposure_mode(self, mode):          # phone cameras manage exposure themselves
        return False

    def toggle_exposure(self):
        return False

    def open_driver_settings(self):
        return False

    def __init__(self, spec, hub, stall_timeout=2.5):
        self.spec = spec
        self.source = getattr(spec, "source", "mobile")
        self.hub = hub
        self.stall_timeout = float(stall_timeout)
        self.lock = threading.Condition()       # single-slot latest frame; consumers wake on publish
        self.frame = None
        self.ret = False
        self.frame_id = 0
        self.last_frame_time = 0.0
        self.status = "waiting"                 # waiting (for phone) | live | reconnecting
        self.reconnect_count = 0
        self.active_api = None
        self.last_error = ""
        self.running = True
        self._ts = deque(maxlen=60)

    def publish(self, frame, ts=None):
        if not self.running:
            return
        now = time.time() if ts is None else ts
        with self.lock:
            self.frame = frame
            self.ret = True
            self.frame_id += 1
            self.last_frame_time = now
            self._ts.append(now)
            self.lock.notify_all()
        if self.status != "live":
            if self.status == "reconnecting":
                self.reconnect_count += 1
                _log("Phone stream restored.")
            self.status = "live"

    def mark_disconnected(self):
        if self.status == "live":
            self.status = "reconnecting"

    @property
    def native_fps(self):
        with self.lock:
            if len(self._ts) < 2 or time.time() - self._ts[-1] > 2.0:
                return 0.0
            return (len(self._ts) - 1) / max(1e-3, self._ts[-1] - self._ts[0])

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def read_latest(self):
        with self.lock:
            if self.frame is None:
                return None, 0, 0.0
            return self.frame.copy(), self.frame_id, self.last_frame_time

    def read_with_id(self):
        with self.lock:
            if self.frame is None:
                return False, None, 0
            return True, self.frame.copy(), self.frame_id

    def wait_latest(self, last_id, timeout=0.05):
        with self.lock:
            if self.frame is None or self.frame_id == last_id:
                self.lock.wait(timeout)
            if self.frame is None:
                return None, 0, 0.0
            return self.frame.copy(), self.frame_id, self.last_frame_time

    @property
    def seconds_since_frame(self):
        return float("inf") if self.last_frame_time == 0 else time.time() - self.last_frame_time

    @property
    def is_degraded(self):
        return self.status != "live" or self.seconds_since_frame > self.stall_timeout

    @property
    def waiting_for_phone(self):
        """True -> show the QR pairing screen instead of a (stale or missing) frame."""
        return self.frame_id == 0 or (not self.hub.connected and self.seconds_since_frame > 1.5) \
            or self.seconds_since_frame > max(4.0, 2 * self.stall_timeout)

    def release(self):
        self.running = False
        self.hub.detach(self)


class MobileController:
    """Interface the host engine implements so the phone can drive court calibration.
    All corner arrays are canvas pixels (TL, TR, BR, BL -> court (0,0), (10,0), (10,20), (0,20))."""

    def calibration_snapshot(self):          # -> dict(state, corners|None, frame_shape (h, w)|None, ...)
        return {"state": "idle", "corners": None, "frame_shape": None}

    def auto_detect(self):                   # may block (run in a worker thread); -> message str
        raise NotImplementedError

    def preview_corners(self, corners_px):   # live drag (not locked yet)
        raise NotImplementedError

    def lock(self, corners_px=None):         # -> message str; raises ValueError on failure
        raise NotImplementedError

    def on_phone_connected(self, meta):      # e.g. auto-switch the engine to the mobile source (must not block)
        pass

    def hud_jpeg(self):                      # optional: latest annotated JPEG for the phone's HUD view
        return None


class MobileHub:
    """Pairing state, phone metadata, latest frame and the currently attached MobileCamera."""

    def __init__(self, https_port=MOBILE_HTTPS_PORT, lan_ip=None, token=None, fiducial_size_cm=DEFAULT_FIDUCIAL_SIZE_CM):
        self.lan_ip = get_lan_ip(lan_ip)
        _log(f"LAN IP for the QR code: {self.lan_ip} (adapter: {LAN_ADAPTER or '?'}; override with --lan-ip)")
        self.https_port = int(https_port)
        self.token = token or secrets.token_urlsafe(9)
        self.https_enabled = False
        self.https_error = ""
        self.controller = MobileController()
        self.lock = threading.Lock()
        self.cam = None
        self.clients = 0
        self.meta = {}
        self.frames_received = 0
        self.bytes_received = 0
        self.connected_at = 0.0
        self.last_frame_at = 0.0
        self.last_frame_shape = None
        self.decode_ms = 0.0                    # EMA of JPEG decode time per phone frame
        self.frame_kb = 0.0                     # EMA of phone JPEG size
        self.fiducial = FiducialScaleDetector(default_size_cm=fiducial_size_cm)
        self._qr_cache = {}

    # ------------------------------------------------------------ pairing
    @property
    def url(self):
        return f"https://{self.lan_ip}:{self.https_port}/mobile-camera?k={self.token}"

    @property
    def connected(self):
        return self.clients > 0

    def qr_png(self, size=512):
        key = ("png", self.url, size)
        if key not in self._qr_cache:
            self._qr_cache[key] = qr_png_bytes(self.url, size=size)
        return self._qr_cache[key]

    def qr_image(self, size=420):
        key = ("bgr", self.url, size)
        if key not in self._qr_cache:
            self._qr_cache[key] = qr_bgr(self.url, size=size)
        return self._qr_cache[key]

    def check_token(self, token):
        return bool(token) and secrets.compare_digest(str(token), self.token)

    # ------------------------------------------------------------- camera
    def create_camera(self, spec, stall_timeout=2.5):
        cam = MobileCamera(spec, self, stall_timeout=stall_timeout)
        with self.lock:
            self.cam = cam
        return cam

    def detach(self, cam):
        with self.lock:
            if self.cam is cam:
                self.cam = None

    def push_frame(self, frame, arrived=None):
        """arrived = wall time the JPEG bytes were received (before decode) -> E2E latency starts there."""
        now = time.time() if arrived is None else arrived
        self.frames_received += 1
        self.last_frame_at = now
        self.last_frame_shape = frame.shape[:2]
        cam = self.cam
        if cam is not None:
            cam.publish(frame, now)

    def client_connected(self):
        with self.lock:
            self.clients += 1
            self.connected_at = time.time()

    def client_disconnected(self):
        with self.lock:
            self.clients = max(0, self.clients - 1)
            if self.clients == 0 and self.cam is not None:
                self.cam.mark_disconnected()

    def status(self, include_token=False):
        cam = self.cam
        d = {"url": self.url if include_token else self.url.split("?")[0], "lan_ip": self.lan_ip,
             "https_port": self.https_port, "https_enabled": self.https_enabled, "https_error": self.https_error,
             "connected": self.connected, "clients": self.clients, "meta": self.meta,
             "frames_received": self.frames_received,
             "stream_fps": round(cam.native_fps, 1) if cam is not None else 0.0,
             "decode_ms": round(self.decode_ms, 2), "frame_kb": round(self.frame_kb, 1),
             "seconds_since_frame": None if not self.last_frame_at else round(time.time() - self.last_frame_at, 2),
             "frame_size": None if self.last_frame_shape is None else [self.last_frame_shape[1], self.last_frame_shape[0]],
             "source_active": cam is not None, "fiducial": self.fiducial.public_state()}
        if include_token:
            d["token"] = self.token
        return d


# =====================================================================
# 4. QR pairing screen drawn inside the live video feed
# =====================================================================
_FONT_CACHE = {}


def _pil_font(size, bold=False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    font = None
    try:
        from PIL import ImageFont
        names = (["segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold
                 else ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"])
        windir = os.environ.get("WINDIR", r"C:\Windows")
        for n in names:
            for path in (os.path.join(windir, "Fonts", n), n):
                try:
                    font = ImageFont.truetype(path, size)
                    break
                except OSError:
                    continue
            if font:
                break
    except ImportError:
        font = None
    _FONT_CACHE[key] = font
    return font


def _draw_texts(img, items):
    """items: [(text, (x, y_top), size, (b, g, r), bold)] - anti-aliased PIL text, cv2 fallback."""
    try:
        from PIL import Image, ImageDraw
        fonts_ok = all(_pil_font(sz, b) is not None for _, _, sz, _, b in items)
    except ImportError:
        fonts_ok = False
    if fonts_ok:
        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil)
        for text, (x, y), sz, (b, g, r), bold in items:
            d.text((x, y), text, font=_pil_font(sz, bold), fill=(r, g, b))
        img[:] = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
        return
    for text, (x, y), sz, color, bold in items:
        cv2.putText(img, text, (x, y + sz), cv2.FONT_HERSHEY_SIMPLEX, sz / 30.0, color, 2 if bold else 1, cv2.LINE_AA)


def _wrap_px(text, size, bold, max_w):
    """Greedy word wrap by rendered pixel width (PIL font metrics; ~0.55 em/char estimate as fallback)."""
    font = _pil_font(size, bold)
    width = (lambda t: font.getlength(t)) if font is not None else (lambda t: len(t) * size * 0.55)
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if cur and width(cand) > max_w:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return lines


_WAIT_CACHE = {}


def render_waiting_screen(hub, size=(1280, 720), detail=""):
    """Full-frame pairing screen: big QR + instructions. Static part is cached; only the status line animates."""
    w, h = int(size[0]), int(size[1])
    key = (hub.url, w, h, hub.https_enabled)
    base = _WAIT_CACHE.get(key)
    if base is None:
        s = h / 720.0
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
        base = np.zeros((h, w, 3), np.uint8)
        base[:, :, 0] = (38 + 30 * y).astype(np.uint8)
        base[:, :, 1] = (24 + 18 * y).astype(np.uint8)
        base[:, :, 2] = (14 + 8 * y).astype(np.uint8)
        qsz = int(min(h * 0.74, w * 0.42))
        qr = hub.qr_image(qsz)
        qh, qw = qr.shape[:2]
        qx, qy = int(70 * s), (h - qh) // 2
        cv2.rectangle(base, (qx - 10, qy - 10), (qx + qw + 10, qy + qh + 10), (90, 220, 120), max(2, int(4 * s)),
                      cv2.LINE_AA)
        base[qy:qy + qh, qx:qx + qw] = qr
        tx = qx + qw + int(60 * s)
        max_w = w - tx - int(30 * s)
        items = [("PadelIQ  |  MOBILE QR CAMERA", (tx, int(70 * s)), int(22 * s), (120, 220, 255), True)]
        yy = int(108 * s)
        for ln in _wrap_px("Scan this QR code with your phone camera", int(36 * s), True, max_w):
            items.append((ln, (tx, yy), int(36 * s), (255, 255, 255), True))
            yy += int(46 * s)
        for ln in _wrap_px("to stream live & auto-calibrate the court", int(28 * s), False, max_w):
            items.append((ln, (tx, yy), int(28 * s), (235, 235, 235), False))
            yy += int(38 * s)
        yy += int(22 * s)
        steps = ["1. Phone on the same Wi-Fi as this laptop",
                 "2. Open the link, accept the local security warning",
                 "   (self-signed certificate: Advanced > Proceed / Show Details > Visit)",
                 "3. Allow camera access, hold the phone in landscape",
                 "4. Tap Auto-Detect Court, fine-tune corners, then Lock"]
        for st in steps:
            items.append((st, (tx, yy), int(21 * s), (200, 210, 220), False))
            yy += int(31 * s)
        yy += int(16 * s)
        items.append((hub.url.split("?")[0], (tx, yy), int(22 * s), (90, 220, 120), True))
        if not hub.https_enabled:
            items.append(("HTTPS bridge is not running: " + (hub.https_error or "starting ...")[:70],
                          (tx, yy + int(36 * s)), int(19 * s), (80, 120, 255), False))
        _draw_texts(base, items)
        _WAIT_CACHE.clear()
        _WAIT_CACHE[key] = base
    img = base.copy()
    s = h / 720.0
    dots = "." * (1 + int(time.time() * 2) % 3)
    msg = detail or f"Waiting for phone{dots}"
    col = (60, 200, 255) if "disconnect" in msg.lower() or "paused" in msg.lower() else (90, 220, 120)
    cx = int(w - 360 * s)
    cv2.circle(img, (cx, int(h - 38 * s)), int(7 * s), col, -1, cv2.LINE_AA)
    cv2.putText(img, msg[:42], (cx + int(18 * s), int(h - 30 * s)), cv2.FONT_HERSHEY_SIMPLEX, 0.62 * s, col,
                max(1, int(2 * s)), cv2.LINE_AA)
    return img


def waiting_detail(cam, hub):
    if cam is None or cam.frame_id == 0:
        return "Phone connected - starting camera ..." if hub.connected else ""
    if hub.connected:
        return "Phone paused (screen off / app in background)"
    return "Phone disconnected - scan again to reconnect"


# =====================================================================
# 5. Physical / on-screen QR scale fiducial (cv2.QRCodeDetector)
# =====================================================================
class FiducialScaleDetector:
    """Finds a QR code lying on the court floor, measures px/cm and planar tilt, builds the metric
    floor-plane homography of the marker (image px -> cm) and cross-checks the court homography:
    the marker's known edge length measured *through* the court H must match its printed size.

    Physical size: decoded payload 'PADELIQ-SCALE:<n>cm' (see /api/fiducial.png) or the default size."""

    def __init__(self, default_size_cm=DEFAULT_FIDUCIAL_SIZE_CM, assumed_hfov_deg=70.0):
        self.default_size_cm = float(default_size_cm)
        self.assumed_hfov_deg = float(assumed_hfov_deg)
        self.detectors = [cv2.QRCodeDetector()]
        if hasattr(cv2, "QRCodeDetectorAruco"):
            self.detectors.append(cv2.QRCodeDetectorAruco())
        self.lock = threading.Lock()
        self.last = None

    def _size_from_payload(self, text):
        m = FIDUCIAL_PAYLOAD_RE.search(text or "")
        if not m:
            return None
        v, unit = float(m.group(1)), (m.group(2) or "cm").lower()
        return v * {"cm": 1.0, "mm": 0.1, "m": 100.0}[unit]

    def detect(self, frame_bgr, H_i2c=None, now=None):
        """-> dict or None. H_i2c: court homography image px -> court metres (optional)."""
        now = time.time() if now is None else now
        gray = frame_bgr if frame_bgr.ndim == 2 else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        scale = 1.0
        if gray.shape[1] > 1600:
            scale = 1600.0 / gray.shape[1]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        text, pts = "", None
        # classic detector (1x), then the faster perspective-tolerant Aruco-based one at 1x and 2x (small markers)
        attempts = [(self.detectors[0], 1.0)] + [(d, up) for d in self.detectors[1:]
                                                 for up in ((1.0, 2.0) if gray.shape[1] <= 1600 else (1.0,))]
        for det, up in attempts:
            g = gray if up == 1.0 else cv2.resize(gray, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
            try:
                text, found, _ = det.detectAndDecode(g)
                if found is None or np.asarray(found).size < 8:
                    ok, found = det.detect(g)            # corners without payload -> default size
                    text = ""
                    found = found if ok else None
            except cv2.error:
                found = None
            if found is not None and np.asarray(found).size >= 8:
                pts = np.asarray(found, np.float64).reshape(-1, 2)[:4] / up
                break
            text = ""
        if pts is None:
            return None
        quad = np.asarray(pts, np.float64).reshape(-1, 2)[:4] / scale
        if not cv2.isContourConvex(quad.astype(np.float32).reshape(-1, 1, 2)) or cv2.contourArea(quad.astype(np.float32)) < 64:
            return None
        payload_size = self._size_from_payload(text)
        size_cm = payload_size or self.default_size_cm
        sides = np.linalg.norm(quad - np.roll(quad, -1, axis=0), axis=1)
        px_per_cm = float(sides.mean() / size_cm)

        # metric floor-plane homography of the marker: image px -> cm (marker TL at origin)
        sq = np.array([[0, 0], [size_cm, 0], [size_cm, size_cm], [0, size_cm]], np.float64)
        H_marker = cv2.getPerspectiveTransform(quad.astype(np.float32), sq.astype(np.float32))

        # planar tilt from a square PnP with assumed phone intrinsics (approximate: browser exposes no FOV)
        h, w = frame_bgr.shape[:2]
        f = (w / 2.0) / np.tan(np.radians(self.assumed_hfov_deg) / 2.0)
        K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], np.float64)
        obj = np.array([[-size_cm / 2, size_cm / 2, 0], [size_cm / 2, size_cm / 2, 0],
                        [size_cm / 2, -size_cm / 2, 0], [-size_cm / 2, -size_cm / 2, 0]], np.float64)
        tilt_deg, distance_m = None, None
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, quad, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            if ok:
                R, _ = cv2.Rodrigues(rvec)
                normal = R[:, 2]
                tilt_deg = float(np.degrees(np.arccos(np.clip(abs(normal[2]), 0, 1))))
                distance_m = float(np.linalg.norm(tvec) / 100.0)
        except cv2.error:
            pass

        res = {"t": now, "decoded": text or "", "size_cm": round(size_cm, 2),
               "size_source": "payload" if payload_size else "default",
               "corners_px": np.round(quad, 1).tolist(), "px_per_cm": round(px_per_cm, 3),
               "tilt_deg": None if tilt_deg is None else round(tilt_deg, 1),
               "distance_m_approx": None if distance_m is None else round(distance_m, 2),
               "marker_homography": np.round(H_marker, 8).tolist()}

        if H_i2c is not None:
            Hc = np.asarray(H_i2c, np.float64)
            p = cv2.perspectiveTransform(quad.reshape(-1, 1, 2), Hc).reshape(-1, 2)
            if np.all(np.isfinite(p)):
                edges_m = np.linalg.norm(p - np.roll(p, -1, axis=0), axis=1)
                measured_cm = float(edges_m.mean() * 100.0)
                factor = size_cm / measured_cm if measured_cm > 1e-6 else None
                centre = p.mean(axis=0)
                on_court = bool(-0.5 <= centre[0] <= COURT_W_M + 0.5 and -0.5 <= centre[1] <= COURT_L_M + 0.5)
                agree = None if factor is None else max(0.0, 100.0 * (1.0 - abs(1.0 - factor)))
                res.update({"court_measured_cm": round(measured_cm, 1),
                            "court_scale_factor": None if factor is None else round(factor, 4),
                            "scale_agreement_pct": None if agree is None else round(agree, 1),
                            "court_position_m": np.round(centre, 2).tolist(), "on_court": on_court,
                            "court_edges_cm": np.round(edges_m * 100, 1).tolist(),
                            "verdict": ("court scale verified" if agree is not None and agree >= 93
                                        else "court corners look off - re-check calibration")})
        with self.lock:
            self.last = res
        return res

    def recent(self, max_age=1.5):
        with self.lock:
            r = self.last
        return r if r is not None and time.time() - r["t"] <= max_age else None

    def public_state(self):
        r = self.recent(5.0)
        if r is None:
            return None
        return {k: v for k, v in r.items() if k != "marker_homography"}


def draw_fiducial_overlay(frame, fid):
    """Outline + scale readout of a detected floor QR marker (call on the HUD frame)."""
    if not fid:
        return
    q = np.asarray(fid["corners_px"], np.float32)
    if q.shape != (4, 2):
        return
    s = frame.shape[0] / 720.0
    cv2.polylines(frame, [q.astype(np.int32).reshape(-1, 1, 2)], True, (255, 0, 255), max(2, int(3 * s)), cv2.LINE_AA)
    for p in q:
        cv2.circle(frame, (int(p[0]), int(p[1])), max(3, int(5 * s)), (255, 255, 255), -1, cv2.LINE_AA)
    lines = [f"QR SCALE {fid['size_cm']:.0f} cm | {fid['px_per_cm']:.2f} px/cm"
             + (f" | tilt ~{fid['tilt_deg']:.0f} deg" if fid.get("tilt_deg") is not None else "")]
    if fid.get("scale_agreement_pct") is not None:
        lines.append(f"court H measures {fid['court_measured_cm']:.0f} cm -> {fid['scale_agreement_pct']:.0f}% agreement")
    x = int(q[:, 0].min())
    y = int(q[:, 1].min()) - int(10 * s)
    for i, ln in enumerate(reversed(lines)):
        yy = max(int(20 * s), y - i * int(24 * s))
        cv2.putText(frame, ln, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (0, 0, 0), max(3, int(4 * s)), cv2.LINE_AA)
        cv2.putText(frame, ln, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.55 * s, (255, 180, 255), max(1, int(1.5 * s)),
                    cv2.LINE_AA)


# =====================================================================
# 6. FastAPI router: phone web app, frame WebSocket, QR image
# =====================================================================
MOBILE_PAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "enterprise_ui", "mobile_camera.html")


def _is_loopback(request_or_ws):
    host = getattr(getattr(request_or_ws, "client", None), "host", "") or ""
    return host in ("127.0.0.1", "::1", "localhost") or host.startswith("127.")


def _decode_jpeg(data):
    """Binary JPEG -> BGR (OpenCV's libjpeg-turbo, GIL released during decode). -> (frame, ms)."""
    t0 = time.perf_counter()
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return frame, (time.perf_counter() - t0) * 1000.0


def build_mobile_router(hub, admin_local_only=True):
    """Routes shared by the HTTP dashboard app and the LAN HTTPS bridge app.
    Admin routes (QR image with the pairing token, full status) are loopback-only when admin_local_only."""
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Request
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse

    router = APIRouter()

    def admin_ok(req):
        return (not admin_local_only) or _is_loopback(req)

    @router.get("/mobile-camera", response_class=HTMLResponse)
    def mobile_camera_page():
        if not os.path.isfile(MOBILE_PAGE):
            return HTMLResponse("<h1>enterprise_ui/mobile_camera.html missing</h1>", status_code=500)
        return FileResponse(MOBILE_PAGE, media_type="text/html",
                            headers={"Cache-Control": "no-cache", "Permissions-Policy": "camera=(self)"})

    @router.get("/api/qr.png")
    def qr_png(request: Request, size: int = 512):
        if not admin_ok(request):
            return JSONResponse({"ok": False, "error": "QR is only served to this computer"}, status_code=403)
        return Response(hub.qr_png(int(np.clip(size, 128, 1400))), media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @router.get("/api/fiducial.png")
    def fiducial_png(size_cm: float = DEFAULT_FIDUCIAL_SIZE_CM, px: int = 900):
        """Printable / on-screen scale marker. Print it so the black code area (inside the white margin)
        measures exactly size_cm; the size is encoded in the payload, so any camera picks it up."""
        text = f"PADELIQ-SCALE:{float(size_cm):g}cm"
        return Response(qr_png_bytes(text, size=int(np.clip(px, 200, 2400)), border=4), media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @router.get("/api/mobile/pair")
    def mobile_pair(request: Request):
        """Pairing token for a phone that opened /mobile-camera without ?k= (typed URL / bookmark).
        Only served to loopback or private-LAN (RFC 1918) clients - never to public addresses."""
        host = getattr(getattr(request, "client", None), "host", "") or ""
        if not (_is_loopback(request) or _is_private_v4(host)):
            return JSONResponse({"ok": False, "error": "pairing is only available on the local network"},
                                status_code=403)
        return {"ok": True, "token": hub.token}

    @router.get("/api/mobile/status")
    def mobile_status(request: Request):
        return hub.status(include_token=admin_ok(request))

    @router.get("/api/mobile/hud.mjpg")
    def mobile_hud(k: str = ""):
        if not hub.check_token(k):
            return JSONResponse({"ok": False, "error": "invalid pairing token"}, status_code=403)

        def gen():
            last = None
            while True:
                jpg = hub.controller.hud_jpeg()
                if jpg is None:
                    break
                if jpg is not last:
                    last = jpg
                    yield (b"--padeliqframe\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(0.01)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=padeliqframe",
                                 headers={"Cache-Control": "no-cache, no-store"})

    @router.websocket("/ws/mobile-cam")
    async def ws_mobile_cam(ws: WebSocket):
        token = ws.query_params.get("k", "")
        await ws.accept()
        if not hub.check_token(token):
            await ws.send_text(json.dumps({"type": "error", "message": "Invalid pairing token - rescan the QR code."}))
            await ws.close(code=4403)
            return
        hub.client_connected()
        send_lock = asyncio.Lock()
        alive = True
        frames = 0
        first_frame = True

        async def send(obj):
            async with send_lock:
                await ws.send_text(json.dumps(obj))

        def calib_msg():
            snap = hub.controller.calibration_snapshot() or {}
            fs = snap.get("frame_shape")
            corners = snap.get("corners")
            norm = None
            if corners is not None and fs:
                c = np.asarray(corners, np.float64).reshape(4, 2)
                norm = np.round(c / np.array([fs[1], fs[0]], np.float64), 5).tolist()
            out = {"type": "calib", "state": snap.get("state"), "corners": norm,
                   "version": snap.get("version"), "confidence_pct": snap.get("confidence_pct"),
                   "method": snap.get("method"), "angle": snap.get("angle"), "banner": snap.get("banner"),
                   "auto_confirm_in_s": snap.get("auto_confirm_in_s"),
                   "players": snap.get("players"), "inference_fps": snap.get("inference_fps"),
                   "stream_fps": round(hub.cam.native_fps, 1) if hub.cam is not None else 0.0,
                   "source_active": hub.cam is not None, "fiducial": hub.fiducial.public_state(),
                   "hud": hub.controller.hud_jpeg() is not None, "locked": bool(snap.get("locked"))}
            return out

        async def pusher():
            while alive:
                try:
                    await send(await asyncio.to_thread(calib_msg))
                except Exception:
                    return
                await asyncio.sleep(0.2)

        async def run_action(action, fn):
            try:
                msg = await asyncio.to_thread(fn)
                await send({"type": "result", "action": action, "ok": True, "message": msg or "ok"})
            except Exception as exc:
                try:
                    await send({"type": "result", "action": action, "ok": False, "message": str(exc)})
                except Exception:
                    pass

        def to_px(pts):
            fs = (hub.controller.calibration_snapshot() or {}).get("frame_shape") or hub.last_frame_shape
            if not fs:
                raise ValueError("no video frame yet")
            a = np.asarray(pts, np.float64).reshape(4, 2)
            if not np.all(np.isfinite(a)):
                raise ValueError("invalid corners")
            return (a * np.array([fs[1], fs[0]], np.float64)).astype(np.float32)

        await send({"type": "welcome", "server": "PadelIQ", "lan_ip": hub.lan_ip, "ack_every": 1,
                    "max_in_flight": 2})
        push_task = asyncio.create_task(pusher())
        tasks = set()
        try:
            while True:
                msg = await ws.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                data = msg.get("bytes")
                if data:
                    arrived = time.time()
                    hub.bytes_received += len(data)
                    frame, dec_ms = await asyncio.to_thread(_decode_jpeg, data)
                    frames += 1
                    hub.decode_ms = dec_ms if frames == 1 else 0.9 * hub.decode_ms + 0.1 * dec_ms
                    hub.frame_kb = len(data) / 1024 if frames == 1 else 0.9 * hub.frame_kb + 0.1 * len(data) / 1024
                    if frame is not None and frame.size:
                        hub.push_frame(frame, arrived)
                        if first_frame:
                            first_frame = False
                            _log(f"Phone streaming {frame.shape[1]}x{frame.shape[0]} "
                                 f"({hub.meta.get('lens') or 'camera'}, {hub.meta.get('orientation') or '?'})")
                            await asyncio.to_thread(hub.controller.on_phone_connected, dict(hub.meta))
                    await send({"type": "ack", "n": frames})
                    continue
                text = msg.get("text")
                if not text:
                    continue
                try:
                    m = json.loads(text)
                except ValueError:
                    continue
                typ = m.get("type")
                if typ == "meta":
                    hub.meta = {k: m.get(k) for k in ("w", "h", "orientation", "lens", "zoom", "lenses", "ua",
                                                      "facing", "fps", "track_w", "track_h") if k in m}
                elif typ == "ping":
                    await send({"type": "pong", "t": m.get("t")})
                elif typ == "auto_detect":
                    t = asyncio.create_task(run_action("auto_detect", hub.controller.auto_detect))
                    tasks.add(t)
                    t.add_done_callback(tasks.discard)
                elif typ == "corners":
                    try:
                        px = to_px(m.get("pts"))
                        await asyncio.to_thread(hub.controller.preview_corners, px)
                    except Exception as exc:
                        await send({"type": "result", "action": "corners", "ok": False, "message": str(exc)})
                elif typ == "lock":
                    pts = m.get("pts")
                    px = to_px(pts) if pts else None
                    t = asyncio.create_task(run_action("lock", lambda: hub.controller.lock(px)))
                    tasks.add(t)
                    t.add_done_callback(tasks.discard)
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            pass
        finally:
            alive = False
            push_task.cancel()
            hub.client_disconnected()
            _log(f"Phone disconnected after {frames} frames.")

    return router


def build_https_app(hub):
    """Minimal LAN-facing app for the HTTPS port: only the phone routes (the full dashboard API stays on HTTP)."""
    from fastapi import FastAPI
    from fastapi.responses import RedirectResponse
    app = FastAPI(title="PadelIQ Mobile Bridge", docs_url=None, redoc_url=None, openapi_url=None)
    app.include_router(build_mobile_router(hub, admin_local_only=True))

    @app.get("/")
    def root():
        return RedirectResponse("/mobile-camera")

    return app


def port_free(port, host="0.0.0.0"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def start_https_server(app, hub, cert_dir, host="0.0.0.0"):
    """Serve `app` over HTTPS on hub.https_port in a daemon thread. Returns the uvicorn.Server or None."""
    import uvicorn
    try:
        cert, key = ensure_self_signed_cert(cert_dir, hub.lan_ip)
    except Exception as exc:
        hub.https_error = f"certificate: {exc}"
        _log(f"HTTPS disabled - {hub.https_error}")
        return None
    if not port_free(hub.https_port, host):
        hub.https_error = f"port {hub.https_port} is in use"
        _log(f"HTTPS disabled - {hub.https_error}")
        return None
    config = uvicorn.Config(app, host=host, port=hub.https_port, ssl_certfile=cert, ssl_keyfile=key,
                            log_level="warning", access_log=False, ws_max_size=8 * 1024 * 1024)
    server = uvicorn.Server(config)

    def run():
        try:
            server.run()
        except Exception as exc:                      # pragma: no cover
            hub.https_enabled = False
            hub.https_error = repr(exc)
            _log(f"HTTPS bridge stopped: {exc!r}")

    threading.Thread(target=run, name="PadelIQ-https", daemon=True).start()
    t_end = time.time() + 8.0
    while time.time() < t_end and not server.started:
        time.sleep(0.05)
    hub.https_enabled = bool(server.started)
    if hub.https_enabled:
        _log(f"Mobile camera bridge -> {hub.url.split('?')[0]}  (scan the QR code on screen)")
    else:
        hub.https_error = hub.https_error or "HTTPS server failed to start"
    return server
