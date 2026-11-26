# -*- coding: utf-8 -*-
"""
PadelIQ Vision v2 - perception & analytics core for PadelIQ Enterprise v2.0
==========================================================================

  PoseEstimator      top-down YOLOv8n-pose on player crops (OpenVINO > PyTorch fallback)
  ground_contact     ankle keypoints -> ground point; hip/shoulder anthropometric projection when
                     the feet are hidden (ratio learned online per player); bbox bottom as last resort
  Appearance         HSV shirt/shorts histograms with court-colour masking (Re-ID signature)
  TrackerV2          Hungarian slot lock: distance + team side + appearance + physical speed gate,
                     per-measurement noise from the local homography scale, Re-ID after occlusion
  BallTracker        colour + motion blob detection with a pixel-space Kalman filter
  RallySegmenter     rally / rest state machine from player motion energy + ball activity
  FallMonitor        "player down" (torso angle / box aspect + immobility)
  MatchAnalytics     team gap, formation, net dominance, fatigue index, tactical insights
  Calibration v2     court-surface segmentation, painted-line RANSAC refinement, confidence score,
                     camera height / tilt / FOV from the homography, drift monitor
  AccuracyModel      spatial confidence: painted-line residuals (ground truth = court geometry)
                     + measured tracking repeatability, per court half
"""
import os
import math
import time
import threading
from collections import deque

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import padeliq_live_stream as E

COURT_W, COURT_L, NET_Y = E.COURT_WIDTH_M, E.COURT_LENGTH_M, E.NET_Y_M
SERVICE_FAR, SERVICE_NEAR = E.SERVICE_LINE_NEAR_M, E.SERVICE_LINE_FAR_M      # 3.05 (far half), 16.95 (near half)

# COCO-17 keypoints
KP = dict(nose=0, l_eye=1, r_eye=2, l_ear=3, r_ear=4, l_sh=5, r_sh=6, l_el=7, r_el=8, l_wr=9, r_wr=10,
          l_hip=11, r_hip=12, l_kn=13, r_kn=14, l_an=15, r_an=16)
SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

MODES = {
    "padel":    {"slots": (1, 2, 3, 4), "team_lock": True,  "coach": True,  "label": "Padel 10x20 m - 4 players"},
    "training": {"slots": (1, 2),       "team_lock": False, "coach": True,  "label": "Mini-court / training - 2 players"},
    "safety":   {"slots": (1, 2, 3, 4), "team_lock": False, "coach": False, "label": "Facility safety monitoring"},
}
TEAMS = {"north": (1, 2), "south": (3, 4)}          # north = far half (y < 10), south = near half
ODO_K = 3.5              # odometer step threshold = ODO_K x Kalman position sigma + ODO_MIN_M
ODO_MIN_M = 0.10
ODO_WINDOW_S = 1.0       # speed = odometer distance over this window


def _now():
    return time.time()


# =====================================================================
# 1. Pose estimation (top-down on detector crops)
# =====================================================================
class PoseEstimator:
    """Runs YOLOv8n-pose on square crops around each detected person (batched).
    Crop-based pose sees far players at ~4x the resolution of a full-frame pose pass."""

    def __init__(self, app_dir=E.APP_DIR, crop_size=192, prefer="openvino"):
        self.model, self.backend, self.size = None, "none", int(crop_size)
        self.errors = 0
        candidates = []
        ov_dir = os.path.join(app_dir, "yolov8n-pose_openvino_model")
        pt = os.path.join(app_dir, "yolov8n-pose.pt")
        if prefer == "openvino" and os.path.isdir(ov_dir):
            candidates.append(("openvino", ov_dir))
        if os.path.isfile(pt):
            candidates.append(("pytorch", pt))
        for name, path in candidates:
            try:
                from ultralytics import YOLO
                m = YOLO(path, task="pose")
                m.predict([np.zeros((self.size, self.size, 3), np.uint8)] * 2, imgsz=self.size, verbose=False)
                self.model, self.backend = m, name
                E.log(f"Pose model ready ({name}, crops {self.size}px)")
                break
            except Exception as exc:
                E.log(f"Pose backend {name} unavailable: {exc!r}")

    @property
    def ready(self):
        return self.model is not None

    def _crop(self, frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box
        s = max(x2 - x1, y2 - y1) * 1.25
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        X1, Y1 = int(round(cx - s / 2)), int(round(cy - s / 2))
        S = max(8, int(round(s)))
        canvas = np.zeros((S, S, 3), np.uint8)
        sx1, sy1, sx2, sy2 = max(0, X1), max(0, Y1), min(w, X1 + S), min(h, Y1 + S)
        if sx2 > sx1 and sy2 > sy1:
            canvas[sy1 - Y1:sy2 - Y1, sx1 - X1:sx2 - X1] = frame[sy1:sy2, sx1:sx2]
        return cv2.resize(canvas, (self.size, self.size), interpolation=cv2.INTER_LINEAR), (X1, Y1, S / self.size)

    def __call__(self, frame, boxes):
        """Returns one (17, 3) array [x, y, conf] in frame pixels (or None) per box."""
        out = [None] * len(boxes)
        if self.model is None or len(boxes) == 0:
            return out
        try:
            crops, xf = zip(*[self._crop(frame, b) for b in boxes])
            res = self.model.predict(list(crops), imgsz=self.size, conf=0.25, verbose=False)
            for i, (r, (X1, Y1, k)) in enumerate(zip(res, xf)):
                if r.keypoints is None or r.boxes is None or len(r.boxes) == 0:
                    continue
                # the person we cropped for sits in the middle of the crop with a known box
                b = np.asarray(boxes[i], dtype=np.float32)
                exp = np.array([(b[0] - X1) / k, (b[1] - Y1) / k, (b[2] - X1) / k, (b[3] - Y1) / k])
                pb = r.boxes.xyxy.cpu().numpy()
                ious = [_iou(exp, q) for q in pb]
                j = int(np.argmax(ious))
                if ious[j] < 0.3:
                    continue
                kxy = r.keypoints.xy.cpu().numpy()[j]
                kc = r.keypoints.conf.cpu().numpy()[j] if r.keypoints.conf is not None else np.ones(17)
                pts = np.zeros((17, 3), np.float32)
                pts[:, 0] = kxy[:, 0] * k + X1
                pts[:, 1] = kxy[:, 1] * k + Y1
                pts[:, 2] = kc
                out[i] = pts
        except Exception as exc:
            self.errors += 1
            if self.errors <= 3:
                E.log(f"Pose error (frame skipped): {exc!r}")
        return out


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


# =====================================================================
# 2. Ground contact point from pose
# =====================================================================
KCONF = 0.45
DEFAULT_LEG_TORSO_RATIO = 1.70        # (sole - hip) / (hip - shoulder) in image y for an athletic stance


def ground_contact(box, kps, leg_ratio=DEFAULT_LEG_TORSO_RATIO):
    """-> (foot_px (x, y), method, sigma_px, measured_ratio_or_None)"""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    bh = max(1.0, y2 - y1)
    if kps is not None:
        an = [kps[i] for i in (15, 16) if kps[i][2] > KCONF]
        hips = [kps[i] for i in (11, 12) if kps[i][2] > KCONF]
        shs = [kps[i] for i in (5, 6) if kps[i][2] > KCONF]
        ratio = None
        if an:
            fy = max(a[1] for a in an) + 0.025 * bh           # ankle joint -> sole
            fx = float(np.mean([a[0] for a in an]))
            if hips and shs:
                hy, sy = np.mean([p[1] for p in hips]), np.mean([p[1] for p in shs])
                if hy - sy > 4:
                    ratio = (fy - hy) / (hy - sy)
            return (fx, fy), "ankle", 3.0 + 0.015 * bh, ratio
        if hips and shs:
            hy, sy = float(np.mean([p[1] for p in hips])), float(np.mean([p[1] for p in shs]))
            hx = float(np.mean([p[0] for p in hips]))
            if hy - sy > 4:
                return (hx, hy + leg_ratio * (hy - sy)), "hip_projection", 5.0 + 0.05 * bh, None
    return ((x1 + x2) / 2, y2), "bbox", 10.0, None


# =====================================================================
# 3. Appearance signature (Re-ID)
# =====================================================================
class Appearance:
    BINS = (18, 4)

    def __init__(self):
        self.court_hue = None          # set by calibration (OpenCV hue 0..180)

    def descriptor(self, frame, box, kps=None):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = [int(round(v)) for v in box[:4]]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 - x1 < 6 or y2 - y1 < 12:
            return None
        bh, bw = y2 - y1, x2 - x1
        up = (x1 + int(0.2 * bw), y1 + int(0.18 * bh), x2 - int(0.2 * bw), y1 + int(0.48 * bh))
        lo = (x1 + int(0.2 * bw), y1 + int(0.48 * bh), x2 - int(0.2 * bw), y1 + int(0.68 * bh))
        if kps is not None and all(kps[i][2] > KCONF for i in (5, 6, 11, 12)):
            sx = sorted([kps[5][0], kps[6][0]])
            sy, hy = min(kps[5][1], kps[6][1]), max(kps[11][1], kps[12][1])
            if hy - sy > 6:
                up = (int(sx[0]), int(sy), int(max(sx[1], sx[0] + 4)), int(hy))
                kn = [kps[i][1] for i in (13, 14) if kps[i][2] > KCONF]
                ly = int(min(kn)) if kn else int(hy + 0.6 * (hy - sy))
                lo = (int(sx[0]), int(hy), int(max(sx[1], sx[0] + 4)), max(int(hy) + 4, ly))
        parts = []
        for (a, b, c, d) in (up, lo):
            a, b, c, d = max(0, a), max(0, b), min(w, c), min(h, d)
            if c - a < 3 or d - b < 3:
                return None
            hsv = cv2.cvtColor(frame[b:d, a:c], cv2.COLOR_BGR2HSV)
            mask = None
            if self.court_hue is not None:
                dh = np.abs(hsv[..., 0].astype(np.int16) - int(self.court_hue))
                dh = np.minimum(dh, 180 - dh)
                mask = (~((dh < 10) & (hsv[..., 1] > 70))).astype(np.uint8) * 255
                if cv2.countNonZero(mask) < 6:
                    mask = None
            hist = cv2.calcHist([hsv], [0, 1], mask, list(self.BINS), [0, 180, 0, 256]).flatten()
            s = hist.sum()
            if s <= 0:
                return None
            parts.append((hist / s).astype(np.float32))
        return parts

    @staticmethod
    def distance(a, b):
        if a is None or b is None:
            return 0.5
        du = cv2.compareHist(a[0], b[0], cv2.HISTCMP_BHATTACHARYYA)
        dl = cv2.compareHist(a[1], b[1], cv2.HISTCMP_BHATTACHARYYA)
        return float(0.6 * du + 0.4 * dl)


# =====================================================================
# 4. Player state & tracker
# =====================================================================
class PlayerV2(E.LivePlayerState):
    def __init__(self, slot_id, default_pos_m):
        super().__init__(slot_id, default_pos_m)
        self.kps = None
        self.kps_time = 0.0
        self.ground_method = "none"
        self.method_counts = {"ankle": 0, "hip_projection": 0, "bbox": 0}
        self.leg_ratios = deque(maxlen=80)
        self.proto = None
        self.proto_n = 0
        self.raw_hist = deque(maxlen=120)           # (t, x, y) raw ground measurements (m)
        self.torso_angles = deque(maxlen=30)
        self.down_since = None
        self.alert_active = False
        self.speed_log = deque(maxlen=4000)         # (t, distance) for fatigue
        self.reid_recoveries = 0
        self.odo_anchor = None
        self.odo_anchor_t = 0.0
        self.odo_win = deque()
        self._v2_recent = deque(maxlen=12)

    # ---- noise-adaptive odometer (replaces v1 per-step distance; rejects far-court depth jitter)
    def update(self, bbox, foot_px, pos_m, now, meas_sigma=None):
        d0, a0, m0 = self.distance_m, self.active_seconds_total, self.max_speed_ms
        last_upd, last_seen = self.last_update_time, self.last_seen_time
        super().update(bbox, foot_px, pos_m, now, meas_sigma)
        dt = max(0.016, min(0.5, now - last_upd))
        self.distance_m, self.active_seconds_total, self.max_speed_ms = d0, a0, m0
        pos = self.kf_x[:2].copy()
        if self.odo_anchor is None or now - last_seen > 1.2:
            self.odo_anchor = pos
            self.odo_win.clear()
            inc = 0.0
        else:
            sig = float(np.sqrt(max(np.linalg.eigvalsh(self.kf_P[:2, :2]))))
            thr = ODO_K * sig + ODO_MIN_M
            disp = float(np.linalg.norm(pos - self.odo_anchor))
            inc = 0.0
            if disp > thr:
                inc = min(disp, E.MAX_PLAYER_SPEED_MS * max(dt, now - self.odo_anchor_t))
                self.odo_anchor = pos
                self.odo_anchor_t = now
        if inc == 0.0 and self.odo_anchor_t == 0.0:
            self.odo_anchor_t = now
        self.distance_m += inc
        self.odo_win.append((now, inc))
        while self.odo_win and now - self.odo_win[0][0] > ODO_WINDOW_S:
            self.odo_win.popleft()
        span = max(0.25, min(ODO_WINDOW_S, now - self.odo_win[0][0])) if self.odo_win else ODO_WINDOW_S
        spd = min(E.MAX_PLAYER_SPEED_MS, sum(v for _, v in self.odo_win) / span)
        self.instant_speed_ms = 0.5 * self.instant_speed_ms + 0.5 * spd
        if self.instant_speed_ms > E.MOVE_SPEED_MS:
            self.active_seconds_total += dt
        self._v2_recent.append(self.instant_speed_ms)
        if len(self._v2_recent) >= 6:
            self.max_speed_ms = max(self.max_speed_ms, float(np.median(self._v2_recent)))
        if self.tracked_seconds_total > 0.5:
            self.avg_speed_ms = self.distance_m / self.tracked_seconds_total

    @property
    def leg_ratio(self):
        if len(self.leg_ratios) >= 5:
            return float(np.clip(np.median(self.leg_ratios), 1.2, 2.3))
        return DEFAULT_LEG_TORSO_RATIO

    @property
    def team(self):
        return "north" if self.slot_id in TEAMS["north"] else "south"

    @property
    def torso_angle(self):
        return float(np.median(self.torso_angles)) if self.torso_angles else 0.0


class TrackerV2(E.LiveSlotManager):
    APP_WEIGHT = 7.0

    def __init__(self, H, mode="padel"):
        self.mode = mode if mode in MODES else "padel"
        self.appearance = Appearance()
        self.lock_events = 0
        super().__init__(H)

    def reset_players(self):
        anchors = {1: (2.5, 4.5), 2: (7.5, 4.5), 3: (2.5, 15.5), 4: (7.5, 15.5)}
        if self.mode == "training":
            anchors = {1: (5.0, 15.0), 2: (5.0, 5.0)}
        self.players = {sid: PlayerV2(sid, anchors[sid]) for sid in MODES[self.mode]["slots"]}

    def set_mode(self, mode):
        if mode in MODES and mode != self.mode:
            self.mode = mode
            self.reset_players()

    def _local_sigma(self, foot_px, pos_m, sigma_px):
        dx = self.pixel_to_meters((foot_px[0] + sigma_px, foot_px[1])) - pos_m
        dy = self.pixel_to_meters((foot_px[0], foot_px[1] + sigma_px)) - pos_m
        return float(np.clip(np.hypot(*dx), 0.03, 1.5)), float(np.clip(np.hypot(*dy), 0.03, 1.5))

    def update_v2(self, frame, boxes, poses, now):
        """boxes: (N,4) detector boxes; poses: list of (17,3) or None (same order)."""
        if self.H is None or boxes is None or len(boxes) == 0:
            return []
        cands = []
        for i, box in enumerate(boxes):
            kps = poses[i] if poses is not None and i < len(poses) else None
            foot_px, method, sig_px, ratio = ground_contact(box, kps)
            pos_m = self.pixel_to_meters(foot_px)
            if not np.all(np.isfinite(pos_m)):
                continue
            if not (-1.0 <= pos_m[0] <= COURT_W + 1.0 and -1.5 <= pos_m[1] <= COURT_L + 1.5):
                continue
            cands.append(dict(box=tuple(float(v) for v in box[:4]), kps=kps, foot_px=foot_px, method=method,
                              sig_px=sig_px, ratio=ratio, pos_m=pos_m))
        if not cands:
            return []
        if len(cands) > 8:
            c0 = np.array([COURT_W / 2, NET_Y])
            cands.sort(key=lambda c: np.linalg.norm(c["pos_m"] - c0))
            cands = cands[:8]
        for c in cands:
            c["desc"] = self.appearance.descriptor(frame, c["box"], c["kps"])

        slot_ids = list(self.players.keys())
        team_lock = MODES[self.mode]["team_lock"]
        cost = np.zeros((len(slot_ids), len(cands)), dtype=np.float64)
        for r, sid in enumerate(slot_ids):
            pl = self.players[sid]
            since = now - pl.last_seen_time
            pred = pl.last_pos_m.astype(np.float64)
            if pl.kf_x is not None and since < 0.6:
                pred = pl.kf_x[:2] + pl.kf_x[2:] * min(since, 0.4)
            reacq = pl.smoothed_pos_m is None or since > 1.2
            for c_i, c in enumerate(cands):
                pos = c["pos_m"]
                dist = float(np.linalg.norm(pos - pred))
                pen = 0.0
                if team_lock:
                    if sid in TEAMS["north"] and pos[1] > NET_Y + 1.5:
                        pen += 18.0
                    elif sid in TEAMS["south"] and pos[1] < NET_Y - 1.5:
                        pen += 18.0
                if not reacq and dist > E.MAX_PLAYER_SPEED_MS * max(since, 0.05) + 1.0:
                    pen += 1e4
                app = 0.0
                if pl.proto_n >= 8 and c["desc"] is not None:
                    app = Appearance.distance(pl.proto, c["desc"]) * self.APP_WEIGHT * (2.0 if reacq else 1.0)
                cost[r, c_i] = (0.35 if reacq else 1.0) * dist + pen + app
        rows, cols = linear_sum_assignment(cost)
        assigned = []
        for r, c_i in zip(rows, cols):
            if cost[r, c_i] >= 20.0:
                continue
            sid, c = slot_ids[r], cands[c_i]
            pl = self.players[sid]
            if pl.smoothed_pos_m is not None and now - pl.last_seen_time > 1.2:
                pl.reid_recoveries += 1
            if c["method"] == "hip_projection":           # use the player's own learned leg/torso ratio
                c["foot_px"], _, _, _ = ground_contact(c["box"], c["kps"], pl.leg_ratio)
                c["pos_m"] = self.pixel_to_meters(c["foot_px"])
            sig = self._local_sigma(c["foot_px"], c["pos_m"], c["sig_px"])
            pl.update(c["box"], c["foot_px"], c["pos_m"], now, meas_sigma=sig)
            pl.raw_hist.append((now, float(c["pos_m"][0]), float(c["pos_m"][1])))
            pl.speed_log.append((now, pl.distance_m))
            pl.ground_method = c["method"]
            pl.method_counts[c["method"]] += 1
            if c["ratio"] is not None and 1.0 < c["ratio"] < 2.6:
                pl.leg_ratios.append(c["ratio"])
            if c["kps"] is not None:
                pl.kps, pl.kps_time = c["kps"], now
                ang = torso_angle_deg(c["kps"])
                if ang is not None:
                    pl.torso_angles.append(ang)
            # appearance prototype: learn only from unoccluded, confident matches
            if c["desc"] is not None:
                iso = all(_iou(c["box"], o["box"]) < 0.12 for o in cands if o is not c)
                if iso and (pl.proto_n < 8 or Appearance.distance(pl.proto, c["desc"]) < 0.55):
                    if pl.proto is None:
                        pl.proto = [d.copy() for d in c["desc"]]
                    else:
                        a = 1.0 / (pl.proto_n + 1) if pl.proto_n < 20 else 0.05
                        pl.proto = [(1 - a) * p + a * d for p, d in zip(pl.proto, c["desc"])]
                    pl.proto_n += 1
            assigned.append(sid)
        return cands


def torso_angle_deg(kps):
    sh = [kps[i] for i in (5, 6) if kps[i][2] > KCONF]
    hp = [kps[i] for i in (11, 12) if kps[i][2] > KCONF]
    if not sh or not hp:
        return None
    sx, sy = np.mean([p[0] for p in sh]), np.mean([p[1] for p in sh])
    hx, hy = np.mean([p[0] for p in hp]), np.mean([p[1] for p in hp])
    return float(math.degrees(math.atan2(abs(sx - hx), max(1e-3, hy - sy)))) if hy - sy > -1e3 else None


# =====================================================================
# 5. Ball tracking
# =====================================================================
class BallTracker:
    """Fluorescent-ball detection (colour AND motion) + constant-velocity Kalman in image pixels."""
    WORK_W = 640

    def __init__(self):
        self.prev_gray = None
        self.kf = None
        self.last_ts = None
        self.hits = 0
        self.last_seen = 0.0
        self.trail = deque(maxlen=14)
        self.events = deque(maxlen=400)       # (ts, x, y) direction changes (hits / bounces)
        self._vy_hist = deque(maxlen=4)
        self.visible = False
        self.pos = None
        self.speed_pxs = 0.0
        self.scale = 1.0
        self.frames = 0
        self.frames_visible = 0

    def _kf_init(self, x, y):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.processNoiseCov = np.diag([4, 4, 400, 400]).astype(np.float32)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 4
        self.kf.errorCovPost = np.diag([10, 10, 2500, 2500]).astype(np.float32)
        self.kf.statePost = np.array([[x], [y], [0], [0]], np.float32)
        self.kf.statePre = self.kf.statePost.copy()
        self.kf.errorCovPre = self.kf.errorCovPost.copy()

    def process(self, frame, ts, player_boxes=()):
        self.frames += 1
        h, w = frame.shape[:2]
        self.scale = self.WORK_W / float(w)
        small = cv2.resize(frame, (self.WORK_W, int(round(h * self.scale))), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        cand = []
        if self.prev_gray is not None and self.prev_gray.shape == gray.shape:
            motion = cv2.dilate((cv2.absdiff(gray, self.prev_gray) > 16).astype(np.uint8), None, iterations=1)
            color = cv2.inRange(hsv, (22, 70, 120), (48, 255, 255))
            mask = cv2.bitwise_and(color, color, mask=motion)
            for b in player_boxes:                 # shoes / clothing near the feet are not the ball
                x1, y1, x2, y2 = [int(v * self.scale) for v in b[:4]]
                yb = int(y2 - 0.22 * (y2 - y1))
                mask[max(0, yb):max(0, y2 + 3), max(0, x1 - 2):max(0, x2 + 2)] = 0
            n, _, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                area = stats[i, cv2.CC_STAT_AREA]
                bw, bh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
                if 2 <= area <= 70 and max(bw, bh) <= 3.5 * max(1, min(bw, bh)) + 2:
                    cand.append((float(cents[i][0]), float(cents[i][1]), area))
        self.prev_gray = gray
        dt = 1 / 30.0 if self.last_ts is None else float(np.clip(ts - self.last_ts, 1 / 120.0, 0.5))
        self.last_ts = ts
        meas = None
        if self.kf is not None and ts - self.last_seen < 0.5:
            self.kf.transitionMatrix = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
            pred = self.kf.predict()
            px, py = float(pred[0, 0]), float(pred[1, 0])
            gate = 25 + 700 * dt
            best = None
            for (cx, cy, a) in cand:
                d = math.hypot(cx - px, cy - py)
                if d < gate and (best is None or d < best[0]):
                    best = (d, cx, cy)
            if best:
                meas = (best[1], best[2])
        elif cand:
            cx, cy, _ = max(cand, key=lambda c: c[2])
            self._kf_init(cx, cy)
            self.hits = 0
            meas = (cx, cy)
        if meas is not None and self.kf is not None:
            st = self.kf.correct(np.array([[meas[0]], [meas[1]]], np.float32))
            self.hits += 1
            self.last_seen = ts
            vx, vy = float(st[2, 0]), float(st[3, 0])
            self.speed_pxs = math.hypot(vx, vy) / self.scale
            self.pos = (meas[0] / self.scale, meas[1] / self.scale)
            if self.hits >= 3:
                self.trail.append(self.pos)
                self._vy_hist.append(vy)
                if len(self._vy_hist) == 4:
                    a, b = np.mean(list(self._vy_hist)[:2]), np.mean(list(self._vy_hist)[2:])
                    if a * b < 0 and min(abs(a), abs(b)) > 40:
                        if not self.events or ts - self.events[-1][0] > 0.35:
                            self.events.append((ts, self.pos[0], self.pos[1]))
                        self._vy_hist.clear()
        self.visible = self.kf is not None and self.hits >= 3 and ts - self.last_seen < 0.25
        if self.visible:
            self.frames_visible += 1
        elif ts - self.last_seen > 0.6:
            self.trail.clear()
        return self.visible

    def moving(self, ts):
        return self.visible and self.speed_pxs > 80 and ts - self.last_seen < 0.25

    @property
    def visibility_pct(self):
        return 100.0 * self.frames_visible / max(1, self.frames)


# =====================================================================
# 6. Rally segmentation
# =====================================================================
class RallySegmenter:
    START_ACT, END_ACT, END_HOLD = 1.25, 0.75, 2.2

    def __init__(self):
        self.state = "rest"
        self.activity = 0.0
        self.rally_start = None
        self.low_since = None
        self.high_since = None
        self.rallies = []
        self._events_at_start = 0
        self._peak_speed = 0.0
        self._net_samples = []
        self.on_rally_end = None

    def update(self, players, ball, now):
        speeds = sorted([p.instant_speed_ms for p in players if p.is_active(now)], reverse=True)[:3]
        act = float(np.mean(speeds)) if speeds else 0.0
        self.activity = 0.8 * self.activity + 0.2 * act
        ball_live = ball is not None and ball.moving(now)
        hot = self.activity > self.START_ACT or ball_live
        if self.state == "rest":
            if hot:
                self.high_since = self.high_since or now
                if now - self.high_since > 0.6:
                    self.state, self.rally_start, self.low_since = "rally", self.high_since, None
                    self._events_at_start = len(ball.events) if ball else 0
                    self._peak_speed, self._net_samples = 0.0, []
            else:
                self.high_since = None
        else:
            self._peak_speed = max(self._peak_speed, max(speeds) if speeds else 0.0)
            north = [p for p in players if p.is_active(now) and p.slot_id in TEAMS["north"]]
            south = [p for p in players if p.is_active(now) and p.slot_id in TEAMS["south"]]
            self._net_samples.append((sum(abs(p.smoothed_pos_m[1] - NET_Y) < 4 for p in north),
                                      sum(abs(p.smoothed_pos_m[1] - NET_Y) < 4 for p in south)))
            if self.activity < self.END_ACT and not ball_live:
                self.low_since = self.low_since or now
                if now - self.low_since > self.END_HOLD:
                    self._close(self.low_since, ball)
            else:
                self.low_since = None
        return self.state

    def _close(self, end, ball):
        dur = end - self.rally_start
        if dur >= 2.0:
            ns = np.array(self._net_samples) if self._net_samples else np.zeros((1, 2))
            r = {"index": len(self.rallies) + 1, "start": self.rally_start, "end": end,
                 "duration_s": round(dur, 1),
                 "ball_events": (len(ball.events) - self._events_at_start) if ball else 0,
                 "peak_player_speed_ms": round(self._peak_speed, 2),
                 "north_at_net_pct": round(100 * float(np.mean(ns[:, 0] >= 2)), 1),
                 "south_at_net_pct": round(100 * float(np.mean(ns[:, 1] >= 2)), 1)}
            self.rallies.append(r)
            if self.on_rally_end:
                try:
                    self.on_rally_end(r)
                except Exception:
                    pass
        self.state, self.rally_start, self.low_since, self.high_since = "rest", None, None, None

    def current(self, now):
        if self.state != "rally":
            return None
        return {"duration_s": round(now - self.rally_start, 1)}


# =====================================================================
# 7. Player-down / fall monitor
# =====================================================================
class FallMonitor:
    ANGLE_DOWN = 58.0          # torso this far from vertical
    ASPECT_DOWN = 0.95         # box height / width below this
    HOLD_S = 2.5               # immobile & down for this long -> alert

    def __init__(self):
        self.alerts = []        # dicts
        self.on_alert = None

    def update(self, players, now):
        for p in players:
            if not p.is_active(now, window=2.5) or p.bbox is None:
                continue
            bw = max(1.0, float(p.bbox[2] - p.bbox[0]))
            bh = float(p.bbox[3] - p.bbox[1])
            ang = p.torso_angles[-1] if p.torso_angles and now - p.kps_time < 1.0 else None
            lying = (ang is not None and ang > self.ANGLE_DOWN) or (bh / bw < self.ASPECT_DOWN)
            still = p.instant_speed_ms < 0.6
            if lying and still:
                p.down_since = p.down_since or now
                if not p.alert_active and now - p.down_since >= self.HOLD_S:
                    p.alert_active = True
                    a = {"id": len(self.alerts) + 1, "t": now, "type": "player_down", "player": f"P{p.slot_id}",
                         "pos_m": [round(float(v), 2) for v in p.smoothed_pos_m],
                         "torso_angle_deg": None if ang is None else round(ang, 1),
                         "box_aspect": round(bh / bw, 2), "active": True, "acknowledged_by": None}
                    self.alerts.append(a)
                    if self.on_alert:
                        try:
                            self.on_alert(a)
                        except Exception:
                            pass
            else:
                if p.down_since and not lying:
                    p.down_since = None
                    if p.alert_active:
                        p.alert_active = False
                        for a in self.alerts:
                            if a["player"] == f"P{p.slot_id}" and a["active"]:
                                a["active"], a["cleared_t"] = False, now
        return [a for a in self.alerts if a["active"]]


# =====================================================================
# 8. Match / formation analytics
# =====================================================================
class MatchAnalytics:
    NET_ZONE = 4.0          # within 4 m of the net = net position

    def __init__(self):
        self.reset()

    def reset(self):
        self.t0 = None
        self.last = None
        self.team = {t: {"time": 0.0, "both_net": 0.0, "both_back": 0.0, "split": 0.0, "gap_sum": 0.0,
                         "gap_n": 0, "gap_now": None, "formation": "unknown"} for t in TEAMS}
        self.net_control = {"north": 0.0, "south": 0.0}

    def update(self, players, now):
        if self.t0 is None:
            self.t0 = now
        dt = 0.0 if self.last is None else float(np.clip(now - self.last, 0, 1.0))
        self.last = now
        for t, ids in TEAMS.items():
            ps = [players[i] for i in ids if i in players and players[i].is_active(now)]
            st = self.team[t]
            if len(ps) < 2:
                st["gap_now"], st["formation"] = None, "unknown"
                continue
            a, b = ps[0].smoothed_pos_m, ps[1].smoothed_pos_m
            gap = float(np.linalg.norm(a - b))
            st["gap_now"] = gap
            st["gap_sum"] += gap
            st["gap_n"] += 1
            at_net = [abs(p.smoothed_pos_m[1] - NET_Y) < self.NET_ZONE for p in ps]
            form = "both_net" if all(at_net) else ("both_back" if not any(at_net) else "split")
            st["formation"] = form
            st["time"] += dt
            st[form] += dt
        n_net = {t: (self.team[t]["formation"] == "both_net") for t in TEAMS}
        if n_net["north"] != n_net["south"]:
            self.net_control["north" if n_net["north"] else "south"] += dt

    def fatigue(self, p, now):
        """% drop of movement rate (m/s) between the first and second half of the player's tracked time."""
        log = list(p.speed_log)
        if len(log) < 20 or log[-1][0] - log[0][0] < 20:
            return None
        t0, t1 = log[0][0], log[-1][0]
        mid = (t0 + t1) / 2
        d_mid = next(d for t, d in log if t >= mid)
        v1 = (d_mid - log[0][1]) / max(1e-3, mid - t0)
        v2 = (log[-1][1] - d_mid) / max(1e-3, t1 - mid)
        if v1 < 0.2:
            return None
        return round(float(np.clip((1 - v2 / v1) * 100, -100, 100)), 1)

    def summary(self, players, now):
        out = {}
        tot_ctrl = sum(self.net_control.values())
        for t in TEAMS:
            st = self.team[t]
            tt = max(1e-6, st["time"])
            out[t] = {"formation": st["formation"],
                      "gap_m": None if st["gap_now"] is None else round(st["gap_now"], 2),
                      "avg_gap_m": round(st["gap_sum"] / st["gap_n"], 2) if st["gap_n"] else None,
                      "both_net_pct": round(100 * st["both_net"] / tt, 1) if st["time"] > 0 else 0.0,
                      "both_back_pct": round(100 * st["both_back"] / tt, 1) if st["time"] > 0 else 0.0,
                      "split_pct": round(100 * st["split"] / tt, 1) if st["time"] > 0 else 0.0,
                      "net_control_pct": round(100 * self.net_control[t] / tot_ctrl, 1) if tot_ctrl > 0 else None}
        out["fatigue_pct"] = {f"P{sid}": self.fatigue(p, now) for sid, p in players.items()}
        return out

    def insights(self, players, now):
        """Tactical insight keys + params (translated by padeliq_i18n)."""
        s = self.summary(players, now)
        res = []
        for t in TEAMS:
            st = s[t]
            if st["formation"] == "split":
                res.append(("team_split", {"team": t}))
            if st["gap_m"] is not None and st["gap_m"] > 6.0:
                res.append(("team_gap_wide", {"team": t, "gap": st["gap_m"]}))
            elif st["gap_m"] is not None and st["gap_m"] < 1.6:
                res.append(("team_gap_tight", {"team": t, "gap": st["gap_m"]}))
            if st["net_control_pct"] is not None and st["net_control_pct"] >= 65:
                res.append(("team_net_control", {"team": t, "pct": st["net_control_pct"]}))
        for pid, f in s["fatigue_pct"].items():
            if f is not None and f >= 20:
                res.append(("fatigue", {"player": pid, "pct": f}))
        return res


# =====================================================================
# 9. Calibration v2: surface segmentation, line refinement, confidence, camera pose, drift
# =====================================================================
PAINTED_LINES = [((0.0, SERVICE_FAR), (COURT_W, SERVICE_FAR), "service_far"),
                 ((0.0, SERVICE_NEAR), (COURT_W, SERVICE_NEAR), "service_near"),
                 ((COURT_W / 2, SERVICE_FAR), (COURT_W / 2, SERVICE_NEAR), "centre")]


def court_hue_estimate(frame):
    h, w = frame.shape[:2]
    roi = frame[int(0.62 * h):int(0.95 * h), int(0.2 * w):int(0.8 * w)]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    sel = hsv[..., 1] > 60
    if sel.sum() < 100:
        return None
    hist = np.bincount(hsv[..., 0][sel].ravel(), minlength=180).astype(np.float64)
    hist = np.convolve(np.r_[hist[-5:], hist, hist[:5]], np.ones(7) / 7, "same")[5:-5]
    return int(np.argmax(hist))


def court_surface_mask(frame, hue):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dh = np.abs(hsv[..., 0].astype(np.int16) - hue)
    dh = np.minimum(dh, 180 - dh)
    m = ((dh < 12) & (hsv[..., 1] > 50) & (hsv[..., 2] > 40)).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    return cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)


def line_response(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17)))


def _sample_line(p0, p1, n):
    t = np.linspace(0.03, 0.97, n)[:, None]
    return (np.array(p0) * (1 - t) + np.array(p1) * t).astype(np.float32)


def _c2i(H_c2i, pts):
    return cv2.perspectiveTransform(np.asarray(pts, np.float32).reshape(1, -1, 2), H_c2i)[0]


def line_support(frame, H_c2i, resp=None, thr=40):
    """Fraction of visible painted-line model points that land (+-1 px) on a bright line ridge."""
    resp = line_response(frame) if resp is None else resp
    h, w = resp.shape
    near = cv2.dilate((resp > thr).astype(np.uint8), np.ones((3, 3), np.uint8))
    hit = tot = 0
    for p0, p1, _ in PAINTED_LINES:
        for (x, y) in _c2i(H_c2i, _sample_line(p0, p1, 60)):
            if 2 <= x < w - 2 and 0.3 * h <= y < h - 2:
                tot += 1
                hit += int(near[int(y), int(x)] > 0)
    return (hit / tot if tot else 0.0), tot


def refine_with_lines(frame, H_c2i, base_court=None, base_img=None, search_px=10, iters=4):
    """ICP-style painted-line refinement: snap model line samples to the sub-pixel line ridge along the
    image normal, then a robust RANSAC homography over (snapped samples + weighted anchor points).
    Accepted only while the median pixel residual to the painted lines keeps falling."""
    resp = line_response(frame)
    H = H_c2i.copy()
    best = _median_px_residual(resp, H)
    for _ in range(iters):
        court_pts, img_pts, kinds = _snap_lines(resp, H, search_px)
        if len(court_pts) < 25 or len(set(kinds)) < 2:
            break
        cp = np.array(court_pts, np.float32)
        ip = np.array(img_pts, np.float32)
        if base_court is not None and base_img is not None:
            cp = np.vstack([cp] + [np.asarray(base_court, np.float32)] * 4)
            ip = np.vstack([ip] + [np.asarray(base_img, np.float32)] * 4)
        Hn, _ = cv2.findHomography(cp, ip, cv2.RANSAC, 2.0, maxIters=4000, confidence=0.999)
        if Hn is None:
            break
        score = _median_px_residual(resp, Hn)
        if score is None or best is None or score >= best - 0.05:
            break
        H, best = Hn, score
    support, _ = line_support(frame, H, resp)
    return H, support, line_residuals(resp, H), best


def _median_px_residual(resp, H):
    cp, ip, _ = _snap_lines(resp, H, 8)
    if len(cp) < 15:
        return None
    return float(np.median(np.linalg.norm(np.array(ip, np.float32) - _c2i(H, cp), axis=1)))


def _snap_lines(resp, H, search_px=10, min_resp=28):
    """Model line samples -> nearest ridge along the image normal. -> court_pts, img_pts, kinds"""
    h, w = resp.shape
    court_pts, img_pts, kinds = [], [], []
    offs = np.arange(-search_px, search_px + 1)
    for p0, p1, name in PAINTED_LINES:
        cpts = _sample_line(p0, p1, 50)
        ipts = _c2i(H, cpts)
        for j, (cp, ip) in enumerate(zip(cpts, ipts)):
            if not (search_px <= ip[0] < w - search_px and 0.3 * h <= ip[1] < h - search_px):
                continue
            nb = ipts[min(j + 1, len(ipts) - 1)] - ipts[max(j - 1, 0)]          # local tangent
            nrm = np.array([-nb[1], nb[0]]) / (np.linalg.norm(nb) + 1e-9)
            xs = np.clip(np.round(ip[0] + offs * nrm[0]).astype(int), 0, w - 1)
            ys = np.clip(np.round(ip[1] + offs * nrm[1]).astype(int), 0, h - 1)
            vals = resp[ys, xs]
            k = int(np.argmax(vals))
            if vals[k] > min_resp and 0 < k < len(offs) - 1:
                # sub-pixel ridge (parabolic fit)
                a_, b_, c_ = float(vals[k - 1]), float(vals[k]), float(vals[k + 1])
                den = a_ - 2 * b_ + c_
                sub = 0.5 * (a_ - c_) / den if abs(den) > 1e-6 else 0.0
                court_pts.append(cp)
                img_pts.append((ip[0] + (offs[k] + sub) * nrm[0], ip[1] + (offs[k] + sub) * nrm[1]))
                kinds.append(name)
    return court_pts, img_pts, kinds


def line_residuals(resp, H):
    """Perpendicular residuals (m) between the painted lines found in the image and the 10x20 m model.
    -> list of (court_y, residual_m)"""
    court_pts, img_pts, kinds = _snap_lines(resp, H, search_px=8)
    if not court_pts:
        return []
    back = cv2.perspectiveTransform(np.array(img_pts, np.float32).reshape(1, -1, 2), np.linalg.inv(H))[0]
    return [(float(cy), float(abs(bx - cx) if kd == "centre" else abs(by - cy)))
            for (cx, cy), (bx, by), kd in zip(court_pts, back, kinds)]


def camera_from_homography(H_c2i, frame_shape):
    """Assume square pixels and a centred principal point; recover focal length, camera height,
    tilt (depression angle) and horizontal FOV from the court-plane homography."""
    h, w = frame_shape[:2]
    cx, cy = w / 2.0, h / 2.0
    Hn = H_c2i / np.linalg.norm(H_c2i[:, 0])
    h1, h2, h3 = Hn[:, 0], Hn[:, 1], Hn[:, 2]
    a1, b1, c1 = h1[0] - cx * h1[2], h1[1] - cy * h1[2], h1[2]
    a2, b2, c2 = h2[0] - cx * h2[2], h2[1] - cy * h2[2], h2[2]
    cands = []
    if abs(c1 * c2) > 1e-12:
        f2 = -(a1 * a2 + b1 * b2) / (c1 * c2)
        if f2 > 0:
            cands.append(math.sqrt(f2))
    if abs(c2 * c2 - c1 * c1) > 1e-12:
        f2 = (a1 * a1 + b1 * b1 - a2 * a2 - b2 * b2) / (c2 * c2 - c1 * c1)
        if f2 > 0:
            cands.append(math.sqrt(f2))
    cands = [f for f in cands if 0.3 * w < f < 6 * w]
    if not cands:
        return None
    f = float(np.median(cands))
    Kinv = np.linalg.inv(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]))
    m1, m2, m3 = Kinv @ h1, Kinv @ h2, Kinv @ h3
    lam = 2.0 / (np.linalg.norm(m1) + np.linalg.norm(m2))
    r1, r2, t = lam * m1, lam * m2, lam * m3
    if t[2] < 0:
        r1, r2, t = -r1, -r2, -t
    R = np.column_stack([r1, r2, np.cross(r1, r2)])
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    C = -R.T @ t
    axis = R.T @ np.array([0.0, 0.0, 1.0])
    return {"focal_px": round(f, 1), "hfov_deg": round(math.degrees(2 * math.atan(w / (2 * f))), 1),
            "height_m": round(abs(float(C[2])), 2),
            "tilt_deg": round(math.degrees(math.asin(min(1.0, abs(float(axis[2]))))), 1),
            "position_m": [round(float(C[0]), 2), round(float(C[1]), 2)],
            "behind_near_baseline_m": round(float(C[1]) - COURT_L, 2),
            "orthogonality_consistency": round(float(abs(np.dot(r1, r2) / (np.linalg.norm(r1) * np.linalg.norm(r2)))), 4)}


def camera_with_uncertainty(H_c2i, frame_shape, px_noise=0.7, runs=40, seed=7):
    """Camera pose + Monte-Carlo spread (court reference points jittered by px_noise and refit).
    A nearly level camera makes focal length / height ill-conditioned - reported, never hidden."""
    base = camera_from_homography(H_c2i, frame_shape)
    ref = np.array([(0, NET_Y), (COURT_W, NET_Y), (COURT_W, SERVICE_NEAR), (0, SERVICE_NEAR),
                    (COURT_W / 2, NET_Y), (COURT_W / 2, SERVICE_NEAR), (0, SERVICE_FAR), (COURT_W, SERVICE_FAR)], np.float32)
    img = _c2i(H_c2i, ref)
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(runs):
        Hj, _ = cv2.findHomography(ref, (img + rng.normal(0, px_noise, img.shape)).astype(np.float32), 0)
        c = camera_from_homography(Hj, frame_shape) if Hj is not None else None
        if c:
            samples.append((c["height_m"], c["tilt_deg"], c["hfov_deg"]))
    out = {"estimate": base, "samples": len(samples)}
    if len(samples) >= 10:
        a = np.array(samples)
        p10, p90 = np.percentile(a, 10, axis=0), np.percentile(a, 90, axis=0)
        out.update({"height_m_range": [round(float(p10[0]), 2), round(float(p90[0]), 2)],
                    "tilt_deg_range": [round(float(p10[1]), 1), round(float(p90[1]), 1)],
                    "hfov_deg_range": [round(float(p10[2]), 1), round(float(p90[2]), 1)]})
        rel = (p90[0] - p10[0]) / max(0.1, float(np.median(a[:, 0])))
        out["conditioning"] = "good" if rel < 0.15 else ("fair" if rel < 0.35 else "ill-conditioned")
    else:
        out["conditioning"] = "ill-conditioned"
    if out["conditioning"] != "good":
        out["note"] = ("Camera is nearly level with the court; height/FOV cannot be pinned down from court "
                       "lines alone. Court mapping is unaffected (it is fitted to the lines directly).")
    return out


def surface_agreement(frame, H_c2i, hue):
    mask = court_surface_mask(frame, hue)
    h, w = mask.shape
    xs, ys = np.meshgrid(np.linspace(0.6, COURT_W - 0.6, 12), np.linspace(0.6, COURT_L - 0.6, 24))
    pts = _c2i(H_c2i, np.stack([xs.ravel(), ys.ravel()], 1))
    vis = [(int(x), int(y)) for x, y in pts if 0 <= x < w and 0 <= y < h]
    if len(vis) < 20:
        return 0.0
    return float(np.mean([mask[y, x] > 0 for x, y in vis]))


def ground_resolution(H_c2i, frame_shape):
    """Metres per pixel (along court depth) at the middle of each half."""
    Hi = np.linalg.inv(H_c2i)
    out = {}
    for name, cy in (("far", 5.0), ("near", 15.0)):
        p = _c2i(H_c2i, [(COURT_W / 2, cy)])[0]
        a = cv2.perspectiveTransform(np.array([[[p[0], p[1]]]], np.float32), Hi)[0][0]
        b = cv2.perspectiveTransform(np.array([[[p[0], p[1] + 1.0]]], np.float32), Hi)[0][0]
        c = cv2.perspectiveTransform(np.array([[[p[0] + 1.0, p[1]]]], np.float32), Hi)[0][0]
        out[name] = {"m_per_px_depth": round(float(np.linalg.norm(b - a)), 4),
                     "m_per_px_lateral": round(float(np.linalg.norm(c - a)), 4),
                     "image_xy": [round(float(p[0]), 1), round(float(p[1]), 1)]}
    return out


def auto_calibrate_v2(frames):
    """Multi-angle self-calibration delegated to `CourtGeometryAI` (`court_geometry_ai.py`)."""
    return E.run_court_autocalibration(frames)


def _rms(v):
    """Robust RMS (1.4826 x median absolute residual) - insensitive to net-mesh / player outliers."""
    return round(float(1.4826 * np.median(np.abs(v))), 4) if v else None


class DriftMonitor:
    """Validates the locked homography on live frames (painted-line support + player foot containment)."""

    def __init__(self):
        self.baseline = None
        self.history = deque(maxlen=12)
        self.low_count = 0
        self.last_check = 0.0
        self.state = "unknown"
        self.foot_out = deque(maxlen=300)

    def set_baseline(self, support):
        self.baseline = support
        self.history.clear()
        self.low_count = 0
        self.state = "locked"

    def note_foot(self, pos_m):
        self.foot_out.append(not (-0.5 <= pos_m[0] <= COURT_W + 0.5 and -0.5 <= pos_m[1] <= COURT_L + 0.5))

    def check(self, frame, H_c2i):
        sup, n = line_support(frame, H_c2i)
        self.last_check = _now()
        if n < 15:
            return self.state
        self.history.append(sup)
        if self.baseline is None:
            self.set_baseline(float(np.median(self.history)))
        cur = float(np.median(list(self.history)[-3:]))
        if cur < 0.6 * self.baseline:
            self.low_count += 1
            self.state = "drift_suspected" if self.low_count < 3 else "drift"
        else:
            self.low_count = 0
            self.state = "locked"
        return self.state

    def summary(self):
        cur = float(np.median(list(self.history)[-3:])) if self.history else None
        fo = float(np.mean(self.foot_out)) if self.foot_out else None
        return {"state": self.state, "baseline_line_support_pct": None if self.baseline is None else round(100 * self.baseline, 1),
                "current_line_support_pct": None if cur is None else round(100 * cur, 1),
                "foot_outside_court_pct": None if fo is None else round(100 * fo, 1)}


# =====================================================================
# 10. Spatial accuracy model
# =====================================================================
def tracking_repeatability(players):
    """1-sigma high-frequency jitter of raw ground measurements (m), per court half.
    Residual = raw - centred 7-sample moving average (motion within ~0.5 s is smooth)."""
    res = {"far": [], "near": []}
    for p in players:
        h = np.array(p.raw_hist, dtype=np.float64)
        if len(h) < 9:
            continue
        for k in (1, 2):
            sm = np.convolve(h[:, k], np.ones(7) / 7, "valid")
            r = h[3:-3, k] - sm
            ys = h[3:-3, 2]
            for yv, rv in zip(ys, r):
                res["far" if yv < NET_Y else "near"].append(rv)
    return {k: (round(float(np.std(v) * math.sqrt(7 / 6)), 3) if len(v) > 20 else None) for k, v in res.items()}


def accuracy_report(calib_info, players):
    rep = tracking_repeatability(players)
    lr = (calib_info or {}).get("line_residual_rms_m") or {}
    gr = (calib_info or {}).get("ground_resolution") or {}
    out = {}
    for half in ("near", "far"):
        cal = lr.get(half)
        if cal is None and gr.get(half):
            cal = round(float(calib_info.get("max_reprojection_error_px", 5)) * gr[half]["m_per_px_depth"] / 2, 3)
        trk = rep.get(half)
        tot = math.sqrt((cal or 0) ** 2 + (trk or 0) ** 2) if (cal is not None or trk is not None) else None
        out[half] = {"calibration_rms_m": cal, "tracking_jitter_m": trk,
                     "combined_1sigma_m": None if tot is None else round(tot, 2),
                     "m_per_px_depth": gr.get(half, {}).get("m_per_px_depth")}
    out["method"] = ("calibration = RMS residual of painted court lines vs the 10x20 m model (geometric ground truth); "
                     "tracking = measured 1-sigma jitter of raw pose ground points; combined in quadrature. "
                     "Not a hand-labelled player-position benchmark.")
    return out


# =====================================================================
# 11. HUD overlay (skeletons, ball, alerts)
# =====================================================================
def draw_v2_overlay(frame, tracker, ball, rally_state, alerts, now, show_skeleton=True):
    s = E.hud_scale(frame)
    if tracker is not None and show_skeleton:
        for sid, p in tracker.players.items():
            if p.kps is None or now - p.kps_time > 0.5:
                continue
            col = E.SLOT_COLORS.get(sid, (255, 255, 255))
            k = p.kps
            if p.bbox is not None and p.bbox_vel is not None:          # follow the predicted box between inferences
                shift = p.bbox_vel[:2] * float(np.clip(now - p.kps_time, 0, 0.25))
            else:
                shift = (0, 0)
            pts = [(int(k[i][0] + shift[0]), int(k[i][1] + shift[1])) for i in range(17)]
            for a, b in SKELETON:
                if k[a][2] > KCONF and k[b][2] > KCONF:
                    cv2.line(frame, pts[a], pts[b], col, max(1, int(2 * s)), cv2.LINE_AA)
            for i in range(5, 17):
                if k[i][2] > KCONF:
                    cv2.circle(frame, pts[i], max(2, int(2.5 * s)), (255, 255, 255), -1, cv2.LINE_AA)
    if ball is not None and ball.visible and ball.pos is not None:
        tr = list(ball.trail)
        for i in range(1, len(tr)):
            cv2.line(frame, (int(tr[i - 1][0]), int(tr[i - 1][1])), (int(tr[i][0]), int(tr[i][1])),
                     (0, 255, 255), max(1, int(1.5 * s)), cv2.LINE_AA)
        cv2.circle(frame, (int(ball.pos[0]), int(ball.pos[1])), max(5, int(7 * s)), (0, 255, 255), max(1, int(2 * s)), cv2.LINE_AA)
    if rally_state == "rally":
        E.draw_glass_box(frame, int(12 * s), int(12 * s), int(120 * s), int(40 * s), bg=(20, 20, 20), alpha=0.82,
                         border=(80, 200, 120))
        cv2.circle(frame, (int(26 * s), int(26 * s)), max(3, int(5 * s)), (80, 200, 120), -1, cv2.LINE_AA)
        E.put_text(frame, "RALLY", (int(38 * s), int(32 * s)), 0.5, (230, 230, 230), 1, s)
    if alerts:
        h, w = frame.shape[:2]
        pulse = 0.6 + 0.4 * abs(math.sin(now * 4))
        bh = int(46 * s)
        E.draw_glass_box(frame, 0, h - bh, w, h, bg=(0, 0, int(150 * pulse)), alpha=0.85, border=(60, 60, 255))
        txt = "MEDICAL ALERT: " + ", ".join(a["player"] for a in alerts) + " DOWN - check player immediately"
        E.put_text(frame, txt, (int(16 * s), h - int(16 * s)), 0.62, (255, 255, 255), 2, s)
        for a in alerts:
            p = tracker.players.get(int(a["player"][1:])) if tracker else None
            if p is not None and p.bbox is not None:
                x1, y1, x2, y2 = [int(v) for v in p.bbox]
                cv2.rectangle(frame, (x1 - 4, y1 - 4), (x2 + 4, y2 + 4), (0, 0, 255), max(2, int(3 * s)))


# =====================================================================
# 12. Synthetic projection benchmark (exact ground truth through the real camera geometry)
# =====================================================================
def synthetic_path(seed, half, seconds=60.0, hz=120.0):
    """Padel-like movement: sprints/shuffles to random targets in one half, with pauses."""
    rng = np.random.default_rng(seed)
    y_lo, y_hi = (0.8, NET_Y - 0.8) if half == "far" else (NET_Y + 0.8, COURT_L - 0.8)
    pos = np.array([rng.uniform(2, 8), rng.uniform(y_lo, y_hi)])
    vel = np.zeros(2)
    out, t, next_change, target, vmax = [], 0.0, 0.0, pos.copy(), 0.0
    dt = 1.0 / hz
    while t < seconds:
        if t >= next_change:
            if rng.random() < 0.25:
                target, vmax = pos.copy(), 0.0                         # pause / ready position
            else:
                target = np.array([rng.uniform(0.8, COURT_W - 0.8), rng.uniform(y_lo, y_hi)])
                vmax = rng.uniform(1.2, 5.5)
            next_change = t + rng.uniform(0.6, 2.2)
        d = target - pos
        dist = np.linalg.norm(d)
        want = (d / dist) * min(vmax, dist * 2.5) if dist > 1e-6 else np.zeros(2)
        dv = want - vel
        a_max = 6.0 * dt
        if np.linalg.norm(dv) > a_max:
            dv = dv / np.linalg.norm(dv) * a_max
        vel = vel + dv
        pos = pos + vel * dt
        out.append((t, pos[0], pos[1]))
        t += dt
    return np.array(out)


def projection_bias_test(H_i2c, px_sigma=5.0, fps=15.0, seconds=60.0, seeds=(1, 2, 3, 4, 5, 6)):
    """Projects known paths into the camera, adds measured pixel noise, tracks them, and reports
    measured/true distance per court half. -> {"far": {...}, "near": {...}}"""
    H_c2i = np.linalg.inv(H_i2c)
    res = {}
    for half in ("far", "near"):
        ratios, spd_err = [], []
        for sd in seeds:
            path = synthetic_path(sd, half, seconds)
            true_d = float(np.sum(np.linalg.norm(np.diff(path[:, 1:], axis=0), axis=1)))
            step = max(1, int(round(120.0 / fps)))
            samp = path[::step]
            rng = np.random.default_rng(100 + sd)
            img = _c2i(H_c2i, samp[:, 1:]) + rng.normal(0, px_sigma, (len(samp), 2))
            tr = TrackerV2(H_i2c, "padel")
            pl = tr.players[1]
            for (t, _, _), ip in zip(samp, img):
                pos = tr.pixel_to_meters(ip)
                sig = tr._local_sigma(ip, pos, px_sigma)
                bb = (ip[0] - 15, ip[1] - 60, ip[0] + 15, ip[1])
                pl.update(bb, (int(ip[0]), int(ip[1])), pos, 1000.0 + t, meas_sigma=sig)
            ratios.append(pl.distance_m / max(1e-6, true_d))
            spd_err.append(pl.avg_speed_ms - true_d / seconds)
        res[half] = {"distance_ratio_mean": round(float(np.mean(ratios)), 3),
                     "distance_ratio_min": round(float(np.min(ratios)), 3),
                     "distance_ratio_max": round(float(np.max(ratios)), 3),
                     "avg_speed_bias_ms": round(float(np.mean(spd_err)), 3), "runs": len(seeds)}
    res["px_sigma"] = px_sigma
    res["fps"] = fps
    return res
