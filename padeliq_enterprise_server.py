# -*- coding: utf-8 -*-
"""
PadelIQ Enterprise Command Center
=================================

Self-hosted (offline-capable) web command & control dashboard for the PadelIQ engine.

    python padeliq_enterprise_server.py                # http://localhost:8000
    python padeliq_enterprise_server.py --source 0     # start on webcam 0
    python padeliq_enterprise_server.py --open-browser
    python padeliq_enterprise_server.py --source webcam # best attached webcam (Logitech BRIO preferred)

Pipeline (fully decoupled, nothing blocks the video):
    Capture thread   ThreadedLiveCamera  (webcam/DirectShow, ESP32 MJPEG, RTSP, YouTube via yt-dlp, files)
    Inference thread YOLO on the newest frame only (stride = whatever the CPU/GPU sustains)
    Render thread    event-driven: wakes the instant a camera frame lands, renders HUD + JPEG at the camera's
                     native rate (capped by --render-fps, default 60) with velocity-predicted boxes
    Telemetry thread coach feed + system health (WebSocket push @ 10 Hz)

HTTP API
    GET  /                       dashboard
    GET  /video.mjpg             annotated MJPEG stream
    GET  /api/status             health
    GET  /api/telemetry          players / heatmap / coach feed / system
    WS   /ws/telemetry           same payload pushed at 10 Hz
    GET  /api/snapshot[?clean=1] latest annotated (or clean) JPEG
    GET  /api/calibration        current calibration
    POST /api/calibration        {"points": [[x,y]x4], "reference": "full|far_half|near_half"}
    POST /api/calibration/auto   (re-)run AI court recognition -> review state
    POST /api/calibration/confirm   lock the reviewed calibration
    POST /api/calibration/manual    enter manual adjust (returns draggable handles)
    POST /api/calibration/handles   {"court_m": [[x,y]x4], "image_px": [[x,y]x4]} dragged handles
    POST /api/source             {"type": "sim|webcam|esp32|rtsp|youtube|auto|url", "value": "..."}
                                 webcam value "" = best attached camera (BRIO preferred), or an index
    POST /api/reset              reset player statistics
    POST /api/hud                {"mode": "compact|full"}
    GET  /api/report.json | /api/report.csv | /api/report.html   official session report
"""
import os
import sys
import csv
import io
import json
import time
import socket
import asyncio
import argparse
import threading
import traceback
import webbrowser
from collections import deque
from datetime import datetime

import numpy as np
import cv2

import padeliq_live_stream as E
import court_geometry_ai as CG
import padeliq_mobile_bridge as MB
import padeliq_orientation as ORI
import padeliq_score as SC

try:
    import psutil
except Exception:          # optional
    psutil = None

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse)
import uvicorn

APP_DIR = E.APP_DIR
UI_DIR = os.path.join(APP_DIR, "enterprise_ui")
VERSION = "1.5.0"
IDLE_RENDER_FPS = 10.0      # refresh rate for banners when no camera frames arrive
SWITCH_FIRST_FRAME_TIMEOUT_S = 20.0   # source swap: keep the current feed until the new one delivers (or give up)
log = E.log


def _safe(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


class RateMeter:
    """Events per second over a sliding window."""

    def __init__(self, window=2.0):
        self.window = window
        self.ts = deque()
        self.lock = threading.Lock()

    def tick(self, t=None):
        t = time.time() if t is None else t
        with self.lock:
            self.ts.append(t)
            while self.ts and t - self.ts[0] > self.window:
                self.ts.popleft()

    @property
    def rate(self):
        with self.lock:
            now = time.time()
            while self.ts and now - self.ts[0] > self.window:
                self.ts.popleft()
            if len(self.ts) < 2:
                return 0.0
            span = max(1e-3, self.ts[-1] - self.ts[0])
            return (len(self.ts) - 1) / span


# =====================================================================
# Engine
# =====================================================================
class PadelIQEngine:
    def __init__(self, args):
        self.args = args
        self.lock = threading.RLock()          # slot manager + calibration + camera swap
        self.running = True
        self.started_at = time.time()
        self.session_started_at = time.time()

        self.detector = E.PersonDetector(args.model, imgsz=args.imgsz, conf=args.conf, backend=args.ai_backend,
                                         pose_stride=args.pose_stride, use_roi=not args.no_roi)
        self.exposure_pref = args.exposure          # applied to every webcam this engine opens
        self.calib_locked = False
        self._stash = {}                            # per-source confirmed calibration (failover swaps)
        self._spec_cache = {}
        self._roi_key = None
        self.epoch = 0
        self.failover_primary = args.source
        self.failover_on_alt = False
        self._toast = ("", 0.0)
        self.cam = None
        self.spec = None
        self.source_state = "starting"         # starting | switching | ready | error
        self.source_error = ""
        self.switch_lock = threading.Lock()

        self.slot_mgr = E.LiveSlotManager(None)
        self.corners = None
        self.H = None
        self.canvas_shape = None
        self.calibration_info = {}
        self.hud_mode = args.hud

        self.jpeg = b""
        self.jpeg_seq = 0
        self.jpeg_cond = threading.Condition()
        self.frame_waiters = set()             # (event loop, asyncio.Event) per MJPEG viewer
        # QR phone camera (HTTPS bridge on :8443 started in main(); routes also mounted on the dashboard app)
        self.mobile = MB.MobileHub(https_port=args.mobile_port, lan_ip=args.lan_ip,
                                   fiducial_size_cm=args.fiducial_size_cm)
        self.mobile.controller = EngineMobileController(self)
        self._waiting = (None, 0.0, "")         # cached QR waiting screen (img, built_at, detail)
        self.rotation = 0                       # 0/90/180/270 applied to every camera frame before anything else
        self.pending_autolock = False           # confirm + lock the court as soon as auto-calibration lands
        self._oriented_cam = None
        self._reorient_at = 0.0
        self._last_raw = None
        self.clean_canvas = None
        self.clean_lock = threading.Lock()

        self.capture_meter = RateMeter()
        self.infer_meter = RateMeter()
        self.render_meter = RateMeter()
        self.infer_latency_ms = 0.0
        self.render_ms = 0.0
        self.e2e_ms = deque(maxlen=300)        # frame captured/arrived -> annotated JPEG ready (ms)
        self.frame_age_ms = deque(maxlen=300)  # frame captured/arrived -> render start (queueing delay)
        self.frames_inferred = 0
        self.frames_rendered = 0
        self.errors = {"capture": 0, "inference": 0, "render": 0, "telemetry": 0, "api": 0}
        self.last_errors = deque(maxlen=20)
        self.mjpeg_clients = 0
        self.ws_clients = 0

        # auto-calibration review + scene-cut detector + ball tracking
        self.review = E.CalibrationReview()
        self.scene_cut = CG.SceneCutDetector()
        self.applied_version = -1
        self.calib_frames = deque(maxlen=24)
        self._last_calib_grab = 0.0
        self.ball = E.PadelBallTracker()
        self.score = SC.PadelScoreEngine()
        self.score.listeners.append(self._on_point)
        self.last_boxes = np.zeros((0, 4), np.float32)
        self.last_yolo_balls = np.zeros((0, 4), np.float32)
        self.last_kps = []
        self.last_det_ts = None
        self.ball_worker = E.BallWorker(lambda: self.cam, self.ball,
                                        lambda: (self.last_boxes, self.last_yolo_balls, self.last_kps, self.last_det_ts),
                                        args.hud_min_height, rotation_getter=lambda: self.rotation)

        self.coach_feed = deque(maxlen=200)
        self._last_tip = {}
        self._last_tip_time = {}
        self.system = {}

        self.threads = []
        for name, target in (("inference", self._inference_loop), ("render", self._render_loop),
                             ("telemetry", self._telemetry_loop)):
            t = threading.Thread(target=self._guarded(name, target), name=f"PadelIQ-{name}", daemon=True)
            t.start()
            self.threads.append(t)

    # ------------------------------------------------------------ resilience
    def _record_error(self, area, exc):
        self.errors[area] = self.errors.get(area, 0) + 1
        msg = f"{datetime.now():%H:%M:%S} [{area}] {exc!r}"
        self.last_errors.append(msg)
        if self.errors[area] <= 5 or self.errors[area] % 100 == 0:
            log(f"Recovered from {area} error: {exc!r}")

    def _guarded(self, name, target):
        """Self-healing thread wrapper: any crash restarts the loop after a short pause."""
        def run():
            while self.running:
                try:
                    target()
                except Exception as exc:
                    self._record_error(name if name in self.errors else "api", exc)
                    traceback.print_exc()
                    time.sleep(0.5)
        return run

    # ---------------------------------------------------------------- source
    def _on_point(self, ev):
        st, now = ev["score"], ev["t"]
        line = (f"POINT {SC.TEAM_LABEL[ev['team']]}!{' GAME!' if ev.get('game') else ''}  "
                f"{st['points']['A']} : {st['points']['B']} | Games {st['games']['A']}-{st['games']['B']}")
        log(line + (f" (auto: {ev['reason']})" if ev["auto"] else " (manual)"))
        self.coach_feed.appendleft({
            "t": round(now, 2), "time": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
            "session_s": round(now - self.session_started_at, 1), "player": "SCORE",
            "text": line + (" - auto" if ev["auto"] else " - manual"), "speed_ms": None, "stability_pct": None})

    def score_point(self, team):
        return self.score.point(team, reason="web +1")

    def toast(self, msg, secs=2.5):
        self._toast = (msg, time.time() + secs)
        log(msg)

    def toast_text(self):
        msg, until = self._toast
        return msg if msg and time.time() < until else ""

    @staticmethod
    def _src_key(spec):
        return f"{spec.kind}:{spec.source}"

    def _stash_calibration(self):
        """Lock held: remember the confirmed calibration of the current source."""
        if self.spec is not None and self.review.state == "confirmed" and self.corners is not None \
                and self.canvas_shape is not None:
            self._stash[self._src_key(self.spec)] = {
                "corners": self.corners.copy(), "info": dict(self.review.info), "shape": self.canvas_shape,
                "locked": self.calib_locked, "floor_mask": self.slot_mgr.floor_mask,
                "floor_top_y": self.slot_mgr.floor_top_y, "calibration_info": dict(self.calibration_info),
                "rotation": self.rotation}

    def _restore_calibration(self, spec):
        """Lock held: restore the target source's calibration, or start a fresh one."""
        self.ball.reset()
        self.pending_autolock = False
        st = self._stash.get(self._src_key(spec))
        if st:
            self.canvas_shape = st["shape"]
            self.corners = st["corners"].copy()
            self.H = E.compute_homography(self.corners)
            self.review.restore(self.corners, st["info"], st["shape"])
            self.applied_version = self.review.version
            self.slot_mgr.set_homography(self.H)
            self.slot_mgr.floor_mask, self.slot_mgr.floor_top_y = st["floor_mask"], st["floor_top_y"]
            self.ball.set_geometry(self.H, self.canvas_shape)
            self.calibration_info = st["calibration_info"]
            self.calib_locked = st["locked"]
            self.rotation = st.get("rotation", 0)
        else:
            self.rotation = 0
            self.canvas_shape = None          # new source -> provisional geometry + fresh auto-calibration
            self.corners = self.H = None
            self.review.reset()
            self.calibration_info = {}
            self.calib_locked = False
        self._roi_key = None
        self._update_roi()

    def set_source(self, source, esp32_ip=None, background=True, failover=False):
        def work():
            with self.switch_lock:
                had_cam = self.cam is not None
                self.source_state = "switching"
                self.source_error = ""
                try:
                    spec = self._spec_cache.get(source) if failover else None
                    spec = spec or E.resolve_source(source, esp32_ip=esp32_ip, backend=self.args.backend)
                    if spec is None:
                        raise ValueError(f"unrecognised source {source!r}")
                    if failover:
                        self._spec_cache[source] = spec
                    if spec.kind == "mobile":
                        new_cam = self.mobile.create_camera(spec, stall_timeout=self.args.stall_timeout)
                    else:
                        new_cam = E.ThreadedLiveCamera(spec, stall_timeout=self.args.stall_timeout,
                                                       cam_width=self.args.cam_width, cam_height=self.args.cam_height,
                                                       cam_fps=self.args.cam_fps, exposure=self.exposure_pref,
                                                       sport_exposure=self.args.sport_exposure)
                except Exception as exc:
                    self.source_state = "ready" if had_cam else "error"
                    self.source_error = str(exc)
                    self.toast(f"Source switch failed: {exc}", 4.0)
                    return
                # Seamless swap: the current source keeps streaming until the new one delivers its first frame.
                # (A phone that has not scanned the QR yet switches at once -> the QR pairing screen is shown.)
                if had_cam and (spec.kind != "mobile" or self.mobile.connected):
                    t_end = time.time() + SWITCH_FIRST_FRAME_TIMEOUT_S
                    while self.running and new_cam.frame_id == 0 and time.time() < t_end:
                        time.sleep(0.02)
                    if new_cam.frame_id == 0:
                        threading.Thread(target=new_cam.release, daemon=True).start()
                        self.source_state = "ready"
                        self.source_error = f"{spec.label}: no frames within {SWITCH_FIRST_FRAME_TIMEOUT_S:.0f} s"
                        self.toast(f"Switch cancelled - {self.source_error}", 4.0)
                        return
                with self.lock:
                    old = self.cam
                    self._stash_calibration()
                    self.cam, self.spec = new_cam, spec
                    self.epoch += 1
                    self._restore_calibration(spec)
                    self.scene_cut.reset()
                    self.calib_frames.clear()
                    self.slot_mgr.reset_players()
                    self.last_boxes, self.last_yolo_balls, self.last_kps = (np.zeros((0, 4), np.float32),
                                                                            np.zeros((0, 4), np.float32), [])
                    self.session_started_at = time.time()
                    self.coach_feed.clear()
                    if failover:
                        self.failover_on_alt = not self.failover_on_alt
                    else:
                        self.failover_primary, self.failover_on_alt = source, False
                self.source_state = "ready"
                self.toast(f"Source active: {spec.label}")
                if old is not None:
                    threading.Thread(target=old.release, daemon=True).start()
        if background:
            threading.Thread(target=work, daemon=True).start()
        else:
            work()

    def failover(self):
        """[F12] live camera <-> clean replay (or --failover-source); the current view stays live until ready."""
        if self.source_state == "switching":
            raise ValueError("a source switch is already in progress")
        if self.failover_on_alt:
            target = self.failover_primary
        else:
            kind = self.spec.kind if self.spec else "file"
            target = self.args.failover_source or ("sim" if kind != "file" else "webcam")
        self.toast(f"Failover -> {target} (current view stays live until it is ready)", 3.0)
        self.set_source(target, failover=True)
        return target

    def set_exposure(self, mode):
        cam = self.cam
        if mode == "toggle":
            cur = cam.exposure_mode if cam is not None else self.exposure_pref
            mode = "sport" if cur == "auto" else "auto"
        if mode not in E.EXPOSURE_MODES:
            raise ValueError("mode must be auto | sport | toggle")
        self.exposure_pref = mode
        if cam is not None:
            cam.set_exposure_mode(mode)
        applies = cam is not None and cam.kind == "webcam"
        self.toast(f"Camera: {E.EXPOSURE_LABELS[mode]}" + ("" if applies else " (applies to USB webcams)"))
        return {"mode": mode, "label": E.EXPOSURE_LABELS[mode], "applies_to_current_source": applies}

    def open_camera_settings(self):
        cam = self.cam
        if cam is None or cam.kind != "webcam":
            raise ValueError("the current source is not a USB webcam")
        cam.open_driver_settings()
        return True

    def set_lock(self, locked):
        with self.lock:
            if locked == "toggle":
                locked = not self.calib_locked
            locked = bool(locked)
            if locked:
                self._sync_review()
                if self.review.state in ("review", "manual") and self.review.confirm("operator (lock)"):
                    self._save_confirmed()
                if self.review.state != "confirmed":
                    raise ValueError("confirm the court calibration before locking it")
            self.calib_locked = locked
        self.toast("Calibration Locked" if locked else "Calibration Unlocked")
        return locked

    # ----------------------------------------------------------- orientation
    def set_rotation(self, deg, auto_lock=False, why=""):
        """Lock held. New rotation -> fresh geometry + court re-calibration on the next (rotated) frame."""
        self.rotation = ORI.norm_deg(deg)
        self.epoch += 1
        self.corners = self.H = None
        self.canvas_shape = None                 # _ensure_geometry() restarts auto-calibration on the next frame
        self.calibration_info = {}
        self.review.reset()
        self.slot_mgr.set_homography(None)
        self.slot_mgr.reset_players()
        self.ball.reset()
        self.scene_cut.reset()
        self.calib_frames.clear()
        self.last_boxes, self.last_yolo_balls, self.last_kps = (np.zeros((0, 4), np.float32),
                                                                np.zeros((0, 4), np.float32), [])
        self.pending_autolock = bool(auto_lock)
        self._roi_key = None
        self._update_roi()
        self.toast(f"Camera rotation {self.rotation} deg{(' - ' + why) if why else ''} - re-calibrating court", 2.5)

    def auto_orient(self, raw, auto_lock=False, first=False):
        """Lock held. Score the current rotation; below 55 % test 90 CW / 90 CCW / 180 and take the most upright."""
        deg, score, scores = ORI.detect_orientation(raw, None if first else self.rotation)
        log(f"Orientation check {raw.shape[1]}x{raw.shape[0]}: " +
            ", ".join(f"{d} deg {v:.0f}%" for d, v in scores.items()) + f" -> {deg} deg")
        self.set_rotation(deg, auto_lock=auto_lock, why=f"auto-orientation {score:.0f}%")
        return deg

    def rotate(self, step=90):
        """Always allowed (rotation invalidates the geometry): releases the stage lock, re-locks after."""
        raw = self._last_raw
        new = ORI.norm_deg(self.rotation + int(step))
        upright = raw is None or ORI.orientation_score(ORI.landscape(raw, new))[0] >= ORI.MIN_CONF
        with self.lock:
            self.calib_locked = False
            self.set_rotation(new, auto_lock=bool(self.spec and self.spec.kind == "mobile") and upright,
                              why="operator")
            if not upright:                  # never auto-lock a court that looks sideways / upside down
                self.toast(f"Rotated to {new} deg - court does not look upright: Rotate again or Auto-calibrate", 3.5)
        return self.rotation

    def _autolock_tick(self):
        """Lock held: instant court lock after an orientation check (no 3 s review wait)."""
        rv = self.review
        if not self.pending_autolock:
            return
        if rv.state == "review" and rv.corners is not None and self.corners is not None:
            self.pending_autolock = False
            if rv.confirm("auto (orientation lock)"):
                self._save_confirmed()
                self.calib_locked = True
                self.toast(f"Court locked at {self.rotation} deg ({rv.info.get('confidence_pct')}%) - analytics live", 3.0)
        elif rv.state == "failed":
            self.pending_autolock = False
            self.toast("Court not found - press Rotate or drag the corners", 3.5)

    def _check_unlocked(self):
        if self.calib_locked:
            raise ValueError("calibration is locked - unlock it first")

    def _update_roi(self):
        """Lock held: court-ROI guided inference once the calibration is confirmed."""
        rv = self.review
        key = (rv.state == "confirmed", rv.version, self.canvas_shape, self.applied_version)
        if key == self._roi_key:
            return
        self._roi_key = key
        roi = None
        if key[0] and self.corners is not None and self.canvas_shape is not None:
            roi = CG.PoseCourtDetector.court_roi(self.corners, self.canvas_shape)
        self.detector.set_roi(roi)

    # ----------------------------------------------------------- calibration
    def _ensure_geometry(self, canvas):
        """Called with self.lock held. Never falls back to a fake static trapezoid."""
        shape = canvas.shape[:2]
        if shape == self.canvas_shape:
            return
        if self.canvas_shape is not None and E.aspect_changed(self.canvas_shape, shape):
            log(f"Frame aspect changed {self.canvas_shape[1]}x{self.canvas_shape[0]} -> {shape[1]}x{shape[0]} "
                f"(phone rotated / lens switched) -> fresh court calibration.")
            self.corners = self.H = None
            self.review.reset()
        if self.corners is not None and self.canvas_shape is not None:
            oh, ow = self.canvas_shape
            self.corners = self.corners * np.array([shape[1] / ow, shape[0] / oh], dtype=np.float32)
            self.H = E.compute_homography(self.corners)
        else:
            self.corners, self.H, self.calibration_info = None, None, {}
        self.canvas_shape = shape
        self.slot_mgr.set_homography(self.H)
        self.slot_mgr.reset_players()
        self.ball.set_geometry(self.H, shape)
        if self.review.state == "idle":
            self._start_review(seed_frame=canvas)

    # ------------------------------------------------- auto-calibration review
    def _calibration_frames(self):
        cam = self.cam
        if cam is not None:
            collected = list(self.calib_frames)[-6:]
            last_id = -1
            t_end = time.time() + 0.75
            while time.time() < t_end and len(collected) < 15:
                raw, fid, _ = cam.read_latest()
                if raw is not None and fid != last_id:
                    last_id = fid
                    collected.append(E.prepare_canvas(raw, self.args.hud_min_height, self.rotation))
                else:
                    time.sleep(0.012)
            if len(collected) >= 5:
                return collected
        spec = self.spec
        if spec is not None and spec.kind == "file":
            return [E.prepare_canvas(f, self.args.hud_min_height, self.rotation)
                    for f in E.collect_calibration_frames(spec, count=16)]
        end = time.time() + 1.4
        while time.time() < end and len(self.calib_frames) < 15:
            time.sleep(0.04)
        return list(self.calib_frames)

    def _start_review(self, seed_frame=None):
        self.calib_frames.clear()
        if seed_frame is not None:
            self.calib_frames.append(seed_frame.copy())
            self.scene_cut.set_reference(seed_frame)
        self.review.start(self._calibration_frames, self.canvas_shape, seed_frame=seed_frame)

    def _sync_review(self):
        """Called with self.lock held: auto-confirm timer + apply review corners & floor mask to tracker/ball."""
        if self.review.tick():
            self._save_confirmed()
        rv = self.review
        if rv.version != self.applied_version and rv.corners is not None and self.canvas_shape is not None:
            H = E.compute_homography(rv.corners)
            if H is not None:
                ai = E.get_court_ai()
                self.corners, self.H = rv.corners.copy(), H
                self.slot_mgr.set_homography(
                    H, floor_mask=ai.last_floor_mask, floor_top_y=rv.info.get("floor_top_y", ai.last_floor_top_y),
                    floor_bottom_y=rv.info.get("floor_bottom_y"), occluder_bottom_y=rv.info.get("occluder_bottom_y")
                )
                if rv.drag_idx is None and not rv.info.get("provisional"):
                    self.slot_mgr.reset_players()
                self.ball.set_geometry(H, self.canvas_shape)
                self.calibration_info = dict(rv.info)
            self.applied_version = rv.version
        self._autolock_tick()
        self._update_roi()

    def _save_confirmed(self):
        if self.corners is None or self.canvas_shape is None:
            return
        h, w = self.canvas_shape
        save_path, _ = E.calibration_files_for(self.spec.kind if self.spec else "file")
        extra = {k: v for k, v in dict(self.review.info, source_label=self.spec.label if self.spec else None).items()
                 if k not in ("corners", "frame_size")}
        _safe(lambda: E.save_calibration(save_path, self.corners, (h, w, 3), extra=extra))
        self.calibration_info = dict(self.review.info, file=os.path.basename(save_path))

    def confirm_calibration(self, by="operator (web)"):
        with self.lock:
            self._sync_review()
            if not self.review.confirm(by):
                raise ValueError(f"nothing to confirm (calibration state: {self.review.state})")
            self._save_confirmed()
        return self.review.status()

    def manual_calibration(self):
        self._check_unlocked()
        with self.lock:
            if self.canvas_shape is None:
                raise ValueError("no video frame yet")
            self.review.enter_manual(self.corners, (self.canvas_shape[0], self.canvas_shape[1]))
            self._sync_review()
        return self.review_payload()

    def set_handles(self, court_m, image_px, dragged_idx=None):
        self._check_unlocked()
        self.review.set_from_handles(court_m, image_px, dragged_idx=dragged_idx)
        with self.lock:
            self._sync_review()
        return self.review_payload()

    def review_payload(self):
        st = self.review.status()
        st["lines_px"] = court_lines_px(self.review.corners, self.canvas_shape or (720, 1280))
        st["frame_size"] = [self.canvas_shape[1], self.canvas_shape[0]] if self.canvas_shape else None
        return st

    def apply_calibration(self, corners, info):
        """Manual 4-point (web clicks) calibration: applied and confirmed immediately."""
        with self.lock:
            if self.canvas_shape is None:
                raise ValueError("no video frame yet")
            if E.compute_homography(corners) is None:
                raise ValueError("points do not form a valid court")
            self.review.frame_shape = tuple(self.canvas_shape)
            self.review.set_manual_corners(corners, info)
            self.review.confirm("operator (web 4-point)")
            self._sync_review()
            self._save_confirmed()

    def calibrate_from_points(self, points, reference):
        self._check_unlocked()
        if reference not in E.CALIBRATION_REFERENCES:
            raise ValueError(f"reference must be one of {list(E.CALIBRATION_REFERENCES)}")
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape != (4, 2) or not np.all(np.isfinite(pts)):
            raise ValueError("exactly 4 [x, y] points are required")
        corners = E.corners_from_reference(pts, reference)
        if corners is None:
            raise ValueError("points must be convex and clicked in order TL, TR, BR, BL")
        self.apply_calibration(corners, {"method": "manual_web", "reference": reference,
                                         "reference_image_points": pts.round(1).tolist()})
        return corners

    def auto_calibrate(self, wait_s=30.0):
        self._check_unlocked()
        """(Re-)run AI court recognition; returns once the result is in review (or failed)."""
        if self.spec is None or self.canvas_shape is None:
            raise ValueError("no active source / frame yet")
        raw = self._last_raw
        if raw is not None:
            with self.lock:
                self.auto_orient(raw, auto_lock=bool(self.spec and self.spec.kind == "mobile"))
        else:
            with self.clean_lock:
                seed = None if self.clean_canvas is None else self.clean_canvas.copy()
            with self.lock:
                self._start_review(seed_frame=seed)
        end = time.time() + wait_s
        while time.time() < end and self.review.state in ("idle", "detecting"):
            time.sleep(0.1)
        if self.review.state == "failed":
            raise ValueError(f"auto-calibration failed: {self.review.error}")
        if self.review.state == "detecting":
            raise ValueError("auto-calibration still running - check /api/calibration")
        with self.lock:
            self._sync_review()
        return self.review.corners, self.review.info

    # -------------------------------------------------------------- workers
    def _inference_loop(self):
        last_fid = -1
        last_cam = None
        while self.running:
            cam = self.cam
            if cam is None:
                time.sleep(0.05)
                continue
            if cam is not last_cam:
                last_cam, last_fid = cam, -1
            frame, fid, ts = E.cam_wait_latest(cam, last_fid, 0.05)
            if frame is None or fid == last_fid:
                continue
            last_fid = fid
            try:
                if cam.kind == "mobile" and self._reorient_at and time.time() >= self._reorient_at:
                    self._reorient_at = 0.0                     # phone rotated: explicit operator action -> re-lock
                    with self.lock:
                        self.calib_locked = False
                        self.auto_orient(frame, auto_lock=True, first=True)
                elif cam.kind == "mobile" and self._oriented_cam is not cam and self.review.state != "confirmed":
                    self._oriented_cam = cam                    # first phone frame: orientation check + instant lock
                    with self.lock:
                        self.auto_orient(frame, auto_lock=True, first=True)
                rot = self.rotation
                canvas = E.prepare_canvas(frame, self.args.hud_min_height, rot)
                self._last_raw = frame
                with self.lock:
                    if cam is not self.cam or rot != self.rotation:
                        continue
                    self._ensure_geometry(canvas)
                    if self.review.state == "detecting":
                        if time.time() - self._last_calib_grab > 0.03:
                            self.calib_frames.append(canvas)
                            self._last_calib_grab = time.time()
                        self._sync_review()
                        time.sleep(0.035)
                        continue
                    elif self.review.state in ("review", "confirmed", "failed") and not self.calib_locked:
                        cut, metrics = self.scene_cut.update(canvas, now=ts)
                        if cut:
                            log(f"Scene cut detected (HSV diff={metrics.get('hist_diff')}, SSIM={metrics.get('ssim')}) "
                                f"-> fast 15-frame court re-calibration!")
                            self.corners = self.H = None
                            self.slot_mgr.set_homography(None)
                            self.slot_mgr.reset_players()
                            self.ball.reset()
                            self._start_review(seed_frame=canvas)
                    self._sync_review()
                    epoch = self.epoch
                t0 = time.perf_counter()
                boxes = self.detector(canvas)
                if epoch == self.epoch:
                    self.last_boxes, self.last_yolo_balls = boxes, self.detector.last_balls
                    self.last_kps, self.last_det_ts = self.detector.last_keypoints, ts
                self.infer_latency_ms = 0.8 * self.infer_latency_ms + 0.2 * (time.perf_counter() - t0) * 1000
                with self.lock:
                    if cam is self.cam and epoch == self.epoch:
                        self.slot_mgr.update_detections(
                            boxes, now=ts, keypoints=self.detector.last_keypoints, frame_shape=canvas.shape
                        )
                        if self.slot_mgr.wants_full_frame(ts):
                            self.detector.full_frame_next = True
                self.frames_inferred += 1
                self.infer_meter.tick()
            except Exception as exc:
                self._record_error("inference", exc)
                time.sleep(0.05)

    @property
    def _jpeg_params(self):
        return [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.jpeg_quality), int(cv2.IMWRITE_JPEG_OPTIMIZE), 0]

    def _render_loop(self):
        """Event-driven: blocks on the camera's latest-frame slot and renders the moment a frame lands, so
        video FPS = camera FPS (capped at --render-fps) and no frame ever waits for a fixed render tick."""
        min_dt = 1.0 / max(1.0, float(self.args.render_fps))
        idle_dt = 1.0 / IDLE_RENDER_FPS
        last_fid, last_canvas, last_cam, last_ts = -1, None, None, 0.0
        last_render = 0.0
        while self.running:
            try:
                cam = self.cam
                if cam is not last_cam:
                    last_cam, last_fid = cam, -1          # keep showing the last frame until the new source's first
                new = False
                if cam is not None:
                    frame, fid, ts = E.cam_wait_latest(cam, last_fid, idle_dt)
                    if frame is not None and fid != last_fid:
                        if last_fid >= 0:
                            for _ in range(min(10, fid - last_fid)):
                                self.capture_meter.tick()
                        last_fid, last_ts, new = fid, ts, True
                        last_canvas = E.prepare_canvas(frame, self.args.hud_min_height, self.rotation)
                        with self.clean_lock:
                            self.clean_canvas = last_canvas
                else:
                    time.sleep(idle_dt)
                since = time.perf_counter() - last_render
                if not new and since < idle_dt:
                    continue                                   # nothing new: no duplicate frames on the wire
                if new and since < min_dt:
                    time.sleep(min_dt - since)                 # only when the camera outruns the FPS cap
                t0 = time.perf_counter()
                last_render = t0
                if new:
                    self.frame_age_ms.append((time.time() - last_ts) * 1000.0)
                if isinstance(cam, MB.MobileCamera) and cam.waiting_for_phone:
                    self._render_waiting(cam)
                    self.render_ms = 0.8 * self.render_ms + 0.2 * (time.perf_counter() - t0) * 1000
                    continue
                if last_canvas is not None:
                    hud = last_canvas.copy()
                else:
                    hud = E.placeholder_frame()
                degraded = cam is None or cam.is_degraded
                if cam is None:
                    detail = "Starting ..."
                elif cam.frame_id == 0:
                    detail = f"Connecting to {self.spec.label if self.spec else ''} {cam.last_error}".strip()
                else:
                    detail = f"No frames for {cam.seconds_since_frame:.1f}s | attempt #{max(1, cam.reconnect_count)}"
                with self.lock:
                    have_geo = last_canvas is not None and self.canvas_shape == last_canvas.shape[:2]
                    if have_geo:
                        self._sync_review()
                        if self.review.state == "confirmed" and not degraded:
                            self.score.update_from_ball(self.ball)
                    E.render_live_hud(hud, self.slot_mgr if have_geo else None, self.corners if have_geo else None,
                                      self.infer_meter.rate, H=self.H,
                                      source_label=self.spec.label if self.spec else "starting",
                                      reconnecting=degraded, reconnect_detail=detail, hud_mode=self.hud_mode,
                                      ball=self.ball if have_geo else None,
                                      review=self.review if have_geo else None, web=True,
                                      held=last_canvas is not None, locked=self.calib_locked,
                                      toast=self.toast_text(), score=self.score)
                # Encoded once per render tick; every MJPEG viewer / projector tab shares these bytes.
                ok, buf = cv2.imencode(".jpg", hud, self._jpeg_params)
                if ok:
                    with self.jpeg_cond:
                        self.jpeg = buf.tobytes()
                        self.jpeg_seq += 1
                        self.jpeg_cond.notify_all()
                    for loop, ev in list(self.frame_waiters):      # wake async MJPEG viewers (no polling)
                        try:
                            loop.call_soon_threadsafe(ev.set)
                        except RuntimeError:
                            self.frame_waiters.discard((loop, ev))
                    self.frames_rendered += 1
                    self.render_meter.tick()
                    if new:
                        self.e2e_ms.append((time.time() - last_ts) * 1000.0)
                self.render_ms = 0.8 * self.render_ms + 0.2 * (time.perf_counter() - t0) * 1000
            except Exception as exc:
                self._record_error("render", exc)
                time.sleep(0.05)

    def _publish_jpeg(self, img):
        ok, buf = cv2.imencode(".jpg", img, self._jpeg_params)
        if not ok:
            return False
        with self.jpeg_cond:
            self.jpeg = buf.tobytes()
            self.jpeg_seq += 1
            self.jpeg_cond.notify_all()
        for loop, ev in list(self.frame_waiters):
            try:
                loop.call_soon_threadsafe(ev.set)
            except RuntimeError:
                self.frame_waiters.discard((loop, ev))
        return True

    def _render_waiting(self, cam):
        """Big scannable QR inside the live feed until the phone streams (rebuilt at 2 Hz, PIL text)."""
        detail = MB.waiting_detail(cam, self.mobile)
        img, built, last_detail = self._waiting
        now = time.time()
        if img is None or detail != last_detail or now - built > 0.5:
            img = MB.render_waiting_screen(self.mobile, (1280, 720), detail)
            toast = self.toast_text()
            if toast:
                E.draw_toast(img, img.shape[0] / 720.0, toast)
            self._waiting = (img, now, detail)
            if self._publish_jpeg(img):
                self.frames_rendered += 1
                self.render_meter.tick()

    def _telemetry_loop(self):
        proc = _safe(lambda: psutil.Process(os.getpid())) if psutil else None
        if proc:
            _safe(lambda: proc.cpu_percent(None))
        while self.running:
            try:
                now = time.time()
                with self.lock:
                    active = [p for p in self.slot_mgr.players.values() if p.is_active(now)]
                    for p in active:
                        tip = E.coach_suggestions(p, now)[0]
                        if tip != self._last_tip.get(p.slot_id) and now - self._last_tip_time.get(p.slot_id, 0) > 4.0:
                            self._last_tip[p.slot_id] = tip
                            self._last_tip_time[p.slot_id] = now
                            self.coach_feed.appendleft({
                                "t": round(now, 2), "time": datetime.fromtimestamp(now).strftime("%H:%M:%S"),
                                "session_s": round(now - self.session_started_at, 1),
                                "player": f"P{p.slot_id}", "text": tip,
                                "speed_ms": round(p.instant_speed_ms, 2), "stability_pct": round(p.stability_pct, 1)})
                sysinfo = {"cpu_pct": None, "ram_pct": None, "process_mb": None}
                if psutil:
                    sysinfo["cpu_pct"] = _safe(lambda: psutil.cpu_percent(None))
                    sysinfo["ram_pct"] = _safe(lambda: psutil.virtual_memory().percent)
                    if proc:
                        sysinfo["process_mb"] = _safe(lambda: round(proc.memory_info().rss / 1e6, 1))
                self.system = sysinfo
            except Exception as exc:
                self._record_error("telemetry", exc)
            time.sleep(1.0)

    # ------------------------------------------------------------- outputs
    def wait_jpeg(self, last_seq, timeout=1.0):
        with self.jpeg_cond:
            if self.jpeg_seq == last_seq:
                self.jpeg_cond.wait(timeout)
            return self.jpeg_seq, self.jpeg

    def player_dict(self, p, now):
        tracked = p.smoothed_pos_m is not None
        pos = p.smoothed_pos_m if tracked else None
        return {
            "id": f"P{p.slot_id}", "slot": p.slot_id,
            "color": "#%02x%02x%02x" % tuple(reversed(E.SLOT_COLORS[p.slot_id])),
            "active": bool(p.is_active(now)), "tracked": bool(tracked),
            "speed_ms": round(p.instant_speed_ms, 2), "speed_kmh": round(p.instant_speed_ms * 3.6, 1),
            "avg_speed_ms": round(p.avg_speed_ms, 2), "avg_speed_kmh": round(p.avg_speed_ms * 3.6, 1),
            "max_speed_ms": round(p.max_speed_ms, 2), "distance_m": round(p.distance_m, 1),
            "coverage_pct": round(p.coverage_pct, 1), "active_s": round(p.active_seconds_total, 1),
            "active_pct": p.active_pct, "tracked_s": round(p.tracked_seconds_total, 1),
            "stability_pct": round(p.stability_pct, 1),
            "pos_m": [round(float(pos[0]), 2), round(float(pos[1]), 2)] if tracked else None,
            "trail_m": [[round(float(m[0]), 2), round(float(m[1]), 2)] for m in list(p.meter_trail)[-25:]],
            "tips": E.coach_suggestions(p, now) if tracked else [],
        }

    def telemetry(self, include_heatmap=True):
        now = time.time()
        with self.lock:
            players = [self.player_dict(p, now) for p in self.slot_mgr.players.values()]
            heat = None
            if include_heatmap:
                grid = sum(p.heat for p in self.slot_mgr.players.values())
                mx = float(grid.max()) if grid.size else 0.0
                heat = {"rows": int(grid.shape[0]), "cols": int(grid.shape[1]), "cell_m": E.HEAT_CELL_M,
                        "max_s": round(mx, 2),
                        "values": (np.round(grid / mx, 3) if mx > 0 else grid).flatten().tolist()}
        return {"t": round(now, 3), "session_s": round(now - self.session_started_at, 1),
                "players": players, "heatmap": heat, "coach_feed": list(self.coach_feed)[:30],
                "ball": self.ball.state(now), "calibration_review": self.review.status(),
                "score": self.score.state(now), "status": self.status()}

    def status(self):
        cam = self.cam
        spec = self.spec
        return {
            "ok": True, "version": VERSION,
            "uptime_s": round(time.time() - self.started_at, 1),
            "session_s": round(time.time() - self.session_started_at, 1),
            "source": {"label": spec.label if spec else None, "kind": spec.kind if spec else None,
                       "state": self.source_state, "error": self.source_error,
                       "camera_status": cam.status if cam else "none",
                       "seconds_since_frame": None if not cam or cam.frame_id == 0 else round(cam.seconds_since_frame, 2),
                       "reconnects": cam.reconnect_count if cam else 0,
                       "native_fps": round(cam.native_fps, 2) if cam else 0},
            "mobile": self.mobile.status(include_token=False),
            "camera": {"exposure": dict(cam.exposure_info) if cam is not None and cam.kind == "webcam"
                       else {"mode": self.exposure_pref, "label": E.EXPOSURE_LABELS[self.exposure_pref]},
                       "is_webcam": bool(cam is not None and cam.kind == "webcam"), "rotation": self.rotation},
            "failover": {"on_alternate": self.failover_on_alt, "primary": E._redact_url(str(self.failover_primary)),
                         "alternate": self.args.failover_source or "sim / webcam"},
            "ai": self.detector.backend_info,
            "pipeline": {"capture_fps": round(self.capture_meter.rate, 1),
                         "inference_fps": round(self.infer_meter.rate, 1),
                         "render_fps": round(self.render_meter.rate, 1),
                         "render_target_fps": self.args.render_fps,
                         "inference_latency_ms": round(self.infer_latency_ms, 1),
                         "render_ms": round(self.render_ms, 1),
                         "frames_inferred": self.frames_inferred, "frames_rendered": self.frames_rendered,
                         "model": os.path.basename(self.args.model), "detector_ready": self.detector.model is not None,
                         "ball_tracker_fps": round(self.ball_worker.rate, 1)},
            "calibration": {"calibrated": self.review.state == "confirmed", "locked": self.calib_locked,
                            "review_state": self.review.state,
                            "confidence_pct": self.review.info.get("confidence_pct"),
                            "method": self.review.info.get("method") or self.calibration_info.get("method"),
                            "frame_size": [self.canvas_shape[1], self.canvas_shape[0]] if self.canvas_shape else None},
            "clients": {"mjpeg": self.mjpeg_clients, "websocket": self.ws_clients},
            "system": self.system, "errors": dict(self.errors), "recent_errors": list(self.last_errors)[-5:],
            "hud_mode": self.hud_mode,
            "latency": self.latency_stats(),
        }

    def latency_stats(self):
        e2e, age = list(self.e2e_ms), list(self.frame_age_ms)
        cam = self.cam
        return {"e2e_ms_median": round(float(np.median(e2e)), 1) if e2e else None,
                "e2e_ms_p95": round(float(np.percentile(e2e, 95)), 1) if e2e else None,
                "frame_age_at_render_ms": round(float(np.median(age)), 1) if age else None,
                "render_ms": round(self.render_ms, 1), "inference_ms": round(self.infer_latency_ms, 1),
                "capture_fps": round(self.capture_meter.rate, 1), "render_fps": round(self.render_meter.rate, 1),
                "camera_mode": getattr(cam, "mode", "") or None,
                "note": "e2e = frame captured by the camera reader -> annotated JPEG ready to send"}

    def calibration_payload(self):
        with self.lock:
            return {"frame_size": [self.canvas_shape[1], self.canvas_shape[0]] if self.canvas_shape else None,
                    "corners": None if self.corners is None else np.round(self.corners, 1).tolist(),
                    "info": self.review.info or self.calibration_info,
                    "review": self.review_payload(),
                    "references": E.CALIBRATION_REFERENCE_LABELS}

    def reset(self):
        with self.lock:
            self.slot_mgr.reset_players()
            self.ball.shots = self.ball.total_shots = 0
            self.session_started_at = time.time()
            self.coach_feed.clear()

    def shutdown(self):
        self.running = False
        self.ball_worker.stop()
        if self.cam:
            self.cam.release()


class EngineMobileController(MB.MobileController):
    """Phone -> engine: auto-detect, live corner drag, lock (all respect the stage calibration lock) and
    auto-switch to the phone camera the moment a scanned phone streams its first frame."""

    def __init__(self, engine):
        self.e = engine

    def _mobile_active(self):
        return self.e.spec is not None and self.e.spec.kind == "mobile" and self.e.source_state == "ready"

    def calibration_snapshot(self):
        e = self.e
        if not self._mobile_active():
            return {"state": "idle" if e.source_state != "switching" else "detecting", "corners": None,
                    "frame_shape": None, "locked": e.calib_locked, "rotation": e.rotation}
        rv, now = e.review, time.time()
        st = rv.status()
        with rv.lock:
            corners = None if rv.corners is None else rv.corners.copy()
            version = rv.version
        with e.lock:
            players = sum(1 for p in e.slot_mgr.players.values() if p.is_active(now))
            shape = e.canvas_shape
        return dict(st, corners=corners, frame_shape=shape, version=version, players=players,
                    inference_fps=round(e.infer_meter.rate, 1), locked=e.calib_locked, rotation=e.rotation)

    def auto_detect(self):
        if not self._mobile_active():
            raise ValueError("the phone camera is not the active source yet")
        _, info = self.e.auto_calibrate()
        return (f"Court detected: {info.get('confidence_pct') or 0:.0f}% ({info.get('angle') or 'Baseline'}) "
                f"- drag corners to fine-tune, then Lock")

    def preview_corners(self, corners_px):
        e = self.e
        e._check_unlocked()
        with e.lock:
            if e.canvas_shape is None:
                raise ValueError("no phone frame yet")
            if E.compute_homography(corners_px) is None:
                raise ValueError("corners must form a convex court in order TL, TR, BR, BL")
            e.review.frame_shape = tuple(e.canvas_shape)
            e.review.set_manual_corners(corners_px, dict(e.review.info, method="phone_touch"))
            e._sync_review()

    def lock(self, corners_px=None):
        e = self.e
        if corners_px is not None:
            e._check_unlocked()
            e.apply_calibration(corners_px, dict(e.review.info, method="phone_touch",
                                                 phone_corners_px=np.round(corners_px, 1).tolist()))
        elif e.review.state != "confirmed":
            e.confirm_calibration("operator (phone)")
        if not e.calib_locked:
            e.set_lock(True)
        return "Court locked - live analytics started"

    def on_phone_connected(self, meta):
        e = self.e
        if e.args.no_mobile_autoswitch or (e.spec is not None and e.spec.kind == "mobile"):
            return
        log("Phone paired via QR -> switching the live source to the phone camera")
        e.set_source("mobile", background=True)   # never block the phone's WebSocket loop

    def on_phone_rotated(self, meta):
        self.e._reorient_at = time.time() + 0.4      # let frames in flight with the old rotation drain first

    def hud_jpeg(self):
        e = self.e
        return e.jpeg if (e.running and e.jpeg and self._mobile_active()) else None


def court_lines_px(corners, frame_shape=(720, 1280)):
    """Court outline, net, service and centre lines safely projected & horizon-clipped for the web drag editor."""
    H = E.compute_homography(corners) if corners is not None else None
    Hc = CG.invert_homography_c2i(H) if H is not None else None
    if Hc is None:
        return []
    segs = [[(0, 0), (10, 0)], [(10, 0), (10, 20)], [(10, 20), (0, 20)], [(0, 20), (0, 0)],
            [(0, E.NET_Y_M), (10, E.NET_Y_M)], [(0, E.SERVICE_LINE_NEAR_M), (10, E.SERVICE_LINE_NEAR_M)],
            [(0, E.SERVICE_LINE_FAR_M), (10, E.SERVICE_LINE_FAR_M)],
            [(5, E.SERVICE_LINE_NEAR_M), (5, E.SERVICE_LINE_FAR_M)]]
    out = []
    for a, b in segs:
        pts = CG.sample_clipped_segment(Hc, a, b, frame_shape, steps=32)
        out.append([] if pts is None else np.round(pts, 1).tolist())
    return out


# =====================================================================
# Reports
# =====================================================================
def build_report(engine):
    tel = engine.telemetry(include_heatmap=True)
    st = tel["status"]
    players = []
    for p in tel["players"]:
        eff = (p["coverage_pct"] / max(1.0, p["distance_m"])) * 10.0 if p["distance_m"] > 0 else 0.0
        players.append(dict(p, coverage_efficiency=round(eff, 2)))
    heat = tel["heatmap"]
    zones = {}
    if heat and heat["max_s"] > 0:
        g = np.array(heat["values"]).reshape(heat["rows"], heat["cols"])
        half = g.shape[0] // 2
        zones = {"far_half_share_pct": round(100 * g[:half].sum() / max(1e-9, g.sum()), 1),
                 "near_half_share_pct": round(100 * g[half:].sum() / max(1e-9, g.sum()), 1)}
    return {
        "report": "PadelIQ Official Session Report",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "system_version": VERSION,
        "match_duration_s": tel["session_s"],
        "source": st["source"], "calibration": st["calibration"],
        "players": [{k: v for k, v in p.items() if k not in ("trail_m",)} for p in players],
        "court_dominance": zones,
        "ball": {k: tel["ball"][k] for k in ("total_shots", "visibility_pct", "speed_kmh")},
        "coach_feed": tel["coach_feed"],
        "system_health": {"pipeline": st["pipeline"], "system": st["system"], "errors": st["errors"],
                          "uptime_s": st["uptime_s"], "clients": st["clients"]},
    }


CSV_FIELDS = ["id", "active", "speed_ms", "speed_kmh", "avg_speed_ms", "avg_speed_kmh", "max_speed_ms",
              "distance_m", "coverage_pct", "coverage_efficiency", "active_s", "active_pct", "tracked_s",
              "stability_pct"]


def report_csv(rep):
    out = io.StringIO()
    out.write(f"# {rep['report']}\n# generated_at,{rep['generated_at']}\n# match_duration_s,{rep['match_duration_s']}\n")
    out.write(f"# source,{rep['source'].get('label')}\n")
    w = csv.DictWriter(out, fieldnames=CSV_FIELDS, extrasaction="ignore")
    w.writeheader()
    for p in rep["players"]:
        w.writerow(p)
    return out.getvalue()


def _esc(v):
    return (str(v).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def report_html(rep):
    mins, secs = divmod(int(rep["match_duration_s"]), 60)
    rows = "".join(
        f"<tr><td><b style='color:{p['color']}'>{p['id']}</b></td><td>{p['distance_m']}</td>"
        f"<td>{p['avg_speed_ms']} ({p['avg_speed_kmh']} km/h)</td><td>{p['max_speed_ms']}</td>"
        f"<td>{p['coverage_pct']}%</td><td>{p['coverage_efficiency']}</td><td>{p['active_s']} s ({p['active_pct']}%)</td>"
        f"<td>{p['stability_pct']}%</td></tr>" for p in rep["players"])
    feed = "".join(f"<li><span>{_esc(c['time'])}</span> <b>{_esc(c['player'])}</b> {_esc(c['text'])}</li>"
                   for c in rep["coach_feed"][:15]) or "<li>No insights recorded.</li>"
    pl = rep["system_health"]["pipeline"]
    sy = rep["system_health"]["system"] or {}
    dom = rep["court_dominance"] or {}
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>PadelIQ Session Report</title>
<style>
:root{{--ink:#0d1117;--mute:#57606a;--line:#d0d7de;--accent:#0a7d55}}
*{{box-sizing:border-box}} body{{font-family:Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);margin:0;background:#fff}}
.page{{max-width:980px;margin:0 auto;padding:32px 28px}}
header{{display:flex;justify-content:space-between;align-items:flex-end;border-bottom:3px solid var(--ink);padding-bottom:12px}}
h1{{font-size:24px;margin:0}} .sub{{color:var(--mute);font-size:13px}}
.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}}
.kpi{{border:1px solid var(--line);border-radius:8px;padding:12px}} .kpi b{{display:block;font-size:22px}}
.kpi span{{color:var(--mute);font-size:12px;text-transform:uppercase;letter-spacing:.05em}}
table{{width:100%;border-collapse:collapse;font-size:13px;margin-top:8px}}
th,td{{border-bottom:1px solid var(--line);padding:8px 6px;text-align:left}} th{{background:#f6f8fa;font-size:12px;text-transform:uppercase}}
h2{{font-size:16px;margin:26px 0 6px}} ul{{padding-left:18px;font-size:13px}} li span{{color:var(--mute);font-variant-numeric:tabular-nums}}
.foot{{margin-top:28px;color:var(--mute);font-size:11px;border-top:1px solid var(--line);padding-top:10px}}
.btn{{border:1px solid var(--line);background:#f6f8fa;border-radius:6px;padding:6px 12px;cursor:pointer}}
@media print{{.btn{{display:none}} .page{{padding:0}}}}
@media (max-width:640px){{.kpis{{grid-template-columns:repeat(2,1fr)}} table{{font-size:11px}}}}
</style></head><body><div class="page">
<header><div><h1>PadelIQ Official Session Report</h1>
<div class="sub">Executive summary &middot; generated {_esc(rep['generated_at'])} &middot; system v{_esc(rep['system_version'])}</div></div>
<button class="btn" onclick="window.print()">Print / Save PDF</button></header>
<div class="kpis">
<div class="kpi"><span>Match duration</span><b>{mins:02d}:{secs:02d}</b></div>
<div class="kpi"><span>Players tracked</span><b>{sum(1 for p in rep['players'] if p['tracked'])} / 4</b></div>
<div class="kpi"><span>Total distance</span><b>{round(sum(p['distance_m'] for p in rep['players']), 1)} m</b></div>
<div class="kpi"><span>Calibration</span><b style="font-size:15px">{_esc(rep['calibration'].get('method') or 'n/a')}</b></div>
</div>
<h2>Player telemetry</h2>
<table><thead><tr><th>Player</th><th>Distance (m)</th><th>Avg speed (m/s)</th><th>Max (m/s)</th><th>Coverage</th>
<th>Coverage eff.</th><th>Active time</th><th>Stability</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Court dominance</h2>
<p style="font-size:13px">Far half: <b>{dom.get('far_half_share_pct', 'n/a')}%</b> &middot; Near half: <b>{dom.get('near_half_share_pct', 'n/a')}%</b> of tracked player-time.</p>
<h2>AI tactical coach insights (latest)</h2><ul>{feed}</ul>
<h2>System health diagnostics</h2>
<table><tbody>
<tr><th>Source</th><td>{_esc(rep['source'].get('label'))} ({_esc(rep['source'].get('kind'))}), status {_esc(rep['source'].get('camera_status'))}, reconnects {rep['source'].get('reconnects')}</td></tr>
<tr><th>Pipeline</th><td>capture {pl['capture_fps']} fps &middot; AI inference {pl['inference_fps']} fps ({pl['inference_latency_ms']} ms) &middot; render {pl['render_fps']} / {pl['render_target_fps']} fps</td></tr>
<tr><th>Host</th><td>CPU {sy.get('cpu_pct')}% &middot; RAM {sy.get('ram_pct')}% &middot; process {sy.get('process_mb')} MB</td></tr>
<tr><th>Recovered errors</th><td>{_esc(json.dumps(rep['system_health']['errors']))}</td></tr>
</tbody></table>
<div class="foot">Coverage efficiency = coverage % per 10 m travelled. Speeds are physically capped at {E.MAX_PLAYER_SPEED_MS} m/s.
Metrics derive from monocular homography tracking and are estimates.</div>
</div></body></html>"""


# =====================================================================
# FastAPI app
# =====================================================================
def create_app(engine):
    app = FastAPI(title="PadelIQ Enterprise Command Center", version=VERSION, docs_url="/api/docs", redoc_url=None)

    @app.exception_handler(Exception)
    async def all_errors(request: Request, exc: Exception):
        engine._record_error("api", exc)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)

    def err(msg, code=400):
        return JSONResponse({"ok": False, "error": msg}, status_code=code)

    @app.get("/", response_class=HTMLResponse)
    def index():
        path = os.path.join(UI_DIR, "index.html")
        if not os.path.isfile(path):
            return HTMLResponse("<h1>enterprise_ui/index.html missing</h1>", status_code=500)
        return FileResponse(path, media_type="text/html", headers={"Cache-Control": "no-cache"})

    # phone portal on the dashboard port too: /mobile-camera, /ws/mobile-cam, /api/qr.png, /api/mobile/status
    app.include_router(MB.build_mobile_router(engine.mobile, admin_local_only=False))

    @app.get("/video.mjpg")
    async def video():
        """Latest-frame-only MJPEG: an async generator on the event loop (no thread hop per frame). A slow viewer
        simply skips to the newest JPEG - nothing is ever queued per client. (asyncio sets TCP_NODELAY.)"""
        boundary = "padeliqframe"

        async def gen():
            engine.mjpeg_clients += 1
            seq = -1
            waiter = (asyncio.get_running_loop(), asyncio.Event())
            engine.frame_waiters.add(waiter)
            try:
                while engine.running:
                    seq2, jpg = engine.jpeg_seq, engine.jpeg
                    if not jpg or seq2 == seq:
                        waiter[1].clear()
                        if engine.jpeg_seq == seq:
                            try:
                                await asyncio.wait_for(waiter[1].wait(), 1.0)
                            except asyncio.TimeoutError:
                                pass
                        continue
                    seq = seq2
                    yield (b"--" + boundary.encode() + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            finally:
                engine.frame_waiters.discard(waiter)
                engine.mjpeg_clients = max(0, engine.mjpeg_clients - 1)

        return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}",
                                 headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                                          "Pragma": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/status")
    def status():
        return engine.status()

    @app.get("/api/telemetry")
    def telemetry():
        return engine.telemetry()

    @app.get("/api/snapshot")
    def snapshot(clean: int = 0):
        if clean:
            with engine.clean_lock:
                canvas = engine.clean_canvas
            if canvas is None:
                return err("no frame yet", 503)
            ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
            data = buf.tobytes() if ok else b""
        else:
            data = engine.jpeg
        if not data:
            return err("no frame yet", 503)
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/calibration")
    def get_calibration():
        return engine.calibration_payload()

    @app.post("/api/calibration")
    async def post_calibration(request: Request):
        try:
            body = await request.json()
            corners = engine.calibrate_from_points(body.get("points"), str(body.get("reference", "full")))
        except Exception as exc:
            return err(str(exc))
        return {"ok": True, "corners": np.round(corners, 1).tolist()}

    @app.post("/api/calibration/auto")
    async def auto_calibration():
        try:
            corners, info = await asyncio.to_thread(engine.auto_calibrate)
        except Exception as exc:
            return err(str(exc))
        return {"ok": True, "corners": np.round(corners, 1).tolist(), "info": info, "review": engine.review_payload()}

    @app.post("/api/calibration/confirm")
    def confirm_calibration():
        try:
            return {"ok": True, "review": engine.confirm_calibration()}
        except Exception as exc:
            return err(str(exc))

    @app.post("/api/calibration/manual")
    def manual_calibration():
        try:
            return {"ok": True, "review": engine.manual_calibration()}
        except Exception as exc:
            return err(str(exc))

    @app.post("/api/calibration/handles")
    async def calibration_handles(request: Request):
        try:
            body = await request.json()
            court, img = body.get("court_m"), body.get("image_px")
            dragged_idx = body.get("dragged_idx")
            arr_c, arr_i = np.asarray(court, dtype=np.float64), np.asarray(img, dtype=np.float64)
            if ( arr_c.ndim != 2 or arr_c.shape[1] != 2 or arr_c.shape != arr_i.shape
                 or not (4 <= arr_c.shape[0] <= 8) or not np.all(np.isfinite(arr_i)) ):
                raise ValueError("court_m and image_px must be 4 to 6 [x, y] handle pairs")
            return {"ok": True, "review": engine.set_handles(arr_c, arr_i, dragged_idx=dragged_idx)}
        except Exception as exc:
            return err(str(exc))

    @app.post("/api/calibration/lock")
    async def calibration_lock(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            return {"ok": True, "locked": engine.set_lock(body.get("locked", "toggle"))}
        except Exception as exc:
            return err(str(exc))

    @app.post("/api/camera/exposure")
    async def camera_exposure(request: Request):
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            return {"ok": True, **engine.set_exposure(str(body.get("mode", "toggle")))}
        except Exception as exc:
            return err(str(exc))

    @app.get("/api/score")
    def score_get():
        return engine.score.state()

    @app.post("/api/score/point")
    async def score_point(request: Request):
        """{"team": "A"} (+1 Team A = P1/P2) or {"team": "B"} (+1 Team B = P3/P4)."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            ev = engine.score_point(str((body or {}).get("team", "")).upper())
            return {"ok": True, "team": ev["team"], "game": ev["game"], "score": ev["score"]}
        except Exception as exc:
            return err(str(exc))

    @app.post("/api/score/reset")
    def score_reset():
        engine.score.reset()
        engine.toast("Score reset: 0 : 0 | Games 0-0")
        return {"ok": True, "score": engine.score.state()}

    @app.post("/api/score/undo")
    def score_undo():
        ok = engine.score.undo()
        return {"ok": ok, "score": engine.score.state()} if ok else err("nothing to undo")

    @app.post("/api/score/auto")
    async def score_auto(request: Request):
        """{"enabled": true|false} automatic point detection from the ball tracker."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        engine.score.auto = bool((body or {}).get("enabled", not engine.score.auto))
        return {"ok": True, "auto": engine.score.auto}

    @app.post("/api/camera/rotate")
    async def camera_rotate(request: Request):
        """{"step": 90} (default) rotates the live feed by 90 deg clockwise and re-calibrates the court."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        try:
            deg = await asyncio.to_thread(engine.rotate, int((body or {}).get("step", 90)))
            return {"ok": True, "rotation": deg}
        except Exception as exc:
            return err(str(exc))

    @app.post("/api/camera/settings")
    def camera_settings():
        try:
            engine.open_camera_settings()
            return {"ok": True, "note": "driver dialog opened on the server machine (Power Line Frequency 50 Hz)"}
        except Exception as exc:
            return err(str(exc))

    @app.post("/api/failover")
    def failover():
        try:
            return {"ok": True, "switching_to": E._redact_url(engine.failover())}
        except Exception as exc:
            return err(str(exc))

    @app.post("/api/source")
    async def post_source(request: Request):
        try:
            body = await request.json()
        except Exception:
            return err("JSON body required")
        typ = str(body.get("type", "")).lower().strip()
        val = str(body.get("value", "") or "").strip()
        if typ in ("sim", "replay", "clean"):
            source = "sim"
        elif typ == "webcam":
            source = val or "webcam"              # empty -> best colour webcam (Logitech BRIO preferred)
            if source != "webcam" and not source.isdigit():
                return err("webcam index must be a number (or empty for the best camera)")
        elif typ == "esp32":
            if not val:
                return err("ESP32 IP or URL required")
            source = E.normalize_esp32_url(val)
        elif typ == "rtsp":
            if not val.lower().startswith(("rtsp://", "rtsps://")):
                return err("RTSP URL must start with rtsp://")
            source = val
        elif typ == "youtube":
            if not E.is_youtube_url(val):
                return err("not a YouTube URL")
            source = val
        elif typ in ("mobile", "phone", "qr"):
            source = "mobile"
        elif typ == "auto":
            source = "auto"
        elif typ == "url":
            source = val
        else:
            return err("type must be sim|webcam|mobile|esp32|rtsp|youtube|auto|url")
        if typ in ("esp32", "rtsp", "youtube", "url") and E.resolve_source(source) is None:
            return err(f"unusable source: {source}")
        engine.set_source(source, esp32_ip=val if typ == "auto" and val else None)
        out = {"ok": True, "switching_to": E._redact_url(source)}
        if source == "mobile":
            out["mobile"] = engine.mobile.status(include_token=True)
        return out

    @app.post("/api/reset")
    def reset():
        engine.reset()
        return {"ok": True}

    @app.post("/api/hud")
    async def hud(request: Request):
        body = await request.json()
        mode = str(body.get("mode", "compact"))
        if mode not in ("compact", "full"):
            return err("mode must be compact|full")
        engine.hud_mode = mode
        return {"ok": True, "mode": mode}

    def _stamp():
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    @app.get("/api/report.json")
    def report_json():
        rep = build_report(engine)
        return Response(json.dumps(rep, indent=2), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="padeliq_report_{_stamp()}.json"'})

    @app.get("/api/report.csv")
    def report_csv_ep():
        return Response(report_csv(build_report(engine)), media_type="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="padeliq_report_{_stamp()}.csv"'})

    @app.get("/api/report.html", response_class=HTMLResponse)
    def report_html_ep():
        return HTMLResponse(report_html(build_report(engine)))

    @app.websocket("/ws/telemetry")
    async def ws_telemetry(ws: WebSocket):
        await ws.accept()
        engine.ws_clients += 1
        try:
            tick = 0
            while engine.running:
                payload = await asyncio.to_thread(engine.telemetry, tick % 5 == 0)   # heatmap @ 2 Hz
                await ws.send_text(json.dumps(payload))
                tick += 1
                await asyncio.sleep(0.1)
        except (WebSocketDisconnect, RuntimeError, ConnectionError):
            pass
        except Exception as exc:
            engine._record_error("api", exc)
        finally:
            engine.ws_clients = max(0, engine.ws_clients - 1)

    return app


def port_free(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1" if host in ("0.0.0.0", "") else host, port)) != 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="PadelIQ Enterprise Command Center")
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to serve the LAN")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--source", default="sim",
                    help="sim (clean match replay, default) | webcam (best camera, BRIO preferred) | mobile (QR phone camera) | auto | 0 | ESP32 IP | rtsp://... | YouTube URL | file")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=E.DEFAULT_IMGSZ, help="YOLO input size (512 default; 640 = slower)")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "any"])
    ap.add_argument("--render-fps", type=float, default=60.0,
                    help="max annotated stream FPS; the stream follows the camera's native rate up to this cap")
    ap.add_argument("--jpeg-quality", type=int, default=80, help="MJPEG quality (encoded once per frame, shared by all viewers)")
    ap.add_argument("--cam-width", type=int, default=None, help="webcam width (default 1280, MJPG)")
    ap.add_argument("--cam-height", type=int, default=None, help="webcam height (default 720, MJPG)")
    ap.add_argument("--cam-fps", type=int, default=None, help="webcam FPS request (default 60; driver clamps)")
    ap.add_argument("--hud", default="compact", choices=["compact", "full"])
    ap.add_argument("--ai-backend", default="auto", choices=["auto", "openvino", "pytorch"],
                    help="auto = OpenVINO detection + pose with PyTorch fallback")
    ap.add_argument("--pose-stride", type=int, default=2, help="fresh pose every N AI ticks per player (1 = every tick)")
    ap.add_argument("--no-roi", action="store_true", help="always run detection on the full frame")
    ap.add_argument("--exposure", default="auto", choices=list(E.EXPOSURE_MODES),
                    help="webcam exposure at start: auto | sport (fast shutter, max FPS)")
    ap.add_argument("--sport-exposure", type=float, default=E.SPORT_EXPOSURE,
                    help="sport-shutter exposure value (DirectShow log2 s: -5 = 1/32 s)")
    ap.add_argument("--mobile-port", type=int, default=MB.MOBILE_HTTPS_PORT,
                    help="HTTPS port of the QR phone-camera portal (self-signed certs/ generated on first run)")
    ap.add_argument("--lan-ip", default=None, help="LAN IP encoded in the QR code (auto-detected by default)")
    ap.add_argument("--no-https", action="store_true", help="do not start the HTTPS phone portal")
    ap.add_argument("--no-mobile-autoswitch", action="store_true",
                    help="do not switch to the phone camera automatically when a phone starts streaming")
    ap.add_argument("--fiducial-size-cm", type=float, default=MB.DEFAULT_FIDUCIAL_SIZE_CM, help=argparse.SUPPRESS)
    ap.add_argument("--failover-source", default=None,
                    help="F12 / Failover target (default: sim replay when live, best webcam when replaying)")
    ap.add_argument("--hud-min-height", type=int, default=720)
    ap.add_argument("--stall-timeout", type=float, default=E.STALL_TIMEOUT_S)
    ap.add_argument("--open-browser", action="store_true")
    args = ap.parse_args(argv)
    args.render_fps = float(np.clip(args.render_fps, 1, 60))

    if not port_free(args.host, args.port):
        log(f"ERROR: port {args.port} is already in use (is the server already running?).")
        return 2
    E.tune_runtime_threads()          # GIL switch interval 2 ms + torch threads = cores - 2

    engine = PadelIQEngine(args)
    if args.no_https:
        engine.mobile.https_error = "disabled with --no-https"
    else:                                             # phone portal: https://<LAN_IP>:8443/mobile-camera
        MB.start_https_server(MB.build_https_app(engine.mobile), engine.mobile, os.path.join(APP_DIR, "certs"))
    engine.set_source(args.source, background=True)
    app = create_app(engine)
    url = f"http://{'localhost' if args.host in ('0.0.0.0', '127.0.0.1') else args.host}:{args.port}"
    log(f"PadelIQ Enterprise Command Center -> {url}")
    if args.open_browser:
        threading.Timer(2.5, lambda: webbrowser.open(url)).start()
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)
    finally:
        engine.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
