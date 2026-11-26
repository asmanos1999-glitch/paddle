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

try:
    import psutil
except Exception:          # optional
    psutil = None

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse)
import uvicorn

APP_DIR = E.APP_DIR
UI_DIR = os.path.join(APP_DIR, "enterprise_ui")
VERSION = "1.3.0"
IDLE_RENDER_FPS = 10.0      # refresh rate for banners / QR screen when no camera frames arrive
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

        self.detector = E.PersonDetector(args.model, imgsz=args.imgsz, conf=args.conf)
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
        self.last_boxes = np.zeros((0, 4), np.float32)
        self.last_yolo_balls = np.zeros((0, 4), np.float32)
        self.ball_worker = E.BallWorker(lambda: self.cam, self.ball,
                                        lambda: (self.last_boxes, self.last_yolo_balls), args.hud_min_height)

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
    def set_source(self, source, esp32_ip=None, background=True):
        def work():
            with self.switch_lock:
                self.source_state = "switching"
                self.source_error = ""
                try:
                    spec = E.resolve_source(source, esp32_ip=esp32_ip, backend=self.args.backend)
                    if spec is None:
                        raise ValueError(f"unrecognised source {source!r}")
                    new_cam = E.ThreadedLiveCamera(spec, stall_timeout=self.args.stall_timeout,
                                                   cam_width=self.args.cam_width, cam_height=self.args.cam_height,
                                                   cam_fps=self.args.cam_fps)
                except Exception as exc:
                    self.source_state = "error"
                    self.source_error = str(exc)
                    log(f"Source switch failed: {exc}")
                    return
                with self.lock:
                    old = self.cam
                    self.cam, self.spec = new_cam, spec
                    self.canvas_shape = None          # new source -> provisional geometry + fresh auto-calibration
                    self.corners = self.H = None
                    with self.review.lock:
                        self.review.generation += 1
                        self.review.state, self.review.corners, self.review.info = "idle", None, {}
                        self.review.handles_img = self.review.handles_court = None
                    self.scene_cut.reset()
                    self.calib_frames.clear()
                    self.slot_mgr.reset_players()
                    self.session_started_at = time.time()
                    self.coach_feed.clear()
                with self.clean_lock:
                    self.clean_canvas = None
                self.source_state = "ready"
                log(f"Source active: {spec.label}")
                if old is not None:
                    threading.Thread(target=old.release, daemon=True).start()
        if background:
            threading.Thread(target=work, daemon=True).start()
        else:
            work()

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
                    collected.append(E.prepare_canvas(raw, self.args.hud_min_height))
                else:
                    time.sleep(0.012)
            if len(collected) >= 5:
                return collected
        spec = self.spec
        if spec is not None and spec.kind == "file":
            return [E.prepare_canvas(f, self.args.hud_min_height) for f in E.collect_calibration_frames(spec, count=16)]
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
                    H, floor_mask=ai.last_floor_mask, floor_top_y=rv.info.get("floor_top_y", ai.last_floor_top_y)
                )
                if rv.drag_idx is None and not rv.info.get("provisional"):
                    self.slot_mgr.reset_players()
                self.ball.set_geometry(H, self.canvas_shape)
                self.calibration_info = dict(rv.info)
            self.applied_version = rv.version

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
        with self.lock:
            if self.canvas_shape is None:
                raise ValueError("no video frame yet")
            self.review.enter_manual(self.corners, (self.canvas_shape[0], self.canvas_shape[1]))
            self._sync_review()
        return self.review_payload()

    def set_handles(self, court_m, image_px, dragged_idx=None):
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
        """(Re-)run AI court recognition; returns once the result is in review (or failed)."""
        if self.spec is None or self.canvas_shape is None:
            raise ValueError("no active source / frame yet")
        with self.clean_lock:
            seed = None if self.clean_canvas is None else self.clean_canvas.copy()
        with self.lock:
            self._start_review(seed_frame=seed)
        end = time.time() + wait_s
        while time.time() < end and self.review.state == "detecting":
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
                canvas = E.prepare_canvas(frame, self.args.hud_min_height)
                with self.lock:
                    if cam is not self.cam:
                        continue
                    self._ensure_geometry(canvas)
                    if self.review.state == "detecting":
                        if time.time() - self._last_calib_grab > 0.03:
                            self.calib_frames.append(canvas)
                            self._last_calib_grab = time.time()
                        self._sync_review()
                        time.sleep(0.035)
                        continue
                    elif self.review.state in ("review", "confirmed", "failed"):
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
                t0 = time.perf_counter()
                boxes = self.detector(canvas)
                self.last_boxes, self.last_yolo_balls = boxes, self.detector.last_balls
                self.infer_latency_ms = 0.8 * self.infer_latency_ms + 0.2 * (time.perf_counter() - t0) * 1000
                with self.lock:
                    if cam is self.cam:
                        self.slot_mgr.update_detections(
                            boxes, now=ts, keypoints=self.detector.last_keypoints, frame_shape=canvas.shape
                        )
                self.frames_inferred += 1
                self.infer_meter.tick()
            except Exception as exc:
                self._record_error("inference", exc)
                time.sleep(0.05)

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
                    last_cam, last_fid, last_canvas = cam, -1, None
                new = False
                if cam is not None:
                    frame, fid, ts = E.cam_wait_latest(cam, last_fid, idle_dt)
                    if frame is not None and fid != last_fid:
                        if last_fid >= 0:
                            for _ in range(min(10, fid - last_fid)):
                                self.capture_meter.tick()
                        last_fid, last_ts, new = fid, ts, True
                        last_canvas = E.prepare_canvas(frame, self.args.hud_min_height)
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
                if last_canvas is not None:
                    hud = last_canvas.copy()
                else:
                    hud = E.placeholder_frame()
                degraded = cam is None or cam.is_degraded or self.source_state == "switching"
                if self.source_state == "switching":
                    detail = "Switching source ..."
                elif cam is None:
                    detail = "Starting ..."
                elif cam.frame_id == 0:
                    detail = f"Connecting to {self.spec.label if self.spec else ''} {cam.last_error}".strip()
                else:
                    detail = f"No frames for {cam.seconds_since_frame:.1f}s | attempt #{max(1, cam.reconnect_count)}"
                with self.lock:
                    have_geo = last_canvas is not None and self.canvas_shape == last_canvas.shape[:2]
                    if have_geo:
                        self._sync_review()
                    E.render_live_hud(hud, self.slot_mgr if have_geo else None, self.corners if have_geo else None,
                                      self.infer_meter.rate, H=self.H,
                                      source_label=self.spec.label if self.spec else "starting",
                                      reconnecting=degraded, reconnect_detail=detail, hud_mode=self.hud_mode,
                                      ball=self.ball if have_geo else None,
                                      review=self.review if have_geo else None, web=True)
                ok, buf = cv2.imencode(".jpg", hud, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.args.jpeg_quality)])
                if ok:
                    with self.jpeg_cond:
                        self.jpeg = buf.tobytes()
                        self.jpeg_seq += 1
                        self.jpeg_cond.notify_all()
                    self.frames_rendered += 1
                    self.render_meter.tick()
                    if new:
                        self.e2e_ms.append((time.time() - last_ts) * 1000.0)
                self.render_ms = 0.8 * self.render_ms + 0.2 * (time.perf_counter() - t0) * 1000
            except Exception as exc:
                self._record_error("render", exc)
                time.sleep(0.05)

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
                "status": self.status()}

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
            "pipeline": {"capture_fps": round(self.capture_meter.rate, 1),
                         "inference_fps": round(self.infer_meter.rate, 1),
                         "render_fps": round(self.render_meter.rate, 1),
                         "render_target_fps": self.args.render_fps,
                         "inference_latency_ms": round(self.infer_latency_ms, 1),
                         "render_ms": round(self.render_ms, 1),
                         "frames_inferred": self.frames_inferred, "frames_rendered": self.frames_rendered,
                         "model": os.path.basename(self.args.model), "detector_ready": self.detector.model is not None,
                         "ball_tracker_fps": round(self.ball_worker.rate, 1)},
            "calibration": {"calibrated": self.review.state == "confirmed",
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

    @app.get("/video.mjpg")
    async def video():
        """Latest-frame-only MJPEG: an async generator on the event loop (no thread hop per frame). A slow viewer
        simply skips to the newest JPEG - nothing is ever queued per client. (asyncio sets TCP_NODELAY.)"""
        boundary = "padeliqframe"

        async def gen():
            engine.mjpeg_clients += 1
            seq = -1
            try:
                while engine.running:
                    seq2, jpg = engine.jpeg_seq, engine.jpeg
                    if not jpg or seq2 == seq:
                        await asyncio.sleep(0.002)
                        continue
                    seq = seq2
                    yield (b"--" + boundary.encode() + b"\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            finally:
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
        elif typ == "auto":
            source = "auto"
        elif typ == "url":
            source = val
        else:
            return err("type must be sim|webcam|esp32|rtsp|youtube|auto|url")
        if typ in ("esp32", "rtsp", "youtube", "url") and E.resolve_source(source) is None:
            return err(f"unusable source: {source}")
        engine.set_source(source, esp32_ip=val if typ == "auto" and val else None)
        return {"ok": True, "switching_to": E._redact_url(source)}

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
                    help="sim (clean match replay, default) | webcam (best camera, BRIO preferred) | auto | 0 | ESP32 IP | rtsp://... | YouTube URL | file")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--imgsz", type=int, default=E.DEFAULT_IMGSZ, help="YOLO input size (512 default; 640 = slower)")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf", "any"])
    ap.add_argument("--render-fps", type=float, default=60.0,
                    help="max annotated stream FPS; the stream follows the camera's native rate up to this cap")
    ap.add_argument("--jpeg-quality", type=int, default=78)
    ap.add_argument("--cam-width", type=int, default=None, help="webcam width (default 1280, MJPG)")
    ap.add_argument("--cam-height", type=int, default=None, help="webcam height (default 720, MJPG)")
    ap.add_argument("--cam-fps", type=int, default=None, help="webcam FPS request (default 60; driver clamps)")
    ap.add_argument("--hud", default="compact", choices=["compact", "full"])
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
