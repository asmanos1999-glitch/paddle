# -*- coding: utf-8 -*-
"""
PadelIQ True Multi-Angle AI Court Geometry & Keypoint Homography Engine
=======================================================================

Self-contained computer vision & projective geometry module (`CourtGeometryAI`) shared by
both `padeliq_live_stream.py` and `padeliq_enterprise_server.py`:

  Stage A: Player-Seeded Floor Surface Segmentation (Zero Sky / Wall / Curtain Bleeding)
           - Hybrid YOLOv8n full-frame detector (high recall on both near & behind-net far
             players + sports ball) paired with top-down YOLOv8n-pose on 192px player crops
             (4x keypoint resolution on ankles 15, 16 & hips/shoulders).
           - Seeds LAB + HSV floor color distribution from foreground player ankles + painted-line
             flanks and isolates the Primary Foreground Court Floor via barrier-aware connected
             components (never bleeding onto walls, curtains, ceilings, or adjacent courts).

  Stage B: White-Line Top-Hat Extraction & Two-Vanishing-Point Perspective Homography
           - Top-Hat morphological filter (`cv2.MORPH_TOPHAT`) + adaptive ridge thresholding
             strictly inside the validated floor domain (suppresses ceiling glare).
           - Rejects 3D vertical fence posts / glass mullions off-center so only true horizontal
             floor lines enter the Homogeneous S^2 RANSAC vanishing-point solver (`VP_long`, `VP_trans`).
           - Solves true 10.0m x 20.0m Homography H on straight, elevated, low-baseline, and
             30-45 deg oblique/diagonal angles, fused with Pose-Ankle Trajectory Hull refinement.

  Stage C: Fast Convergence (15-25 frames) & Automatic Scene-Cut Detector (`SceneCutDetector`)
           - Monitors HSV histogram Bhattacharyya distance + structural luminance SSIM to
             automatically re-calibrate within ~1.0-1.8s when a video cuts to a new court.

  Stage D: Strict Main-Court Pose-Ankle Player Filter
           - Maps pose ankles (or anthropometric hip-torso ground projection) through H and
             gates against the primary floor mask + perspective scale to reject 100% of
             adjacent-court players, seated spectators, and wall posters.
"""
import os
import math
import time
import threading

import cv2
import numpy as np

COURT_WIDTH_M = 10.0
COURT_LENGTH_M = 20.0
NET_Y_M = 10.0
SERVICE_OFFSET_M = 6.95
SERVICE_FAR_M = NET_Y_M - SERVICE_OFFSET_M     # 3.05 m
SERVICE_NEAR_M = NET_Y_M + SERVICE_OFFSET_M    # 16.95 m

FULL_COURT_CORNERS_M = np.array([
    [0.0, 0.0],
    [COURT_WIDTH_M, 0.0],
    [COURT_WIDTH_M, COURT_LENGTH_M],
    [0.0, COURT_LENGTH_M],
], dtype=np.float32)

HANDLE_LABELS = ("TL", "TR", "BR", "BL", "NET-L", "NET-R")
KCONF = 0.35
DEFAULT_LEG_TORSO_RATIO = 1.68


# =====================================================================
# 1. Hybrid Full-Recall Person + Crop YOLOv8n-Pose Detector
# =====================================================================
class OVYolo:
    """Direct OpenVINO runner for Ultralytics YOLOv8 exports (detect / pose): letterbox -> infer -> NumPy decode ->
    NMS. Skips the Ultralytics predict wrapper, which costs ~8-10 ms of Python / torch overhead per call on CPU.
    Works with static rectangular inputs (e.g. 288x480 = a 16:9 frame without square letterbox padding)."""

    def __init__(self, model_dir, task="detect"):
        import glob
        import openvino as ov
        xmls = sorted(glob.glob(os.path.join(model_dir, "*.xml")))
        if not xmls:
            raise FileNotFoundError(f"no OpenVINO model in {model_dir}")
        self.task = task
        core = ov.Core()
        model = core.read_model(xmls[0])
        if model.input(0).get_partial_shape().is_dynamic:      # dynamic export -> fix to its metadata size
            sz = PoseCourtDetector._openvino_imgsz(model_dir) or 192
            model.reshape({model.input(0).get_any_name(): [1, 3, int(sz), int(sz)]})
        self.compiled = core.compile_model(model, "CPU", {"PERFORMANCE_HINT": "LATENCY"})
        shp = self.compiled.input(0).get_shape()
        self.in_h, self.in_w = int(shp[2]), int(shp[3])
        self.req = self.compiled.create_infer_request()
        self._canvas = np.full((self.in_h, self.in_w, 3), 114, np.uint8)

    def _infer(self, img):
        h, w = img.shape[:2]
        r = min(self.in_w / float(w), self.in_h / float(h))
        nw, nh = max(1, int(round(w * r))), max(1, int(round(h * r)))
        px, py = (self.in_w - nw) // 2, (self.in_h - nh) // 2
        canvas = self._canvas
        canvas[:] = 114
        canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR) \
            if (nw, nh) != (w, h) else img
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 255.0, swapRB=True)
        self.req.infer({0: blob})
        return self.req.get_output_tensor(0).data[0], r, px, py       # (C, N)

    @staticmethod
    def _nms(cxcywh, scores, conf, iou=0.7):
        if len(scores) == 0:
            return np.zeros((0,), int)
        rects = np.stack([cxcywh[:, 0] - cxcywh[:, 2] / 2, cxcywh[:, 1] - cxcywh[:, 3] / 2,
                          cxcywh[:, 2], cxcywh[:, 3]], axis=1)
        idx = cv2.dnn.NMSBoxes(rects.tolist(), scores.astype(float).tolist(), float(conf), float(iou))
        return np.array(idx, int).reshape(-1)

    @staticmethod
    def _to_xyxy(cxcywh, r, px, py):
        xyxy = np.stack([cxcywh[:, 0] - cxcywh[:, 2] / 2, cxcywh[:, 1] - cxcywh[:, 3] / 2,
                         cxcywh[:, 0] + cxcywh[:, 2] / 2, cxcywh[:, 1] + cxcywh[:, 3] / 2], axis=1)
        xyxy[:, [0, 2]] = (xyxy[:, [0, 2]] - px) / r
        xyxy[:, [1, 3]] = (xyxy[:, [1, 3]] - py) / r
        return xyxy.astype(np.float32)

    def detect(self, img, classes, conf):
        """-> xyxy (M,4), cls (M,), conf (M,) in img pixels."""
        out, r, px, py = self._infer(img)
        all_b, all_c, all_s = [], [], []
        for c in classes:
            sc = out[4 + c]
            keep = np.flatnonzero(sc >= conf)
            if keep.size == 0:
                continue
            b = out[:4, keep].T
            k = self._nms(b, sc[keep], conf)
            if k.size:
                all_b.append(self._to_xyxy(b[k], r, px, py))
                all_c.append(np.full(k.size, c, int))
                all_s.append(sc[keep][k].astype(np.float32))
        if not all_b:
            return np.zeros((0, 4), np.float32), np.zeros((0,), int), np.zeros((0,), np.float32)
        return np.vstack(all_b), np.concatenate(all_c), np.concatenate(all_s)

    def pose(self, img, conf):
        """-> xyxy (M,4), keypoints (M,17,3) [x, y, conf], conf (M,) in img pixels."""
        out, r, px, py = self._infer(img)
        sc = out[4]
        keep = np.flatnonzero(sc >= conf)
        if keep.size == 0:
            return np.zeros((0, 4), np.float32), np.zeros((0, 17, 3), np.float32), np.zeros((0,), np.float32)
        b = out[:4, keep].T
        k = self._nms(b, sc[keep], conf)
        kp = out[5:, keep][:, k].T.reshape(-1, 17, 3).astype(np.float32)
        kp[:, :, 0] = (kp[:, :, 0] - px) / r
        kp[:, :, 1] = (kp[:, :, 1] - py) / r
        return self._to_xyxy(b[k], r, px, py), kp, sc[keep][k].astype(np.float32)


class PoseCourtDetector:
    """Combines full-frame YOLOv8n detection (high recall on occluded far-court players +
    COCO sports-ball class 32) with batched YOLOv8n-pose crop inference (4x resolution on
    ankles 15, 16 and hip/shoulder keypoints). Falls back gracefully if one weight is absent."""

    # Pose refresh policy (per player): fresh pose every POSE_STRIDE AI ticks, or sooner when the box jumped.
    POSE_STRIDE = 2
    POSE_JUMP_FRAC = 0.25          # box centre moved > 25% of its height since the last pose -> fresh pose
    KP_EMA_ALPHA = 0.65            # keypoint temporal smoothing (weight of the new measurement)

    def __init__(self, app_dir=None, model_path="yolov8n.pt", imgsz=640, conf=0.26, crop_size=192, backend="auto"):
        self.app_dir = app_dir or os.path.dirname(os.path.abspath(__file__))
        self.imgsz = int(imgsz)
        self.det_imgsz = int(imgsz)
        self.conf = float(conf)
        self.crop_size = int(crop_size)
        self.backend_pref = backend            # auto (OpenVINO first) | openvino | pytorch
        self.det_model = None
        self.pose_model = None
        self.det_backend = "none"
        self.pose_backend = "none"
        self.lock = threading.Lock()
        self.last_keypoints = []
        self.last_balls = np.zeros((0, 4), dtype=np.float32)
        self.roi = None                        # (x1, y1, x2, y2) court ROI for detection, None = full frame
        self.adaptive_pose = True
        self._pose_cache = []                  # [(box (4,), kps (17,3), ticks_since_pose)]
        self.stats = {"pose_crops": 0, "pose_reused": 0}
        self._load_models(model_path)

    def _resolve(self, p):
        if os.path.isabs(p) or os.path.isfile(p):
            return p
        return os.path.join(self.app_dir, p)

    def _load_models(self, model_path):
        try:
            from ultralytics import YOLO
        except Exception:
            return

        # 1) Primary person + ball detector: OpenVINO FP16 export (<name>_openvino_model, fixed input size read
        #    from its metadata) first, then the PyTorch weights.
        det_p = self._resolve(model_path)
        if not os.path.exists(det_p):
            det_p = self._resolve("yolov8n.pt")
        det_cands = []
        if det_p.endswith("_openvino_model") and os.path.isdir(det_p):
            det_cands.append(("openvino", det_p))
        else:
            ov_det = os.path.splitext(det_p)[0] + "_openvino_model"
            if self.backend_pref != "pytorch" and os.path.isdir(ov_det):
                det_cands.append(("openvino", ov_det))
            if self.backend_pref != "openvino" and os.path.isfile(det_p):
                det_cands.append(("pytorch", det_p))
        for name, path in det_cands:
            if "pose" in os.path.basename(path).lower():
                continue
            if name == "openvino":
                try:
                    m = OVYolo(path, "detect")
                    m.detect(np.zeros((m.in_h, m.in_w, 3), np.uint8), [0], 0.5)
                    self.det_model, self.det_backend, self.det_imgsz = m, "openvino", (m.in_h, m.in_w)
                    break
                except Exception:
                    pass
            try:
                m = YOLO(path, task="detect")
                isz = self.imgsz
                if name == "openvino":
                    isz = self._openvino_imgsz(path) or isz
                m.predict(np.zeros((isz, isz, 3), np.uint8), imgsz=isz, verbose=False)
                self.det_model, self.det_backend, self.det_imgsz = m, name, int(isz)
                break
            except Exception:
                continue

        # 2) Pose keypoint model: OpenVINO (`~6ms` per 192px crop on CPU!) first, PyTorch fallback
        pose_cands = []
        ov_dir = os.path.join(self.app_dir, "yolov8n-pose_openvino_model")
        pt_pose = os.path.join(self.app_dir, "yolov8n-pose.pt")
        if os.path.isdir(ov_dir) and self.backend_pref != "pytorch":
            pose_cands.append(("openvino-pose", ov_dir))
        if os.path.isfile(pt_pose):
            pose_cands.append(("pytorch-pose", pt_pose))
        for name, path in pose_cands:
            if name == "openvino-pose":
                try:
                    pm = OVYolo(path, "pose")
                    pm.pose(np.zeros((pm.in_h, pm.in_w, 3), np.uint8), 0.5)
                    self.pose_model, self.pose_backend = pm, name
                    self.crop_size = pm.in_w
                    break
                except Exception:
                    pass
            try:
                pm = YOLO(path, task="pose")
                pm.predict([np.zeros((self.crop_size, self.crop_size, 3), np.uint8)], imgsz=self.crop_size, verbose=False)
                self.pose_model = pm
                self.pose_backend = name
                break
            except Exception:
                continue

    @staticmethod
    def _openvino_imgsz(path):
        """Static input size of an Ultralytics OpenVINO export (metadata.yaml 'imgsz: [h, w]')."""
        try:
            txt = open(os.path.join(path, "metadata.yaml"), encoding="utf-8").read()
            import re
            m = re.search(r"imgsz:\s*\n\s*-\s*(\d+)", txt) or re.search(r"imgsz:\s*\[(\d+)", txt)
            return int(m.group(1)) if m else None
        except Exception:
            return None

    @property
    def ready(self):
        return self.det_model is not None or self.pose_model is not None

    def set_roi(self, roi):
        """Restrict detection to roi = (x1, y1, x2, y2) in frame pixels (None = full frame)."""
        self.roi = None if roi is None else tuple(int(v) for v in roi)

    @staticmethod
    def court_roi(corners_px, frame_shape, margin=0.15, head_margin=0.30):
        """Bounding box of the court polygon + `margin` on the sides / bottom and `head_margin` on top
        (far-baseline players' heads stick out above the court floor)."""
        if corners_px is None:
            return None
        pts = np.asarray(corners_px, dtype=np.float32).reshape(-1, 2)
        if pts.shape[0] < 4 or not np.all(np.isfinite(pts)):
            return None
        h, w = frame_shape[:2]
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        bw, bh = x2 - x1, y2 - y1
        if bw < 32 or bh < 32:
            return None
        x1, x2 = max(0, x1 - margin * bw), min(w, x2 + margin * bw)
        y1, y2 = max(0, y1 - head_margin * bh), min(h, y2 + margin * bh)
        if (x2 - x1) * (y2 - y1) > 0.92 * w * h:      # ROI ~ whole frame: cropping would not save anything
            return None
        return int(x1), int(y1), int(x2), int(y2)

    def _make_crop(self, frame, box):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = box[:4]
        s = max(x2 - x1, y2 - y1) * 1.28
        cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
        X1, Y1 = int(round(cx - 0.5 * s)), int(round(cy - 0.5 * s))
        S = max(16, int(round(s)))
        canvas = np.zeros((S, S, 3), dtype=np.uint8)
        sx1, sy1 = max(0, X1), max(0, Y1)
        sx2, sy2 = min(w, X1 + S), min(h, Y1 + S)
        if sx2 > sx1 and sy2 > sy1:
            canvas[sy1 - Y1:sy2 - Y1, sx1 - X1:sx2 - X1] = frame[sy1:sy2, sx1:sx2]
        return cv2.resize(canvas, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR), (X1, Y1, S / float(self.crop_size))

    @staticmethod
    def _box_iou(a, b):
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1]) + max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) - inter
        return inter / ua if ua > 0 else 0.0

    def detect(self, frame, conf=None, max_pose_crops=5):
        """Returns `(boxes_xyxy (N,4), keypoints_list [N x (17,3) or None], confs (N,))`
        and populates `self.last_balls` with COCO class-32 ball boxes."""
        self.last_balls = np.zeros((0, 4), dtype=np.float32)
        if frame is None or frame.size == 0 or not self.ready:
            self.last_keypoints = []
            return np.zeros((0, 4), np.float32), [], np.zeros((0,), np.float32)

        thr = self.conf if conf is None else float(conf)
        h, w = frame.shape[:2]

        with self.lock:
            # Step 1: High-recall person + sports-ball detection
            if self.det_model is not None:
                roi = self.roi
                ox = oy = 0
                src = frame
                if roi is not None:
                    rx1, ry1, rx2, ry2 = roi
                    rx1, ry1 = max(0, rx1), max(0, ry1)
                    rx2, ry2 = min(w, rx2), min(h, ry2)
                    if rx2 - rx1 >= 64 and ry2 - ry1 >= 64:
                        src, ox, oy = frame[ry1:ry2, rx1:rx2], rx1, ry1
                xyxy = None
                try:
                    if isinstance(self.det_model, OVYolo):
                        xyxy, cls, cf = self.det_model.detect(src, [0, 32], min(thr, 0.15))
                    else:
                        res = self.det_model.predict(
                            src, classes=[0, 32], conf=min(thr, 0.15), imgsz=self.det_imgsz, verbose=False
                        )
                        if res and res[0].boxes is not None:
                            b = res[0].boxes
                            xyxy = b.xyxy.cpu().numpy().astype(np.float32)
                            cls = b.cls.cpu().numpy().astype(int)
                            cf = b.conf.cpu().numpy().astype(np.float32)
                except Exception:
                    xyxy = None
                if xyxy is None or len(xyxy) == 0:
                    self.last_keypoints = []
                    self._pose_cache = []
                    return np.zeros((0, 4), np.float32), [], np.zeros((0,), np.float32)
                if ox or oy:
                    xyxy[:, [0, 2]] += ox
                    xyxy[:, [1, 3]] += oy
                self.last_balls = xyxy[(cls == 32) & (cf >= 0.15)]
                keep = (cls == 0) & (cf >= thr)
                boxes = xyxy[keep]
                confs = cf[keep]
            else:
                # Pose-only fallback
                if isinstance(self.pose_model, OVYolo):
                    try:
                        boxes, kp, confs = self.pose_model.pose(frame, thr)
                    except Exception:
                        boxes, kp, confs = np.zeros((0, 4), np.float32), np.zeros((0, 17, 3), np.float32), np.zeros((0,), np.float32)
                    kps_out = [kp[i] for i in range(len(boxes))]
                    self.last_keypoints = kps_out
                    return boxes, kps_out, confs
                try:
                    res = self.pose_model.predict(frame, conf=thr, imgsz=self.imgsz, verbose=False)
                except Exception:
                    res = None
                if not res or res[0].boxes is None or len(res[0].boxes) == 0:
                    self.last_keypoints = []
                    return np.zeros((0, 4), np.float32), [], np.zeros((0,), np.float32)
                r = res[0]
                boxes = r.boxes.xyxy.cpu().numpy().astype(np.float32)
                confs = r.boxes.conf.cpu().numpy().astype(np.float32)
                kps_out = [None] * len(boxes)
                if r.keypoints is not None and r.keypoints.xy is not None:
                    kxy = r.keypoints.xy.cpu().numpy()
                    kc = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(kxy.shape[:2], np.float32)
                    for i in range(min(len(boxes), len(kxy))):
                        arr = np.zeros((17, 3), dtype=np.float32)
                        arr[:, :2] = kxy[i]
                        arr[:, 2] = kc[i]
                        kps_out[i] = arr
                self.last_keypoints = kps_out
                return boxes, kps_out, confs

            # Step 2: Top-down YOLOv8n-pose on candidate person crops (prioritizing main-court sized boxes)
            kps_out = [None] * len(boxes)
            if self.pose_model is not None and len(boxes) > 0:
                # Prioritize lower/mid-frame non-tiny boxes up to `max_pose_crops`
                cand_order = sorted(
                    [i for i in range(len(boxes)) if (boxes[i, 3] - boxes[i, 1]) >= 0.075 * h and boxes[i, 3] >= 0.25 * h],
                    key=lambda i: -(boxes[i, 3] - boxes[i, 1]) * (0.4 + 0.6 * (boxes[i, 3] / h)),
                )[:max_pose_crops]
                # Adaptive pose stride: reuse the previous skeleton (shifted / scaled with the box) for players
                # whose pose is fresh enough, run the pose net only on the rest.
                prev_cache, new_cache, matched = self._pose_cache, {}, {}
                if self.adaptive_pose and prev_cache:
                    for idx in cand_order:
                        j = self._match_cache(boxes[idx], prev_cache)
                        if j is None:
                            continue
                        pbox, pkps, age = prev_cache[j]
                        bh = max(1.0, boxes[idx, 3] - boxes[idx, 1])
                        jump = np.hypot((boxes[idx, 0] + boxes[idx, 2] - pbox[0] - pbox[2]) * 0.5,
                                        (boxes[idx, 1] + boxes[idx, 3] - pbox[1] - pbox[3]) * 0.5)
                        matched[idx] = j
                        if age + 1 < self.POSE_STRIDE and jump < self.POSE_JUMP_FRAC * bh:
                            kps_out[idx] = self._warp_kps(pkps, pbox, boxes[idx])
                            new_cache[idx] = (boxes[idx].copy(), kps_out[idx], age + 1)
                            self.stats["pose_reused"] += 1
                run_idx = [i for i in cand_order if i not in new_cache]
                if run_idx:
                    crops, metas = [], []
                    for idx in run_idx:
                        cr, meta = self._make_crop(frame, boxes[idx])
                        crops.append(cr)
                        metas.append((idx, meta))
                    self.stats["pose_crops"] += len(run_idx)
                    try:
                        for (pb, pkxy, pkc), (idx, (X1, Y1, sc)) in zip(self._pose_crops(crops), metas):
                            if len(pb) == 0:
                                continue
                            bx = boxes[idx]
                            exp = np.array([(bx[0] - X1) / sc, (bx[1] - Y1) / sc, (bx[2] - X1) / sc, (bx[3] - Y1) / sc])
                            ious = [self._box_iou(exp, q) for q in pb]
                            j = int(np.argmax(ious))
                            if ious[j] < 0.22:
                                continue
                            kxy, kc = pkxy[j], pkc[j]
                            arr = np.zeros((17, 3), dtype=np.float32)
                            arr[:, 0] = kxy[:, 0] * sc + X1
                            arr[:, 1] = kxy[:, 1] * sc + Y1
                            arr[:, 2] = kc
                            if idx in matched:          # temporal EMA against the previous skeleton
                                pbox, pkps, _ = prev_cache[matched[idx]]
                                arr = self._ema_kps(arr, self._warp_kps(pkps, pbox, boxes[idx]))
                            kps_out[idx] = arr
                            new_cache[idx] = (boxes[idx].copy(), arr, 0)
                    except Exception:
                        pass
                self._pose_cache = list(new_cache.values())

        self.last_keypoints = kps_out
        return boxes, kps_out, confs

    def _pose_crops(self, crops):
        """-> [(boxes (M,4), kps_xy (M,17,2), kps_conf (M,17))] per crop, either backend."""
        out = []
        if isinstance(self.pose_model, OVYolo):
            for cr in crops:
                b, kp, _ = self.pose_model.pose(cr, 0.20)
                out.append((b, kp[:, :, :2], kp[:, :, 2]))
            return out
        for pr in self.pose_model.predict(crops, imgsz=self.crop_size, conf=0.20, verbose=False):
            if pr.keypoints is None or pr.boxes is None or len(pr.boxes) == 0:
                out.append((np.zeros((0, 4)), None, None))
                continue
            kxy = pr.keypoints.xy.cpu().numpy()
            kc = pr.keypoints.conf.cpu().numpy() if pr.keypoints.conf is not None else np.ones(kxy.shape[:2], np.float32)
            out.append((pr.boxes.xyxy.cpu().numpy(), kxy, kc))
        return out

    def _match_cache(self, box, cache, min_iou=0.3):
        best, best_iou = None, min_iou
        for j, (pbox, _, _) in enumerate(cache):
            iou = self._box_iou(box, pbox)
            if iou > best_iou:
                best, best_iou = j, iou
        return best

    @staticmethod
    def _warp_kps(kps, old_box, new_box):
        """Move a skeleton with its box (translation + scale): velocity prediction between pose ticks."""
        out = kps.copy()
        ow, oh = max(1.0, old_box[2] - old_box[0]), max(1.0, old_box[3] - old_box[1])
        nw, nh = max(1.0, new_box[2] - new_box[0]), max(1.0, new_box[3] - new_box[1])
        out[:, 0] = new_box[0] + (kps[:, 0] - old_box[0]) * (nw / ow)
        out[:, 1] = new_box[1] + (kps[:, 1] - old_box[1]) * (nh / oh)
        return out

    def _ema_kps(self, new, prev):
        a = self.KP_EMA_ALPHA
        both = (new[:, 2] > KCONF) & (prev[:, 2] > KCONF)
        out = new.copy()
        out[both, :2] = a * new[both, :2] + (1.0 - a) * prev[both, :2]
        return out


def pose_ground_point(box, kps, frame_shape=None, leg_ratio=DEFAULT_LEG_TORSO_RATIO):
    """Compute true floor contact pixel `(fx, fy)` from YOLOv8-pose ankles (15, 16),
    with hip+shoulder anthropometric projection when feet are occluded by the frame bottom
    or net band. Returns `(foot_xy, method, sigma_px, valid_ankles_xy)`."""
    x1, y1, x2, y2 = [float(v) for v in box[:4]]
    bh = max(1.0, y2 - y1)
    h_img = frame_shape[0] if frame_shape is not None else 99999
    feet_clipped_bottom = (y2 >= h_img - 4)

    ankles_xy = []
    if kps is not None and len(kps) >= 17:
        for idx in (15, 16):
            if kps[idx, 2] > KCONF and kps[idx, 0] > 1 and kps[idx, 1] > 1:
                if not (feet_clipped_bottom and kps[idx, 1] >= h_img - 8):
                    ankles_xy.append((float(kps[idx, 0]), float(kps[idx, 1])))
        hips = [kps[idx] for idx in (11, 12) if kps[idx, 2] > KCONF and kps[idx, 1] > 1]
        shs = [kps[idx] for idx in (5, 6) if kps[idx, 2] > KCONF and kps[idx, 1] > 1]

        if ankles_xy:
            fx = float(np.mean([a[0] for a in ankles_xy]))
            fy = float(max(a[1] for a in ankles_xy)) + 0.022 * bh
            return (fx, fy), "ankle", 3.5 + 0.015 * bh, ankles_xy

        if hips and shs:
            hy = float(np.mean([p[1] for p in hips]))
            sy = float(np.mean([p[1] for p in shs]))
            hx = float(np.mean([p[0] for p in hips]))
            torso = hy - sy
            if torso > 6.0:
                fy = hy + leg_ratio * torso
                return (hx, fy), "hip_projection", 6.0 + 0.04 * bh, []

    return (0.5 * (x1 + x2), y2), "bbox", 10.0, []


# =====================================================================
# 2. Projective & Cheirality-Normalized Homography Utilities
# =====================================================================
def normalize_homography_sign(H, ref_pt):
    """Normalize projective matrix `H` so `H @ [ref_x, ref_y, 1]` has homogeneous `w = +1.0`!
    Eliminates sign ambiguity in `cv2.findHomography` / `np.linalg.inv`."""
    if H is None:
        return None
    H = np.asarray(H, dtype=np.float64)
    w_ref = H[2, 0] * ref_pt[0] + H[2, 1] * ref_pt[1] + H[2, 2]
    if abs(w_ref) < 1e-12:
        return H
    return H / w_ref


def compute_homography(corners_px):
    """Image-to-Court homography `H_i2c` mapping 4 full-court image corners -> `10m x 20m`,
    cheirality-normalized so the court interior has positive homogeneous `w > 0`."""
    try:
        pts = np.asarray(corners_px, dtype=np.float32).reshape(4, 2)
        if not np.all(np.isfinite(pts)):
            return None
        if abs(cv2.contourArea(pts)) < 150.0 or not cv2.isContourConvex(pts.reshape(-1, 1, 2)):
            return None
        H, _ = cv2.findHomography(pts, FULL_COURT_CORNERS_M, 0)
        if H is None or not np.all(np.isfinite(H)) or abs(np.linalg.det(H)) < 1e-13:
            return None
        # Normalize sign at midpoint of TL, TR and midpoint of court so interior points have w > 0
        c_pt = 0.5 * (0.5 * (pts[0] + pts[1]) + 0.5 * (pts[2] + pts[3]))
        return normalize_homography_sign(H, c_pt)
    except Exception:
        return None


def invert_homography_c2i(H_i2c):
    """Invert `H_i2c` to `H_c2i` and normalize cheirality at court center `(5.0, 10.0)`."""
    if H_i2c is None:
        return None
    try:
        H_c2i = np.linalg.inv(H_i2c)
        return normalize_homography_sign(H_c2i, (0.5 * COURT_WIDTH_M, NET_Y_M))
    except np.linalg.LinAlgError:
        return None


def project_pts(H, pts_xy, ref_pt=None):
    """Safe perspective projection of `(N, 2)` points; returns `NaN` only behind camera horizon."""
    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    Hm = H.astype(np.float64)
    if ref_pt is not None:
        Hm = normalize_homography_sign(Hm, ref_pt)
    ones = np.ones((len(pts), 1), dtype=np.float64)
    hom = (Hm @ np.hstack([pts, ones]).T).T
    out = np.full((len(pts), 2), np.nan, dtype=np.float32)
    # Check if H was unnormalized (e.g. from raw cv2.findHomography where H[2,2]=1):
    # If all points have negative w, flip sign of H; only reject points crossing w ~ 0!
    if np.all(hom[:, 2] < -1e-5):
        hom = -hom
    valid = hom[:, 2] > 1e-5
    out[valid, 0] = (hom[valid, 0] / hom[valid, 2]).astype(np.float32)
    out[valid, 1] = (hom[valid, 1] / hom[valid, 2]).astype(np.float32)
    return out


def solve_weighted_homography(src_pts, dst_pts, weights=None):
    """Weighted normalized DLT solver for `dst ~ H @ src` (supports 4-6+ handle correspondences)."""
    src = np.asarray(src_pts, dtype=np.float64).reshape(-1, 2)
    dst = np.asarray(dst_pts, dtype=np.float64).reshape(-1, 2)
    n = len(src)
    if n < 4:
        return None
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64).reshape(-1)

    def _norm_matrix(pts, wt):
        wt_sum = max(1e-9, wt.sum())
        mean = (pts * wt[:, None]).sum(axis=0) / wt_sum
        dist = np.sqrt(((pts - mean) ** 2).sum(axis=1))
        mean_d = max(1e-6, (dist * wt).sum() / wt_sum)
        s = math.sqrt(2.0) / mean_d
        T = np.array([[s, 0.0, -s * mean[0]], [0.0, s, -s * mean[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
        hom = np.column_stack([pts, np.ones(len(pts))])
        return T, (T @ hom.T).T

    Ts, sn = _norm_matrix(src, w)
    Td, dn = _norm_matrix(dst, w)
    A = np.zeros((2 * n, 9), dtype=np.float64)
    for i in range(n):
        x, y, _ = sn[i]
        u, v, _ = dn[i]
        wi = math.sqrt(max(1e-6, w[i]))
        A[2 * i] = np.array([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]) * wi
        A[2 * i + 1] = np.array([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]) * wi
    _, _, Vt = np.linalg.svd(A)
    Hn = Vt[-1].reshape(3, 3)
    try:
        H = np.linalg.inv(Td) @ Hn @ Ts
    except np.linalg.LinAlgError:
        return None
    return normalize_homography_sign(H, src.mean(axis=0)) if np.all(np.isfinite(H)) else None


def build_six_handles(H_i2c, frame_shape):
    """Build the 6 named interactive handles `TL, TR, BR, BL, NET-L, NET-R`.
    When full-court near corners `(10, 20)` / `(0, 20)` lie below the frame on a low-baseline
    camera, slide them along the court sidelines (`x=0` / `x=10`) onto the visible screen
    (e.g. near service line `y=16.95m`) so ALL 6 handles are visible and grabbable!"""
    if H_i2c is None or frame_shape is None:
        return None, None, list(HANDLE_LABELS)
    h, w = frame_shape[:2]
    H_c2i = invert_homography_c2i(H_i2c)
    if H_c2i is None:
        return None, None, list(HANDLE_LABELS)

    margin_x, margin_y = 16.0, 16.0

    def _visible(pt):
        return ( np.all(np.isfinite(pt))
                 and margin_x <= pt[0] <= w - margin_x
                 and margin_y <= pt[1] <= h - margin_y )

    def _slide_along_sideline(x_m, y_target, y_anchor):
        p = project_pts(H_c2i, [[x_m, y_target]])[0]
        if _visible(p):
            return [x_m, y_target], p
        if y_target > NET_Y_M:
            p_svc = project_pts(H_c2i, [[x_m, SERVICE_NEAR_M]])[0]
            if _visible(p_svc):
                best_y, best_p = SERVICE_NEAR_M, p_svc
                for ym in np.linspace(SERVICE_NEAR_M, y_target, 18):
                    cand = project_pts(H_c2i, [[x_m, float(ym)]])[0]
                    if _visible(cand):
                        best_y, best_p = float(ym), cand
                return [x_m, round(best_y, 3)], best_p
        best_y, best_p = y_anchor, project_pts(H_c2i, [[x_m, y_anchor]])[0]
        for ym in np.linspace(y_anchor, y_target, 40):
            cand = project_pts(H_c2i, [[x_m, float(ym)]])[0]
            if _visible(cand):
                best_y, best_p = float(ym), cand
        if not _visible(best_p):
            for alpha in np.linspace(0.0, 0.45, 15):
                xc = (1.0 - alpha) * x_m + alpha * 5.0
                for ym in np.linspace(y_anchor, y_target, 25):
                    cand = project_pts(H_c2i, [[float(xc), float(ym)]])[0]
                    if _visible(cand):
                        return [round(float(xc), 3), round(float(ym), 3)], cand
        return [x_m, round(float(best_y), 3)], np.clip(best_p, [margin_x, margin_y], [w - margin_x, h - margin_y])

    tl_c, tl_i = _slide_along_sideline(0.0, 0.0, NET_Y_M)
    tr_c, tr_i = _slide_along_sideline(COURT_WIDTH_M, 0.0, NET_Y_M)
    br_c, br_i = _slide_along_sideline(COURT_WIDTH_M, COURT_LENGTH_M, NET_Y_M)
    bl_c, bl_i = _slide_along_sideline(0.0, COURT_LENGTH_M, NET_Y_M)
    nl_c, nl_i = _slide_along_sideline(0.0, NET_Y_M, 0.5 * (tl_c[1] + bl_c[1]))
    nr_c, nr_i = _slide_along_sideline(COURT_WIDTH_M, NET_Y_M, 0.5 * (tr_c[1] + br_c[1]))

    court_pts = np.array([tl_c, tr_c, br_c, bl_c, nl_c, nr_c], dtype=np.float32)
    img_pts = np.array([tl_i, tr_i, br_i, bl_i, nl_i, nr_i], dtype=np.float32)
    return court_pts, img_pts, list(HANDLE_LABELS)


def homography_from_handles(court_pts, img_pts, dragged_idx=None, frame_shape=None):
    """Compute updated full-court corners + consistent 6-handle positions when any handle
    (`TL, TR, BR, BL, NET-L, NET-R`) is dragged."""
    court = np.asarray(court_pts, dtype=np.float32).reshape(-1, 2)
    img = np.asarray(img_pts, dtype=np.float32).reshape(-1, 2)
    n = len(court)
    if n < 4 or court.shape != img.shape:
        raise ValueError("Need at least 4 matching court/image handles")

    if n == 4:
        H_c2i, _ = cv2.findHomography(court, img, 0)
        H_c2i = normalize_homography_sign(H_c2i, (5.0, 10.0))
    else:
        weights = np.ones(n, dtype=np.float64)
        if dragged_idx is not None and 0 <= int(dragged_idx) < n:
            d_idx = int(dragged_idx)
            if d_idx < 4:
                weights[:4] = 6.0
                weights[d_idx] = 12.0
                weights[4:] = 0.35
            else:
                weights[:4] = 2.0
                weights[4:] = 4.0
                weights[d_idx] = 14.0
        else:
            weights[:4] = 3.0
            weights[4:] = 2.0
        H_c2i = solve_weighted_homography(court, img, weights)
        H_c2i = normalize_homography_sign(H_c2i, (5.0, 10.0))

    if H_c2i is None:
        raise ValueError("Handles do not define a valid perspective court")
    corners = project_pts(H_c2i, FULL_COURT_CORNERS_M)
    if not np.all(np.isfinite(corners)):
        raise ValueError("Projected court corners invalid")
    H_i2c = compute_homography(corners)
    if H_i2c is None:
        raise ValueError("Handles must form a convex perspective court")

    H_c2i_clean = invert_homography_c2i(H_i2c)
    proj_img = project_pts(H_c2i_clean, court)
    if dragged_idx is not None and 0 <= int(dragged_idx) < n and np.all(np.isfinite(proj_img)):
        proj_img[int(dragged_idx)] = img[int(dragged_idx)]
    return corners.astype(np.float32), court.astype(np.float32), proj_img.astype(np.float32)


def sample_clipped_segment(H_c2i, p0_m, p1_m, frame_shape, steps=48):
    """Project a 3D court floor segment `[p0_m, p1_m]` safely into image pixels, normalizing
    cheirality at court center `(5, 10)` so lines never invert across the camera plane."""
    h, w = frame_shape[:2]
    Hn = normalize_homography_sign(H_c2i, (0.5 * COURT_WIDTH_M, NET_Y_M))
    if Hn is None:
        return None
    ts = np.linspace(0.0, 1.0, steps)
    c_pts = np.column_stack([
        p0_m[0] + (p1_m[0] - p0_m[0]) * ts,
        p0_m[1] + (p1_m[1] - p0_m[1]) * ts,
        np.ones_like(ts),
    ])
    hom = (Hn @ c_pts.T).T
    valid = hom[:, 2] > 0.04
    if valid.sum() < 2:
        return None
    px = hom[valid, :2] / hom[valid, 2:3]
    px = np.clip(px, [-3 * w, -3 * h], [4 * w, 4 * h])
    return px.astype(np.float32)


# =====================================================================
# 3. Stage C Helper: Automatic Scene-Cut Detector
# =====================================================================
class SceneCutDetector:
    """Monitors HSV 3D histogram Bhattacharyya distance + downsampled luminance SSIM
    to trigger instant fast re-calibration whenever a compilation stream cuts to a new
    court or camera angle (or transitions out of a title/intro screen)."""

    THUMB_W, THUMB_H = 160, 90

    def __init__(self, cut_hist_thr=0.38, cut_ssim_thr=0.56, confirm_frames=3, cooldown_s=2.5):
        self.cut_hist_thr = float(cut_hist_thr)
        self.cut_ssim_thr = float(cut_ssim_thr)
        self.confirm_frames = int(confirm_frames)
        self.cooldown_s = float(cooldown_s)
        self.reset()

    def reset(self):
        self.ref_hist = None
        self.ref_gray = None
        self.prev_hist = None
        self.prev_gray = None
        self.diff_streak = 0
        self.last_trigger_t = 0.0

    def _features(self, frame):
        small = cv2.resize(frame, (self.THUMB_W, self.THUMB_H), interpolation=cv2.INTER_AREA)
        roi = small[:int(self.THUMB_H * 0.85), :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 6, 6], [0, 180, 0, 256, 0, 256]).flatten()
        s = hist.sum()
        if s > 0:
            hist = (hist / s).astype(np.float32)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return hist, gray

    @staticmethod
    def _ssim(g1, g2):
        c1, c2 = 6.5025, 58.5225
        mu1, mu2 = float(g1.mean()), float(g2.mean())
        v1, v2 = float(g1.var()), float(g2.var())
        cov = float(((g1 - mu1) * (g2 - mu2)).mean())
        num = (2.0 * mu1 * mu2 + c1) * (2.0 * cov + c2)
        den = (mu1 * mu1 + mu2 * mu2 + c1) * (v1 + v2 + c2)
        return num / max(1e-6, den)

    def set_reference(self, frame):
        if frame is None or frame.size == 0:
            return
        self.ref_hist, self.ref_gray = self._features(frame)
        self.prev_hist, self.prev_gray = self.ref_hist, self.ref_gray
        self.diff_streak = 0
        self.last_trigger_t = time.time()

    def update(self, frame, now=None):
        if frame is None or frame.size == 0:
            return False, {}
        now = time.time() if now is None else float(now)
        hist, gray = self._features(frame)
        if self.ref_hist is None:
            self.ref_hist, self.ref_gray = hist, gray
            self.prev_hist, self.prev_gray = hist, gray
            return False, {"hist_diff": 0.0, "ssim": 1.0}

        d_ref = float(cv2.compareHist(self.ref_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        ssim_ref = float(self._ssim(self.ref_gray, gray))
        d_prev = float(cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_BHATTACHARYYA))
        self.prev_hist, self.prev_gray = hist, gray

        changed = (d_ref > self.cut_hist_thr and ssim_ref < self.cut_ssim_thr) or (d_ref > 0.50) or (ssim_ref < 0.36)
        if changed:
            self.diff_streak += 1
        else:
            self.diff_streak = 0
            if d_ref < 0.16 and ssim_ref > 0.82:
                self.ref_hist = 0.97 * self.ref_hist + 0.03 * hist
                self.ref_gray = 0.97 * self.ref_gray + 0.03 * gray

        if self.diff_streak >= self.confirm_frames and (now - self.last_trigger_t) >= self.cooldown_s:
            self.ref_hist, self.ref_gray = hist, gray
            self.diff_streak = 0
            self.last_trigger_t = now
            return True, {"hist_diff": round(d_ref, 3), "step_diff": round(d_prev, 3), "ssim": round(ssim_ref, 3)}

        return False, {"hist_diff": round(d_ref, 3), "ssim": round(ssim_ref, 3)}


# =====================================================================
# 4. Core Engine: CourtGeometryAI (Stages A, B, D)
# =====================================================================
class CourtGeometryAI:
    """True Multi-Angle AI Court Surface Segmentation & Two-VP Homography Solver."""

    def __init__(self, pose_detector=None, app_dir=None):
        self.app_dir = app_dir or os.path.dirname(os.path.abspath(__file__))
        self.pose = pose_detector or PoseCourtDetector(app_dir=self.app_dir)
        self.last_floor_mask = None
        self.last_floor_top_y = 0
        self.last_H_i2c = None
        self.last_H_c2i = None
        self.last_info = {}

    # ------------------------------------------------------------------
    # Stage A: Player-Seeded Floor Surface Segmentation
    # ------------------------------------------------------------------
    def extract_player_ankle_seeds(self, frames, max_pose_frames=5):
        """Run hybrid detector+pose across subsampled frames to gather foreground player ankle
        coordinates `(x, y, weight, bh)` for floor color learning & hull refinement."""
        if not frames:
            return [], []
        h, w = frames[0].shape[:2]
        idxs = np.linspace(0, len(frames) - 1, min(len(frames), max_pose_frames)).astype(int)
        ankle_seeds = []
        all_person_boxes = []

        for idx in idxs:
            f = frames[int(idx)]
            # Full-frame person pass: a stale / wrong court ROI must never blind calibration to the real players
            saved_roi, self.pose.roi = getattr(self.pose, "roi", None), None
            try:
                boxes, kps_list, confs = self.pose.detect(f, conf=0.25, max_pose_crops=6)
            finally:
                self.pose.roi = saved_roi
            frame_boxes = []
            for b, kp, cf in zip(boxes, kps_list, confs):
                x1, y1, x2, y2 = [float(v) for v in b[:4]]
                bh, bw = y2 - y1, x2 - x1
                if bh >= 0.06 * h and bw >= 6.0 and y2 >= 0.16 * h:
                    frame_boxes.append((x1, y1, x2, y2))     # player-guided court validation (incl. far pair)
                if bh < 0.08 * h or bw < 8.0 or y2 < 0.28 * h:
                    continue
                foot_xy, method, _, ank_list = pose_ground_point(b, kp, f.shape)
                fx, fy = foot_xy
                cx_norm = 1.0 - min(0.85, abs(fx - 0.5 * w) / (0.62 * w))
                y_norm = float(np.clip((fy - 0.22 * h) / (0.78 * h), 0.1, 1.0))
                wt = float((bh / h) ** 2 * (0.35 + 0.65 * y_norm) * (0.4 + 0.6 * cx_norm) * float(cf))
                if ank_list:
                    for ax, ay in ank_list:
                        if 4 <= ax < w - 4 and int(0.28 * h) <= ay < h - 4:
                            ankle_seeds.append((ax, ay, wt * 1.4, bh))
                elif 4 <= fx < w - 4 and int(0.30 * h) <= fy < h - 4:
                    ankle_seeds.append((fx, fy, wt * 0.8, bh))
            all_person_boxes.append((int(idx), frame_boxes))
        return ankle_seeds, all_person_boxes

    @staticmethod
    def foreground_occluder_y(bg_bgr):
        band = CourtGeometryAI.foreground_occluder_band(bg_bgr)
        return None if band is None else band[0]

    @staticmethod
    def foreground_occluder_band(bg_bgr):
        """Top row of a wide dark horizontal foreground band in the lower frame (balcony railing, hand rail,
        camera-mount bar) or None. Pixels below it are hidden or belong to a walkway in front of the court, so
        the court floor is treated as ending there (instead of at the image bottom)."""
        h, w = bg_bgr.shape[:2]
        hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
        V, S = hsv[:, :, 2], hsv[:, :, 1]
        # near-black pixels have unstable chroma (sensor noise -> any saturation), so very dark always counts
        dark = ((V < 62) & ((S < 110) | (V < 36))).astype(np.float32)
        frac = cv2.blur(dark.mean(axis=1).reshape(-1, 1), (1, 3)).ravel()
        run_min = max(6, int(round(0.012 * h)))
        y0, y1 = int(0.50 * h), int(0.97 * h)
        run = 0
        for y in range(y0, y1):
            run = run + 1 if frac[y] > 0.62 else 0
            if run >= run_min:
                top = y - run + 1
                if top <= int(0.52 * h):
                    return None
                # band bottom: last row of the railing structure (rails + balusters: >= 35 % dark)
                bot = y
                while bot + 1 < h and frac[bot + 1] > 0.35:
                    bot += 1
                return int(top), int(bot)
        return None

    @staticmethod
    def _tophat_white(gray, scale=1.0):
        ksz = max(9, int(round(19 * scale)) | 1)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        return cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kern)

    def segment_primary_floor(self, bg_bgr, ankle_seeds=None):
        """Segment the Primary Foreground Court Floor in LAB + HSV space, strictly isolated
        from walls, ceilings, sky, curtains, and adjacent background/side courts."""
        h, w = bg_bgr.shape[:2]
        scale = h / 720.0
        occ_y = self.foreground_occluder_y(bg_bgr)
        if occ_y is not None and ankle_seeds:
            # feet hidden behind a foreground railing put "ankle" keypoints ON the railing: never learn its colour
            ankle_seeds = [sd for sd in ankle_seeds if sd[1] < occ_y - 0.01 * h]
        lab = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2LAB)
        hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY)
        L, A, B = lab[:, :, 0].astype(np.float32), lab[:, :, 1].astype(np.float32), lab[:, :, 2].astype(np.float32)
        H_ch, S_ch, V_ch = hsv[:, :, 0].astype(np.float32), hsv[:, :, 1].astype(np.float32), hsv[:, :, 2].astype(np.float32)

        tophat = self._tophat_white(gray, scale)
        not_line_or_dark = (tophat < 22) & (V_ch > 38) & ~((S_ch < 35) & (V_ch > 200))
        # Colour SAMPLES come from a 5x5 median (thin mesh-fence wires removed): feet standing behind a wire-mesh
        # cage must teach the turf colour, not the wire-darkened turf. Classification still uses the full image.
        smp = cv2.medianBlur(bg_bgr, 5)
        lab_s = cv2.cvtColor(smp, cv2.COLOR_BGR2LAB).astype(np.float32)
        hsv_s = cv2.cvtColor(smp, cv2.COLOR_BGR2HSV).astype(np.float32)
        Ls, As, Bs = lab_s[:, :, 0], lab_s[:, :, 1], lab_s[:, :, 2]
        Hs, Ss = hsv_s[:, :, 0], hsv_s[:, :, 1]

        patch_L, patch_A, patch_B, patch_H, patch_S, patch_W = [], [], [], [], [], []
        if ankle_seeds:
            sorted_seeds = sorted(ankle_seeds, key=lambda s: -s[2])[:18]
            rx, ry1, ry2 = max(12, int(28 * scale)), max(4, int(6 * scale)), max(8, int(18 * scale))
            for ax, ay, wt, _ in sorted_seeds:
                ix, iy = int(round(ax)), int(round(ay))
                x0, x1 = max(0, ix - rx), min(w, ix + rx)
                y0, y1 = max(int(0.28 * h), iy - ry1), min(h, iy + ry2)
                sel = not_line_or_dark[y0:y1, x0:x1]
                if np.count_nonzero(sel) < 12:
                    continue
                ys_s, xs_s = np.where(sel)
                if len(ys_s) > 80:
                    pick = np.linspace(0, len(ys_s) - 1, 80).astype(int)
                    ys_s, xs_s = ys_s[pick], xs_s[pick]
                patch_L.append(Ls[y0:y1, x0:x1][ys_s, xs_s])
                patch_A.append(As[y0:y1, x0:x1][ys_s, xs_s])
                patch_B.append(Bs[y0:y1, x0:x1][ys_s, xs_s])
                patch_H.append(Hs[y0:y1, x0:x1][ys_s, xs_s])
                patch_S.append(Ss[y0:y1, x0:x1][ys_s, xs_s])
                patch_W.append(np.full(len(ys_s), wt, dtype=np.float32))

        line_roi = np.zeros((h, w), dtype=bool)
        line_roi[int(0.45 * h):int(0.95 * h), int(0.16 * w):int(0.84 * w)] = True
        strong_line = (tophat > 32) & line_roi
        if np.count_nonzero(strong_line) > 80:
            d_big = cv2.dilate(strong_line.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
            d_sml = cv2.dilate(strong_line.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            flank = d_big & (~d_sml) & not_line_or_dark & line_roi
            fy_idx, fx_idx = np.where(flank)
            if len(fy_idx) > 40:
                pick = np.linspace(0, len(fy_idx) - 1, min(400, len(fy_idx))).astype(int)
                fy_idx, fx_idx = fy_idx[pick], fx_idx[pick]
                med_w = float(np.median(np.concatenate(patch_W))) if patch_W else 1.0
                patch_L.append(Ls[fy_idx, fx_idx])
                patch_A.append(As[fy_idx, fx_idx])
                patch_B.append(Bs[fy_idx, fx_idx])
                patch_H.append(Hs[fy_idx, fx_idx])
                patch_S.append(Ss[fy_idx, fx_idx])
                patch_W.append(np.full(len(fy_idx), med_w * 0.9, dtype=np.float32))

        if not patch_A:
            y0, y1, x0, x1 = int(0.58 * h), int(0.90 * h), int(0.25 * w), int(0.75 * w)
            sel = not_line_or_dark[y0:y1, x0:x1]
            ys_s, xs_s = np.where(sel)
            if len(ys_s) > 0:
                pick = np.linspace(0, len(ys_s) - 1, min(500, len(ys_s))).astype(int)
                ys_s, xs_s = ys_s[pick], xs_s[pick]
                patch_L.append(Ls[y0:y1, x0:x1][ys_s, xs_s])
                patch_A.append(As[y0:y1, x0:x1][ys_s, xs_s])
                patch_B.append(Bs[y0:y1, x0:x1][ys_s, xs_s])
                patch_H.append(Hs[y0:y1, x0:x1][ys_s, xs_s])
                patch_S.append(Ss[y0:y1, x0:x1][ys_s, xs_s])
                patch_W.append(np.ones(len(ys_s), dtype=np.float32))

        sL = np.concatenate(patch_L)
        sA = np.concatenate(patch_A)
        sB = np.concatenate(patch_B)
        sH = np.concatenate(patch_H)
        sS = np.concatenate(patch_S)
        sW = np.concatenate(patch_W)

        ab = np.column_stack([sA, sB])
        if len(ab) > 25:
            step = max(1, len(ab) // 120)
            probes = ab[::step]
            dists = np.linalg.norm(probes[:, None, :] - ab[None, :, :], axis=2)
            scores = ((dists < 9.5).astype(np.float32) * sW[None, :]).sum(axis=1)
            center_ab = probes[int(np.argmax(scores))]
            in_mode = np.linalg.norm(ab - center_ab[None, :], axis=1) < 12.0
            if np.count_nonzero(in_mode) >= 15:
                sL, sA, sB, sH, sS = sL[in_mode], sA[in_mode], sB[in_mode], sH[in_mode], sS[in_mode]

        mu_L = float(np.median(sL))
        mu_A = float(np.median(sA))
        mu_B = float(np.median(sB))
        mu_S = float(np.median(sS))
        ang = sH * (2.0 * math.pi / 180.0)
        mu_H = (math.atan2(float(np.median(np.sin(ang))), float(np.median(np.cos(ang)))) * 180.0 / (2.0 * math.pi)) % 180.0

        sig_A = float(np.clip(1.4826 * np.median(np.abs(sA - mu_A)), 4.2, 9.5 if mu_S < 80 else 10.5))
        sig_B = float(np.clip(1.4826 * np.median(np.abs(sB - mu_B)), 4.5, 8.5 if mu_S < 80 else 12.5))
        sig_L = float(np.clip(1.4826 * np.median(np.abs(sL - mu_L)), 16.0, 38.0))

        d_ab = np.sqrt(((A - mu_A) / sig_A) ** 2 + ((B - mu_B) / sig_B) ** 2)
        dh = np.abs(H_ch - mu_H)
        dh = np.minimum(dh, 180.0 - dh)
        hue_tol = 17.0 if mu_S > 65.0 else 30.0
        min_sat = max(25.0, 0.35 * mu_S) if mu_S > 75.0 else 8.0

        #Require real surface brightness & chroma (prevents night sky / dark black mesh lens flare bleeding!)
        min_L_strict = max(36.0, mu_L - 2.0 * sig_L)
        mesh_compatible = (
            (np.sqrt(((A - (0.62 * mu_A + 0.38 * 128.0)) / (sig_A * 1.2)) ** 2
                     + ((B - (0.62 * mu_B + 0.38 * 128.0)) / (sig_B * 1.2)) ** 2) < 2.25)
            & (dh < hue_tol + 5.0)
            & (S_ch >= 0.6 * min_sat)
            & (L >= max(34.0, mu_L - 2.1 * sig_L))
            & (L <= min(245.0, mu_L + 2.2 * sig_L))
        )

        raw_floor = (
            ((d_ab < 2.65) & (dh < hue_tol) & (S_ch >= min_sat) & (L >= min_L_strict) & (L <= min(248.0, mu_L + 2.4 * sig_L)))
            | mesh_compatible
        )
        raw_floor[:int(0.12 * h), :] = False

        # Wire-mesh cage fences in front of the floor: 1-3 px dark wires over turf. A 5x5 median removes the wires,
        # so the turf behind them classifies as floor again. Only between 0.30 H and a foreground railing (never
        # the background court at the top, never the walkway below a railing).
        mesh_y0 = int(0.30 * h)
        mesh_y1 = occ_y if occ_y is not None else h
        if mesh_y1 - mesh_y0 > 20:
            mb = cv2.medianBlur(bg_bgr[mesh_y0:mesh_y1], 5)
            lab_m = cv2.cvtColor(mb, cv2.COLOR_BGR2LAB).astype(np.float32)
            hsv_m = cv2.cvtColor(mb, cv2.COLOR_BGR2HSV).astype(np.float32)
            Lm, Am, Bm = lab_m[:, :, 0], lab_m[:, :, 1], lab_m[:, :, 2]
            dhm = np.abs(hsv_m[:, :, 0] - mu_H)
            dhm = np.minimum(dhm, 180.0 - dhm)
            d_abm = np.sqrt(((Am - mu_A) / sig_A) ** 2 + ((Bm - mu_B) / sig_B) ** 2)
            behind_mesh = ((d_abm < 2.65) & (dhm < hue_tol) & (hsv_m[:, :, 1] >= 0.8 * min_sat)
                           & (Lm >= max(30.0, min_L_strict - 0.6 * sig_L)) & (Lm <= min(248.0, mu_L + 2.4 * sig_L)))
            raw_floor[mesh_y0:mesh_y1] |= behind_mesh

        # Bridge thin painted white court lines sitting ON the floor, EXCEPT do NOT bridge across a
        # full-width horizontal white net headband tape (keeps adjacent back courts cleanly disconnected!)
        dil_floor = cv2.dilate(raw_floor.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
        floor_plus_lines = (raw_floor | ((tophat > 24) & dil_floor)).astype(np.uint8) * 255

        # Dark structural fence posts / glass frames / baseboards (`V < 52`) act as hard barriers!
        dark_barrier = ((V_ch < 50) & (tophat < 14)).astype(np.uint8) * 255
        floor_plus_lines[dark_barrier > 0] = 0

        k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(5, int(5 * scale) | 1), max(5, int(5 * scale) | 1)))
        k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(7, int(11 * scale) | 1), max(7, int(11 * scale) | 1)))
        cleaned = cv2.morphologyEx(floor_plus_lines, cv2.MORPH_CLOSE, k_close)
        # Horizontal-only bridge reconnects floor corners split by thin vertical net posts / glass mullions
        # (`width <= 12 px`) while having ZERO vertical reach across horizontal back-wall curbs!
        k_horiz_post = cv2.getStructuringElement(cv2.MORPH_RECT, (max(13, int(19 * scale) | 1), 3))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k_horiz_post)
        # Vertical bridge across thin / medium dark HORIZONTAL obstructions in front of the floor (wire-mesh cage
        # wires, structural bars, hand rails): 15x45 then 7x25 closing. Restricted to the lower 70 % of the frame
        # so it can never reach up into a distant background court, and never bridges INTO the rows below a
        # detected foreground railing (the walkway strip under a balcony rail is not the court).
        y_lo = int(0.30 * h)
        y_hi = occ_y if occ_y is not None else h
        if y_hi - y_lo > 40:
            band = cleaned[y_lo:y_hi]
            for kw, kh in ((15, 45), (7, 25)):
                kv = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, int(kw * scale) | 1), max(5, int(kh * scale) | 1)))
                band = cv2.morphologyEx(band, cv2.MORPH_CLOSE, kv)
            cleaned[y_lo:y_hi] = band
        if occ_y is not None:
            cleaned[occ_y:] = 0                       # floor ends at the railing: nothing below it is the court
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, k_open)

        n_cc, labels, stats, centroids = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        if n_cc <= 1:
            mask = cleaned
        else:
            cc_scores = np.zeros(n_cc, dtype=np.float64)
            floor_bot = occ_y if occ_y is not None else h
            for cid in range(1, n_cc):
                area = float(stats[cid, cv2.CC_STAT_AREA])
                if area < 0.015 * h * w:
                    continue
                cy = centroids[cid][1]
                span = stats[cid, cv2.CC_STAT_HEIGHT]
                y_bot = stats[cid, cv2.CC_STAT_TOP] + span
                # A padel court seen from any usable camera spans >= 28 % of the frame height; a thin strip
                # hugging the bottom edge (walkway under a railing, balcony floor) is never the court.
                plausible = span >= 0.28 * h and cy <= 0.82 * h
                score = area * (0.30 + 0.70 * (cy / h)) * (1.4 if y_bot > 0.68 * floor_bot else 0.65)
                if not plausible:
                    score *= 1e-3
                if ankle_seeds:
                    ank_bonus = 0.0
                    for ax, ay, wt, _ in ankle_seeds:
                        ix, iy = int(np.clip(round(ax), 0, w - 1)), int(np.clip(round(ay), 0, h - 1))
                        patch_lbl = labels[max(0, iy - 8):min(h, iy + 14), max(0, ix - 14):min(w, ix + 14)]
                        if np.any(patch_lbl == cid):
                            ank_bonus += wt * (0.25 * h * w)
                    score += ank_bonus
                cc_scores[cid] = score

            best_cid = int(np.argmax(cc_scores))
            mask = ((labels == best_cid).astype(np.uint8)) * 255

            # Merge far half of the SAME court only on elevated/oblique views where `best_cid` does NOT
            # already include the net-mesh zone, and NEVER across a dark fence curb!
            x_left_main = stats[best_cid, cv2.CC_STAT_LEFT]
            x_right_main = x_left_main + stats[best_cid, cv2.CC_STAT_WIDTH]
            y_top_main = stats[best_cid, cv2.CC_STAT_TOP]
            for cid in range(1, n_cc):
                if cid == best_cid or stats[cid, cv2.CC_STAT_AREA] < 0.015 * h * w:
                    continue
                xl = stats[cid, cv2.CC_STAT_LEFT]
                xr = xl + stats[cid, cv2.CC_STAT_WIDTH]
                yt = stats[cid, cv2.CC_STAT_TOP]
                yb = yt + stats[cid, cv2.CC_STAT_HEIGHT]
                gap_y = y_top_main - yb
                overlap_x = max(0, min(x_right_main, xr) - max(x_left_main, xl))
                width_c = max(1, xr - xl)
                if -4 <= gap_y <= int(14 * scale) and overlap_x > 0.78 * width_c and width_c > 0.45 * (x_right_main - x_left_main):
                    band_y0 = max(0, min(yb, y_top_main) - 3)
                    band_y1 = min(h, max(yb, y_top_main) + 4)
                    band_strip = V_ch[band_y0:band_y1, max(x_left_main, xl):min(x_right_main, xr)]
                    # Must be bright white net tape (NOT a dark horizontal back-wall curb!)
                    if band_strip.size > 0 and float(np.percentile(band_strip, 75)) > 170.0 and float(np.percentile(band_strip, 15)) > 55.0:
                        mask = cv2.bitwise_or(mask, ((labels == cid).astype(np.uint8)) * 255)

        # Moderate interior hole fill (never jumps across fences)
        k_fill = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(9, int(13 * scale) | 1), max(9, int(13 * scale) | 1)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_fill)
        n2, lbl2, st2, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n2 > 2:
            c_best = 1 + int(np.argmax(st2[1:, cv2.CC_STAT_AREA]))
            mask = ((lbl2 == c_best).astype(np.uint8)) * 255

        ys_on = np.where(mask > 0)[0]
        top_y = int(ys_on.min()) if len(ys_on) else int(0.35 * h)
        floor_meta = {
            "occluder_y": occ_y,
            "lab_median": [round(mu_L, 1), round(mu_A, 1), round(mu_B, 1)],
            "court_hue": int(round(mu_H)),
            "court_sat": round(mu_S, 1),
            "floor_top_y": top_y,
            "floor_area_pct": round(100.0 * float(np.count_nonzero(mask)) / (h * w), 1),
        }
        return mask, tophat, floor_meta

    # ------------------------------------------------------------------
    # Stage B: White-Line Extraction & Two-Vanishing-Point Homography
    # ------------------------------------------------------------------
    @staticmethod
    def _is_vertical_fence_post(x1, y1, x2, y2, w):
        """3D vertical fence posts / glass mullions away from image center project nearly vertically
        in the image (`|dx/dy| < 0.14`), whereas true horizontal court floor lines away from center
        tilt strongly toward the vanishing point!"""
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dy < 1e-3:
            return False
        xm = 0.5 * (x1 + x2)
        return (dx / dy < 0.14) and (abs(xm - 0.5 * w) > 0.12 * w)

    @classmethod
    def _extract_segments(cls, bg_bgr, floor_mask, tophat, y_bot=None):
        h, w = bg_bgr.shape[:2]
        scale = h / 720.0
        hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
        S_ch, V_ch = hsv[:, :, 1], hsv[:, :, 2]

        k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(9, int(13 * scale) | 1), max(9, int(13 * scale) | 1)))
        floor_domain = cv2.dilate(floor_mask, k_dil)

        vals = tophat[floor_domain > 0]
        p_thr = float(np.percentile(vals, 93.5)) if len(vals) > 200 else 25.0
        thr = float(np.clip(max(18.0, p_thr * 0.72), 18.0, 42.0))

        white_mask = ((tophat >= thr) & (V_ch >= 105) & (S_ch < 135) & (floor_domain > 0)).astype(np.uint8) * 255
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

        raw_white = cv2.HoughLinesP(
            white_mask, 1.0, np.pi / 360.0,
            threshold=max(24, int(32 * scale)),
            minLineLength=max(24, int(0.032 * w)),
            maxLineGap=max(8, int(14 * scale)),
        )
        segs = []
        if raw_white is not None:
            for x1, y1, x2, y2 in raw_white.reshape(-1, 4):
                if cls._is_vertical_fence_post(x1, y1, x2, y2, w):
                    continue
                ln = float(math.hypot(x2 - x1, y2 - y1))
                if ln >= 0.03 * w:
                    segs.append((float(x1), float(y1), float(x2), float(y2), ln, "white"))

        cnts, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            c_main = max(cnts, key=cv2.contourArea)
            edge_img = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(edge_img, [c_main], -1, 255, 2)
            b_pad = 6
            edge_img[:b_pad, :] = 0
            edge_img[h - b_pad:, :] = 0
            edge_img[:, :b_pad] = 0
            edge_img[:, w - b_pad:] = 0
            if y_bot is not None and y_bot < h:          # floor boundary along a foreground railing = occlusion edge
                edge_img[max(0, y_bot - 3 * b_pad):, :] = 0
            raw_bnd = cv2.HoughLinesP(
                edge_img, 1.0, np.pi / 360.0,
                threshold=max(20, int(26 * scale)),
                minLineLength=max(28, int(0.04 * w)),
                maxLineGap=max(10, int(16 * scale)),
            )
            if raw_bnd is not None:
                for x1, y1, x2, y2 in raw_bnd.reshape(-1, 4):
                    if cls._is_vertical_fence_post(x1, y1, x2, y2, w):
                        continue
                    ln = float(math.hypot(x2 - x1, y2 - y1))
                    if ln >= 0.04 * w:
                        segs.append((float(x1), float(y1), float(x2), float(y2), ln, "bound"))

        return segs, white_mask

    @staticmethod
    def _seg_arrays(segs):
        n = len(segs)
        P1 = np.ones((n, 3), dtype=np.float64)
        P2 = np.ones((n, 3), dtype=np.float64)
        L = np.zeros((n, 3), dtype=np.float64)
        W = np.zeros(n, dtype=np.float64)
        LEN = np.zeros(n, dtype=np.float64)
        kinds = []
        for i, (x1, y1, x2, y2, ln, kd) in enumerate(segs):
            P1[i, 0], P1[i, 1] = x1, y1
            P2[i, 0], P2[i, 1] = x2, y2
            li = np.cross(P1[i], P2[i])
            nrm = math.hypot(li[0], li[1]) + 1e-12
            L[i] = li / nrm
            LEN[i] = ln
            W[i] = ln * (2.5 if kd == "white" else 0.75)
            kinds.append(kd)
        M = 0.5 * (P1 + P2)
        return L, P1, P2, M, W, LEN, kinds

    @staticmethod
    def _vp_residuals(vp, P1, P2, M):
        rays = np.cross(M, vp[None, :])
        nrm = np.hypot(rays[:, 0], rays[:, 1])
        valid = nrm > 1e-11
        err = np.full(len(M), 1e6, dtype=np.float64)
        if np.any(valid):
            rn = rays[valid] / nrm[valid, None]
            e1 = np.abs(np.sum(rn * P1[valid], axis=1))
            e2 = np.abs(np.sum(rn * P2[valid], axis=1))
            err[valid] = np.maximum(e1, e2)
        return err

    @staticmethod
    def _refine_vp(L_in, W_in):
        A = L_in * W_in[:, None]
        _, _, Vt = np.linalg.svd(A)
        v = Vt[-1]
        if v[2] < 0:
            v = -v
        return v / (np.linalg.norm(v) + 1e-12)

    @classmethod
    def _ransac_two_vps(cls, segs, frame_shape, tol_px=3.6):
        if len(segs) < 4:
            return None
        h, w = frame_shape[:2]
        cx, cy = 0.5 * w, 0.5 * h
        L, P1, P2, M, W, LEN, kinds = cls._seg_arrays(segs)
        n = len(segs)

        order = np.argsort(-W)[:min(n, 65)]
        cand_vps = []
        for ii in range(len(order)):
            i = order[ii]
            for jj in range(ii + 1, len(order)):
                j = order[jj]
                ang = abs(L[i, 0] * L[j, 1] - L[i, 1] * L[j, 0])
                if ang < 0.035 and abs(L[i, 0] * L[j, 0] + L[i, 1] * L[j, 1]) > 0.999:
                    dx = P2[i, 0] - P1[i, 0]
                    dy = P2[i, 1] - P1[i, 1]
                    v = np.array([dx, dy, 0.0], dtype=np.float64)
                else:
                    v = np.cross(L[i], L[j])
                nv = np.linalg.norm(v)
                if nv < 1e-11:
                    continue
                v /= nv
                if v[2] < 0:
                    v = -v
                if v[2] > 1e-5:
                    vx, vy = v[0] / v[2], v[1] / v[2]
                    if 0.05 * w < vx < 0.95 * w and vy > 0.62 * h:
                        continue
                cand_vps.append(v)

        cand_vps.append(np.array([1.0, 0.0, 0.0], dtype=np.float64))
        if len(cand_vps) > 900:
            rng = np.random.default_rng(42)
            pick = rng.choice(len(cand_vps), size=900, replace=False)
            cand_vps = [cand_vps[k] for k in pick] + [np.array([1.0, 0.0, 0.0])]

        best_v1, best_s1, best_in1 = None, -1.0, None
        for v in cand_vps:
            err = cls._vp_residuals(v, P1, P2, M)
            inl = err < tol_px
            sc = float(W[inl].sum())
            if sc > best_s1:
                best_s1, best_v1, best_in1 = sc, v, inl

        if best_v1 is None or np.count_nonzero(best_in1) < 2:
            return None
        v1 = cls._refine_vp(L[best_in1], W[best_in1])
        if v1[2] > 1e-5:
            vx1, vy1 = v1[0] / v1[2], v1[1] / v1[2]
            if 0.05 * w < vx1 < 0.95 * w and vy1 > 0.64 * h:
                v1 = best_v1
        in1 = cls._vp_residuals(v1, P1, P2, M) < (tol_px * 1.15)

        def _ortho_valid(va, vb):
            hl = np.cross(va, vb)
            nh = math.hypot(hl[0], hl[1])
            if nh < 1e-9:
                return False, None
            if abs(hl[1]) < 1e-6 or abs(hl[0] / hl[1]) > math.tan(math.radians(28.0)):
                return False, None
            ax, ay, aw = va[0] - cx * va[2], va[1] - cy * va[2], va[2]
            bx, by, bw = vb[0] - cx * vb[2], vb[1] - cy * vb[2], vb[2]
            dot_xy = ax * bx + ay * by
            prod_w = aw * bw
            if abs(prod_w) < 1e-8:
                # True 3D ray cosine check using nominal focal length `f_nom = 1.0 * w`
                f_nom = 1.0 * w
                da = np.array([ax, ay, f_nom * aw], dtype=np.float64)
                db = np.array([bx, by, f_nom * bw], dtype=np.float64)
                cos3d = abs(float(np.dot(da, db))) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-12)
                return cos3d < 0.22, f_nom
            f2 = -dot_xy / prod_w
            if f2 <= 0:
                # If va is very close to principal point (cx, cy), allow near-infinity vb
                f_nom = 1.0 * w
                da = np.array([ax, ay, f_nom * aw], dtype=np.float64)
                db = np.array([bx, by, f_nom * bw], dtype=np.float64)
                cos3d = abs(float(np.dot(da, db))) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-12)
                if cos3d < 0.08:
                    return True, f_nom
                return False, None
            f = math.sqrt(f2)
            if not (0.35 * w <= f <= 5.5 * w):
                f_nom = 1.0 * w
                da = np.array([ax, ay, f_nom * aw], dtype=np.float64)
                db = np.array([bx, by, f_nom * bw], dtype=np.float64)
                cos3d = abs(float(np.dot(da, db))) / (np.linalg.norm(da) * np.linalg.norm(db) + 1e-12)
                if cos3d < 0.08:
                    return True, float(np.clip(f, 0.45 * w, 3.0 * w))
                return False, None
            return True, f

        rem_mask = ~in1
        if not np.any(rem_mask):
            return None

        cand_v2 = [v for v in cand_vps if _ortho_valid(v1, v)[0]]
        for slope in np.linspace(-0.18, 0.18, 15):
            v_inf = np.array([1.0, slope, 0.0], dtype=np.float64)
            v_inf /= np.linalg.norm(v_inf)
            if _ortho_valid(v1, v_inf)[0]:
                cand_v2.append(v_inf)

        best_v2, best_s2, best_in2, best_f = None, -1.0, None, 1.1 * w
        for v in cand_v2:
            ok_o, f_est = _ortho_valid(v1, v)
            if not ok_o:
                continue
            err = cls._vp_residuals(v, P1, P2, M)
            inl = rem_mask & (err < tol_px)
            if not np.any(inl):
                continue
            white_bonus = sum(W[k] * 0.35 for k in np.where(inl)[0] if kinds[k] == "white")
            sc = float(W[inl].sum()) + white_bonus
            if sc > best_s2:
                best_s2, best_v2, best_in2, best_f = sc, v, inl, f_est

        if best_v2 is None:
            return None
        if np.count_nonzero(best_in2) >= 2:
            v2_ref = cls._refine_vp(L[best_in2], W[best_in2])
            ok_o, f_ref = _ortho_valid(v1, v2_ref)
            if ok_o:
                best_v2, best_f = v2_ref, f_ref
        in2 = (~in1) & (cls._vp_residuals(best_v2, P1, P2, M) < (tol_px * 1.2))
        return v1, in1, best_v2, in2, best_f

    @staticmethod
    def _cluster_pencil(vp, segs_sub, ref_pt, merge_px=11.0):
        if not segs_sub:
            return []
        rx, ry = ref_pt

        def _ray_from_pt(mx, my):
            ray = np.cross(np.array([mx, my, 1.0], np.float64), vp)
            nrm = math.hypot(ray[0], ray[1]) + 1e-12
            return ray / nrm

        ref_ray = _ray_from_pt(rx, ry)
        ref_nrm = np.array([ref_ray[0], ref_ray[1]], dtype=np.float64)

        items = []
        for sg in segs_sub:
            x1, y1, x2, y2, ln, kd = sg
            mx, my = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            ray = _ray_from_pt(mx, my)
            if ray[0] * ref_nrm[0] + ray[1] * ref_nrm[1] < 0:
                ray = -ray
            s_val = -(ray[0] * rx + ray[1] * ry + ray[2])
            items.append((s_val, ray, ln, kd, sg))

        items.sort(key=lambda t: t[0])
        clusters = []
        for s_val, ray, ln, kd, sg in items:
            placed = False
            for cl in clusters:
                mx, my = 0.5 * (sg[0] + sg[2]), 0.5 * (sg[1] + sg[3])
                d_px = abs(cl["ray"][0] * mx + cl["ray"][1] * my + cl["ray"][2])
                if d_px <= merge_px or abs(s_val - cl["s"]) <= merge_px * 0.85:
                    w_old = cl["tot_w"]
                    w_add = ln * (2.5 if kd == "white" else 0.8)
                    r_new = w_old * cl["ray"] + w_add * ray
                    r_new /= (math.hypot(r_new[0], r_new[1]) + 1e-12)
                    if r_new[0] * ref_nrm[0] + r_new[1] * ref_nrm[1] < 0:
                        r_new = -r_new
                    cl["ray"] = r_new
                    cl["s"] = -(r_new[0] * rx + r_new[1] * ry + r_new[2])
                    cl["tot_w"] += w_add
                    cl["tot_len"] += ln
                    if kd == "white":
                        cl["white_len"] += ln
                    cl["segs"].append(sg)
                    placed = True
                    break
            if not placed:
                w_add = ln * (2.5 if kd == "white" else 0.8)
                clusters.append({
                    "ray": ray,
                    "s": s_val,
                    "tot_w": w_add,
                    "tot_len": ln,
                    "white_len": ln if kd == "white" else 0.0,
                    "segs": [sg],
                })
        clusters.sort(key=lambda c: c["s"])
        return clusters

    @staticmethod
    def _metric_rectify_H(vp_x, vp_y, focal_px, frame_shape):
        h, w = frame_shape[:2]
        K = np.array([[focal_px, 0.0, 0.5 * w], [0.0, focal_px, 0.5 * h], [0.0, 0.0, 1.0]], dtype=np.float64)
        Kinv = np.linalg.inv(K)
        r1 = Kinv @ vp_x
        r1 /= (np.linalg.norm(r1) + 1e-12)
        r2 = Kinv @ vp_y
        r2 = r2 - np.dot(r2, r1) * r1
        r2 /= (np.linalg.norm(r2) + 1e-12)
        if (K @ r1)[0] < 0:
            r1 = -r1
        if (K @ r2)[1] < 0:
            r2 = -r2
        r3 = np.cross(r1, r2)
        if r3[2] < 0:
            r3 = -r3
        R_rect = np.vstack([r1, r2, r3])
        return K @ R_rect @ Kinv

    @staticmethod
    def _detect_net_features(bg_bgr, floor_mask, vp_trans, floor_top_y, y_svc_limit=None):
        """Detect the continuous white net headland tape (`band_ray`) and net floor projection
        (`net_floor_ray`). Distinguishes the real continuous net tape from dotted sponsor banners
        higher up on back walls!"""
        h, w = bg_bgr.shape[:2]
        hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
        S_ch, V_ch = hsv[:, :, 1], hsv[:, :, 2]

        xs_floor = np.where(floor_mask.any(axis=0))[0]
        if len(xs_floor) < 40:
            return None
        x_lo, x_hi = int(xs_floor.min()), int(xs_floor.max())
        span_w = max(1, x_hi - x_lo)

        white_tape = ((V_ch > 172) & (S_ch < 65)).astype(np.uint8) * 255
        # Elevated cameras (balcony) see the net tape as high as ~0.15 H: follow the floor top, not a fixed row
        y_min_s = max(int(0.10 * h), min(int(0.22 * h), floor_top_y + int(0.04 * h)), floor_top_y - int(0.06 * h))
        y_max_s = int(y_svc_limit - 0.10 * h) if y_svc_limit is not None else int(0.68 * h)
        y_max_s = min(y_max_s, h - 35)
        if y_max_s - y_min_s < 25:
            return None

        roi = np.zeros_like(white_tape)
        roi[y_min_s:y_max_s, x_lo:x_hi] = white_tape[y_min_s:y_max_s, x_lo:x_hi]
        lines = cv2.HoughLinesP(roi, 1.0, np.pi / 360.0, threshold=55, minLineLength=int(0.16 * span_w), maxLineGap=14)
        if lines is None:
            return None

        segs = [(float(x1), float(y1), float(x2), float(y2), float(math.hypot(x2 - x1, y2 - y1)), "white")
                for x1, y1, x2, y2 in lines.reshape(-1, 4)]
        L, P1, P2, M, W, _, _ = CourtGeometryAI._seg_arrays(segs)
        err = CourtGeometryAI._vp_residuals(vp_trans, P1, P2, M)
        inl = err < 5.5
        if not np.any(inl):
            return None

        ref_pt = (0.5 * (x_lo + x_hi), 0.5 * (y_min_s + y_max_s))
        cls = CourtGeometryAI._cluster_pencil(vp_trans, [segs[i] for i in np.where(inl)[0]], ref_pt, merge_px=8.0)
        if not cls:
            return None

        # Evaluate each candidate band by CONTINUOUS bright-white density across the middle 50% of court width
        # (real net tape has >70% continuous white pixels; dotted sponsor logos have <45% and sit higher up!)
        xa_c = int(x_lo + 0.22 * span_w)
        xb_c = int(x_hi - 0.22 * span_w)
        xs_chk = np.linspace(xa_c, xb_c, 80).astype(int)
        scored_bands = []
        for c in cls:
            ray = c["ray"]
            if abs(ray[1]) < 0.35 or c["tot_len"] < 0.25 * span_w:
                continue
            ys_chk = np.clip(np.round(-(ray[0] * xs_chk + ray[2]) / ray[1]).astype(int), 2, h - 3)
            # Max white mask in a +-2 px vertical strip around the ray
            strip_hit = np.mean([np.any(white_tape[y - 2:y + 3, x] > 0) for x, y in zip(xs_chk, ys_chk)])
            if strip_hit < 0.52:
                continue
            y_mid = float(-(ray[0] * ref_pt[0] + ray[2]) / ray[1])
            # Prefer continuous full-width tape AND lower y (closer to main court vs distant back-wall banner)
            b_score = strip_hit * 1.5 + (c["tot_len"] / span_w) + 0.35 * (y_mid / h)
            scored_bands.append((b_score, y_mid, c))

        if not scored_bands:
            return None
        _, y_band_mid, best_band = max(scored_bands, key=lambda t: t[0])
        band_ray = best_band["ray"]

        def _y_on_ray(ray, x):
            return -(ray[0] * x + ray[2]) / ray[1]

        max_drop = min(int(0.20 * h), max(25, (int(y_svc_limit - 0.05 * h) if y_svc_limit else h - 20) - int(y_band_mid)))
        if max_drop < 16:
            return None

        offsets = np.arange(8, max_drop)
        prof = np.zeros(len(offsets), dtype=np.float64)
        for idx_o, dy in enumerate(offsets):
            ys_p = np.clip(np.round(np.array([_y_on_ray(band_ray, x) + dy for x in xs_chk])).astype(int), 0, h - 1)
            prof[idx_o] = float(np.median(V_ch[ys_p, xs_chk]))

        if len(prof) < 8:
            return None
        prof_sm = np.convolve(prof, np.ones(3) / 3.0, mode="same")
        step = prof_sm[4:] - prof_sm[:-4]
        if len(step) < 4:
            return None
        best_k = int(np.argmax(step[1:-1])) + 1
        step_strength = float(step[best_k])
        dy_net = float(offsets[best_k + 2])

        pt_net = np.array([ref_pt[0], y_band_mid + dy_net, 1.0], dtype=np.float64)
        net_ray = np.cross(pt_net, vp_trans)
        net_ray /= (math.hypot(net_ray[0], net_ray[1]) + 1e-12)
        return {
            "band_ray": band_ray,
            "band_y_mid": float(y_band_mid),
            "net_floor_ray": net_ray,
            "net_floor_y_mid": float(y_band_mid + dy_net),
            "step_strength": step_strength,
            "band_len": best_band["tot_len"],
        }

    @staticmethod
    def _intersect_rays(r1, r2):
        p = np.cross(r1, r2)
        if abs(p[2]) < 1e-9:
            return None
        return np.array([p[0] / p[2], p[1] / p[2]], dtype=np.float64)

    def _solve_full_quad_candidate(self, bg_bgr, floor_mask, tophat, vp_long, cl_long, vp_trans, cl_trans, focal_px, floor_top_y, ankle_seeds):
        h, w = bg_bgr.shape[:2]
        if len(cl_long) < 2 or len(cl_trans) < 2:
            return None

        ys_m, xs_m = np.where(floor_mask > 0)
        if len(xs_m) < 200:
            return None
        span_diag = math.hypot(xs_m.max() - xs_m.min(), ys_m.max() - ys_m.min())
        min_l = 0.08 * span_diag

        strong_long = [c for c in cl_long if c["white_len"] >= 0.055 * span_diag or c["tot_len"] >= min_l]
        strong_trans = [c for c in cl_trans if c["white_len"] >= 0.055 * span_diag or c["tot_len"] >= min_l]
        if len(strong_long) < 2 or len(strong_trans) < 2:
            return None

        # Sort longitudinal clusters strictly Left -> Right and transverse clusters strictly Top (Far) -> Bottom (Near)
        cx_m, cy_m = float(np.mean(xs_m)), float(np.mean(ys_m))

        def _x_at_mid(c):
            r = c["ray"]
            if abs(r[0]) > 1e-4:
                return -(r[1] * cy_m + r[2]) / r[0]
            return float(np.mean([0.5 * (sg[0] + sg[2]) for sg in c["segs"]]))

        def _y_at_mid(c):
            r = c["ray"]
            if abs(r[1]) > 1e-4:
                return -(r[0] * cx_m + r[2]) / r[1]
            return float(np.mean([0.5 * (sg[1] + sg[3]) for sg in c["segs"]]))

        strong_long.sort(key=_x_at_mid)
        strong_trans.sort(key=_y_at_mid)

        white_long = [c for c in strong_long if c["white_len"] >= 0.055 * span_diag]
        white_trans = [c for c in strong_trans if c["white_len"] >= 0.038 * span_diag]

        cand_L, cand_R = [], []
        for c in white_long[:2]:
            cand_L.append(c)
        for c in white_long[-2:]:
            if not any(c is x for x in cand_R):
                cand_R.append(c)
        if not any(strong_long[0] is x for x in cand_L):
            cand_L.append(strong_long[0])
        if not any(strong_long[-1] is x for x in cand_R):
            cand_R.append(strong_long[-1])

        cand_F, cand_N = [], []
        for c in white_trans[:2]:
            cand_F.append(c)
        for c in white_trans[-2:]:
            if not any(c is x for x in cand_N):
                cand_N.append(c)
        if not any(strong_trans[0] is x for x in cand_F):
            cand_F.append(strong_trans[0])
        if not any(strong_trans[-1] is x for x in cand_N):
            cand_N.append(strong_trans[-1])

        # Detect net headband direction if present: true transverse axis (`TL->TR`) MUST align with the net!
        net_info = self._detect_net_features(bg_bgr, floor_mask, vp_trans, floor_top_y)

        best = None
        for l_left in cand_L:
            for l_right in cand_R:
                if l_left is l_right:
                    continue
                for t_far in cand_F:
                    for t_near in cand_N:
                        if t_far is t_near:
                            continue
                        tl = self._intersect_rays(l_left["ray"], t_far["ray"])
                        tr = self._intersect_rays(l_right["ray"], t_far["ray"])
                        br = self._intersect_rays(l_right["ray"], t_near["ray"])
                        bl = self._intersect_rays(l_left["ray"], t_near["ray"])
                        if any(p is None for p in (tl, tr, br, bl)):
                            continue
                        if tl[0] > tr[0] and bl[0] > br[0]:
                            tl, tr = tr, tl
                            bl, br = br, bl
                        # Strict non-rotated corner ordering: BOTH far corners must sit above their corresponding
                        # near corners, and TL/BL must sit left of TR/BR (prevents 90-degree cyclic axis swap!)
                        if not (tl[1] < bl[1] - 20.0 and tr[1] < br[1] - 20.0):
                            continue
                        if not (tl[0] < tr[0] - 25.0 and bl[0] < br[0] - 25.0):
                            continue
                        if max(tl[1], tr[1]) >= max(bl[1], br[1]) - 30.0:
                            continue
                        corners = np.array([tl, tr, br, bl], dtype=np.float32)
                        H_i2c = compute_homography(corners)
                        if H_i2c is None:
                            continue
                        if min(tl[1], tr[1]) < floor_top_y - 0.035 * h:
                            continue
                        in_view = sum(
                            (-0.28 * w <= p[0] <= 1.28 * w and -0.08 * h <= p[1] <= 1.25 * h)
                            for p in corners
                        )
                        if in_view < 3:
                            continue
                        H_c2i = invert_homography_c2i(H_i2c)
                        if H_c2i is None:
                            continue
                        sc, meta = self._score_homography(
                            bg_bgr, floor_mask, tophat, H_c2i, floor_top_y, ankle_seeds, mode="full_quad"
                        )
                        # A true full-court quad must explain at least 82% of the segmented primary floor!
                        # (Prevents locking onto a half-court `[sideline -> center line]` or service box on low-baseline views)
                        if meta.get("surface_recall_pct", 0.0) < 82.0:
                            continue
                        w_sup = (
                            (l_left["white_len"] > 0.05 * span_diag)
                            + (l_right["white_len"] > 0.05 * span_diag)
                            + (t_far["white_len"] > 0.035 * span_diag)
                            + (t_near["white_len"] > 0.05 * span_diag)
                        )
                        sc += 0.035 * w_sup
                        if net_info is not None:
                            sc += 0.08
                        if best is None or sc > best[0]:
                            best = (sc, corners, meta)
        return best

    def _solve_low_baseline_candidate(self, bg_bgr, floor_mask, tophat, vp_long, cl_long, vp_trans, cl_trans, floor_top_y, ankle_seeds):
        h, w = bg_bgr.shape[:2]
        y_bot = int(getattr(self, "_y_bot", h) or h)    # image bottom, or the top of a foreground railing
        bot_cov = float(np.mean(floor_mask[max(0, y_bot - 14):max(1, y_bot - 4), :] > 0))
        if bot_cov < 0.42:
            return None

        # Near service line (`Y = 16.95m`) first so net search stays strictly above it!
        svc_cands = []
        for cl in cl_trans:
            if cl["white_len"] < 0.22 * w or abs(cl["ray"][1]) < 0.35:
                continue
            y_mid = -(cl["ray"][0] * (0.5 * w) + cl["ray"][2]) / cl["ray"][1]
            if 0.54 * h * (y_bot / h) ** 0.5 < y_mid < y_bot - 0.015 * h:
                svc_cands.append((cl["white_len"], y_mid, cl))
        if not svc_cands:
            return None
        _, y_svc_mid, t_svc = max(svc_cands, key=lambda t: t[0])

        net_info = self._detect_net_features(bg_bgr, floor_mask, vp_trans, floor_top_y, y_svc_limit=y_svc_mid)
        if net_info is None or net_info["step_strength"] < 7.0:
            return None
        y_net_mid = net_info["net_floor_y_mid"]
        if y_svc_mid - y_net_mid < 0.11 * h:
            return None

        # Center service line (`X = 5.0m`)
        center_cl = None
        best_cw = 0.0
        for cl in cl_long:
            if cl["white_len"] < 0.08 * h or abs(cl["ray"][0]) < 0.25:
                continue
            x_at_svc = -(cl["ray"][1] * y_svc_mid + cl["ray"][2]) / cl["ray"][0]
            if 0.25 * w < x_at_svc < 0.75 * w and cl["white_len"] > best_cw:
                best_cw = cl["white_len"]
                center_cl = cl

        x_center_net = 0.5 * w
        if center_cl is not None and abs(center_cl["ray"][0]) > 1e-6:
            x_center_net = -(center_cl["ray"][1] * y_net_mid + center_cl["ray"][2]) / center_cl["ray"][0]

        # Collect left & right outer court lines (painted sidelines OR side-wall floor boundaries)
        left_segs, right_segs = [], []
        for cl in cl_long:
            if abs(cl["ray"][0]) < 0.12:
                continue
            # Reject steep fence-post-like clusters off-center (`|dy/dx| > 2.2` when real sidelines have `|dy/dx| ~ 0.3..1.2`)
            slope_abs = abs(cl["ray"][0] / max(1e-6, abs(cl["ray"][1])))
            if slope_abs > 1.8:
                continue
            xn = -(cl["ray"][1] * y_net_mid + cl["ray"][2]) / cl["ray"][0]
            wt = cl["white_len"] * 3.0 + cl["tot_len"]
            if -0.06 * w <= xn <= x_center_net - 0.16 * w:
                left_segs.append((wt, cl["ray"], xn))
            elif x_center_net + 0.16 * w <= xn <= 1.06 * w:
                right_segs.append((wt, cl["ray"], xn))

        # Direct side-wall / floor-edge search between `band_y` and `y_svc_mid`
        hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
        ra = max(0, int(net_info["band_y_mid"] + 4))
        rb = min(y_bot - 4, int(y_svc_mid - 10))
        if rb - ra > 16:
            for ch in (hsv[:, :, 1], hsv[:, :, 2], floor_mask):
                edges = cv2.Canny(cv2.GaussianBlur(ch, (5, 5), 0), 18, 55)
                lines = cv2.HoughLinesP(edges[ra:rb, :], 1.0, np.pi / 720.0, 22, minLineLength=int(0.04 * w), maxLineGap=12)
                if lines is None:
                    continue
                for x1_, y1_, x2_, y2_ in lines.reshape(-1, 4):
                    y1_ += ra
                    y2_ += ra
                    if x2_ == x1_:
                        continue
                    slope = (y2_ - y1_) / float(x2_ - x1_)
                    if not (0.14 < abs(slope) < 1.55):
                        continue
                    ln = float(math.hypot(x2_ - x1_, y2_ - y1_))
                    p1h = np.array([float(x1_), float(y1_), 1.0])
                    p2h = np.array([float(x2_), float(y2_), 1.0])
                    ray_seg = np.cross(p1h, p2h)
                    ray_seg /= (math.hypot(ray_seg[0], ray_seg[1]) + 1e-12)
                    xn = -(ray_seg[1] * y_net_mid + ray_seg[2]) / ray_seg[0]
                    if slope < 0 and 0.02 * w <= xn <= x_center_net - 0.16 * w:
                        left_segs.append((ln, ray_seg, xn))
                    elif slope > 0 and x_center_net + 0.16 * w <= xn <= 0.98 * w:
                        right_segs.append((ln, ray_seg, xn))

        if not left_segs or not right_segs:
            return None
        left_segs.sort(key=lambda t: -t[0])
        right_segs.sort(key=lambda t: -t[0])

        best_pair = None
        for wl, rl, xnl in left_segs[:22]:
            for wr, rr, xnr in right_segs[:22]:
                vp_lr = self._intersect_rays(rl, rr)
                if vp_lr is None or vp_lr[1] >= net_info["band_y_mid"] - 10.0 or vp_lr[1] < -0.5 * h:
                    continue
                # Vanishing point of left/right sidelines MUST align closely with the center service line!
                if center_cl is not None:
                    vp_c_err = abs(center_cl["ray"][0] * vp_lr[0] + center_cl["ray"][1] * vp_lr[1] + center_cl["ray"][2]) / w
                else:
                    vp_c_err = abs(vp_lr[0] - x_center_net) / w
                if vp_c_err > 0.065:
                    continue
                dl = x_center_net - xnl
                dr = xnr - x_center_net
                sym_err = abs(dl - dr) / max(dl, dr, 1.0)
                if sym_err > 0.28:
                    continue
                pair_sc = (wl + wr) / w - 6.5 * vp_c_err - 2.2 * sym_err
                if best_pair is None or pair_sc > best_pair[0]:
                    best_pair = (pair_sc, rl, rr, vp_lr)

        if best_pair is None:
            return None
        _, r_left, r_right, _ = best_pair
        r_net = net_info["net_floor_ray"]
        r_svc = t_svc["ray"]

        p_nl = self._intersect_rays(r_left, r_net)
        p_nr = self._intersect_rays(r_right, r_net)
        p_sr = self._intersect_rays(r_right, r_svc)
        p_sl = self._intersect_rays(r_left, r_svc)
        if any(p is None for p in (p_nl, p_nr, p_sr, p_sl)):
            return None

        court_ref = [
            [0.0, NET_Y_M],
            [COURT_WIDTH_M, NET_Y_M],
            [COURT_WIDTH_M, SERVICE_NEAR_M],
            [0.0, SERVICE_NEAR_M],
        ]
        img_ref = [p_nl, p_nr, p_sr, p_sl]

        if center_cl is not None:
            p_cn = self._intersect_rays(center_cl["ray"], r_net)
            p_cs = self._intersect_rays(center_cl["ray"], r_svc)
            if p_cn is not None and p_cs is not None:
                court_ref += [[0.5 * COURT_WIDTH_M, NET_Y_M], [0.5 * COURT_WIDTH_M, SERVICE_NEAR_M]]
                img_ref += [p_cn, p_cs]

        H_c2i, _ = cv2.findHomography(np.array(court_ref, np.float32), np.array(img_ref, np.float32), 0)
        H_c2i = normalize_homography_sign(H_c2i, (5.0, 10.0))
        if H_c2i is None:
            return None

        H_c2i = self._refine_icp_lines(tophat, floor_mask, H_c2i, court_ref, img_ref)
        corners = project_pts(H_c2i, FULL_COURT_CORNERS_M)
        if not np.all(np.isfinite(corners)) or compute_homography(corners) is None:
            return None
        # Plausibility check: cameras suction-mounted on the back glass (`Y_cam ~ 20.2m`) project `Y=20m`
        # to ~15-35x image width (`60x` cap matches `corners_plausible` in `padeliq_live_stream.py`)
        if np.abs(corners).max() > 60.0 * max(w, h):
            return None

        # On a low-baseline view, far baseline (`Y=0`) cannot float high above the net headband tape!
        min_allowed_y = int(net_info["band_y_mid"] - 0.035 * h)
        if min(corners[0, 1], corners[1, 1]) < min_allowed_y:
            return None

        sc, meta = self._score_homography(
            bg_bgr, floor_mask, tophat, H_c2i, floor_top_y, ankle_seeds, mode="low_baseline"
        )
        if center_cl is not None:
            sc += 0.14
        return (sc, corners.astype(np.float32), meta)

    @staticmethod
    def _refine_icp_lines(tophat, floor_mask, H_c2i, anchor_court=None, anchor_img=None, search_px=9, iters=4):
        h, w = tophat.shape[:2]
        model_lines = [
            ((0.0, SERVICE_FAR_M), (COURT_WIDTH_M, SERVICE_FAR_M), "svc_far"),
            ((0.0, SERVICE_NEAR_M), (COURT_WIDTH_M, SERVICE_NEAR_M), "svc_near"),
            ((0.5 * COURT_WIDTH_M, SERVICE_FAR_M), (0.5 * COURT_WIDTH_M, SERVICE_NEAR_M), "center"),
            ((0.0, 0.0), (COURT_WIDTH_M, 0.0), "base_far"),
            ((0.0, COURT_LENGTH_M), (COURT_WIDTH_M, COURT_LENGTH_M), "base_near"),
            ((0.0, 0.0), (0.0, COURT_LENGTH_M), "side_l"),
            ((COURT_WIDTH_M, 0.0), (COURT_WIDTH_M, COURT_LENGTH_M), "side_r"),
        ]
        dom = cv2.dilate(floor_mask, np.ones((15, 15), np.uint8))
        offs = np.arange(-search_px, search_px + 1)

        def _snap(H_cur):
            c_pts, i_pts, names = [], [], []
            for p0, p1, nm in model_lines:
                ts = np.linspace(0.04, 0.96, 42)[:, None]
                cp = (np.array(p0) * (1.0 - ts) + np.array(p1) * ts).astype(np.float32)
                ip = project_pts(H_cur, cp)
                for j in range(len(cp)):
                    x, y = ip[j]
                    if not ( np.isfinite(x) and np.isfinite(y)
                             and search_px <= x < w - search_px and search_px <= y < h - search_px ):
                        continue
                    if dom[int(y), int(x)] == 0:
                        continue
                    j0, j1 = max(0, j - 1), min(len(cp) - 1, j + 1)
                    if not (np.all(np.isfinite(ip[j0])) and np.all(np.isfinite(ip[j1]))):
                        continue
                    tang = ip[j1] - ip[j0]
                    nrm = np.array([-tang[1], tang[0]], dtype=np.float64)
                    nl = np.linalg.norm(nrm)
                    if nl < 1e-6:
                        continue
                    nrm /= nl
                    xs = np.clip(np.round(x + offs * nrm[0]).astype(int), 0, w - 1)
                    ys = np.clip(np.round(x * 0 + y + offs * nrm[1]).astype(int), 0, h - 1)
                    vals = tophat[ys, xs]
                    k = int(np.argmax(vals))
                    if vals[k] >= 24 and 0 < k < len(offs) - 1:
                        a_, b_, c_ = float(vals[k - 1]), float(vals[k]), float(vals[k + 1])
                        den = a_ - 2.0 * b_ + c_
                        sub = 0.5 * (a_ - c_) / den if abs(den) > 1e-6 else 0.0
                        step = offs[k] + sub
                        c_pts.append(cp[j])
                        i_pts.append([x + step * nrm[0], y + step * nrm[1]])
                        names.append(nm)
            return c_pts, i_pts, names

        H = normalize_homography_sign(H_c2i.copy(), (5.0, 10.0))
        cp0, ip0, _ = _snap(H)
        best_res = (
            float(np.median(np.linalg.norm(np.array(ip0) - project_pts(H, cp0), axis=1)))
            if len(cp0) >= 16 else None
        )
        for _ in range(iters):
            cp, ip, names = _snap(H)
            if len(cp) < 22 or len(set(names)) < 2:
                break
            c_arr = np.array(cp, dtype=np.float32)
            i_arr = np.array(ip, dtype=np.float32)
            if anchor_court is not None and anchor_img is not None:
                c_arr = np.vstack([c_arr] + [np.asarray(anchor_court, np.float32)] * 3)
                i_arr = np.vstack([i_arr] + [np.asarray(anchor_img, np.float32)] * 3)
            Hn, _ = cv2.findHomography(c_arr, i_arr, cv2.RANSAC, 2.2, maxIters=2500, confidence=0.995)
            Hn = normalize_homography_sign(Hn, (5.0, 10.0))
            if Hn is None:
                break
            cp_n, ip_n, _ = _snap(Hn)
            if len(cp_n) < 18:
                break
            res_n = float(np.median(np.linalg.norm(np.array(ip_n) - project_pts(Hn, cp_n), axis=1)))
            if best_res is not None and res_n >= best_res - 0.04:
                break
            H, best_res = Hn, res_n
        return H

    def _score_homography(self, bg_bgr, floor_mask, tophat, H_c2i, floor_top_y, ankle_seeds, mode="full_quad"):
        h, w = bg_bgr.shape[:2]
        H_c2i = normalize_homography_sign(H_c2i, (5.0, 10.0))
        try:
            H_i2c = np.linalg.inv(H_c2i)
        except np.linalg.LinAlgError:
            return -1e9, {}
        c_px = project_pts(H_c2i, [[5.0, 10.0]])[0]
        if np.all(np.isfinite(c_px)):
            H_i2c = normalize_homography_sign(H_i2c, c_px)

        gx, gy = np.meshgrid(np.linspace(0.4, COURT_WIDTH_M - 0.4, 14), np.linspace(0.4, COURT_LENGTH_M - 0.4, 28))
        grid_m = np.column_stack([gx.ravel(), gy.ravel()])
        grid_px = project_pts(H_c2i, grid_m)
        vis = (
            np.isfinite(grid_px[:, 0]) & np.isfinite(grid_px[:, 1])
            & (grid_px[:, 0] >= 0) & (grid_px[:, 0] < w)
            & (grid_px[:, 1] >= 0) & (grid_px[:, 1] < h)
        )
        y_bot = int(getattr(self, "_y_bot", h) or h)
        if y_bot < h:
            vis &= grid_px[:, 1] < y_bot - 2            # hidden behind a foreground railing: unknown, not "off floor"
        if np.count_nonzero(vis) < 25:
            return -1e9, {}
        vx = grid_px[vis, 0].astype(int)
        vy = grid_px[vis, 1].astype(int)
        surf_precision = float(np.mean(floor_mask[vy, vx] > 0))

        fy_m, fx_m = np.where(floor_mask[::4, ::4] > 0)
        surf_recall = 0.0
        if len(fx_m) > 20:
            f_pts = np.column_stack([fx_m * 4.0, fy_m * 4.0])
            c_back = project_pts(H_i2c, f_pts)
            ok_b = (
                np.isfinite(c_back[:, 0]) & np.isfinite(c_back[:, 1])
                & (c_back[:, 0] >= -0.8) & (c_back[:, 0] <= COURT_WIDTH_M + 0.8)
                & (c_back[:, 1] >= -0.8) & (c_back[:, 1] <= COURT_LENGTH_M + 0.8)
            )
            surf_recall = float(np.mean(ok_b))

        near_ridge = cv2.dilate((tophat > 24).astype(np.uint8), np.ones((5, 5), np.uint8))
        test_lines = [
            ((0.0, SERVICE_NEAR_M), (COURT_WIDTH_M, SERVICE_NEAR_M)),
            ((0.5 * COURT_WIDTH_M, SERVICE_FAR_M), (0.5 * COURT_WIDTH_M, SERVICE_NEAR_M)),
            ((0.0, SERVICE_FAR_M), (COURT_WIDTH_M, SERVICE_FAR_M)),
        ]
        if mode == "full_quad":
            test_lines += [
                ((0.0, 0.0), (COURT_WIDTH_M, 0.0)),
                ((0.0, COURT_LENGTH_M), (COURT_WIDTH_M, COURT_LENGTH_M)),
                ((0.0, 0.0), (0.0, COURT_LENGTH_M)),
                ((COURT_WIDTH_M, 0.0), (COURT_WIDTH_M, COURT_LENGTH_M)),
            ]
        hit = tot = 0
        for p0, p1 in test_lines:
            ts = np.linspace(0.05, 0.95, 40)[:, None]
            lp = project_pts(H_c2i, np.array(p0) * (1.0 - ts) + np.array(p1) * ts)
            for x, y in lp:
                if np.isfinite(x) and np.isfinite(y) and 3 <= x < w - 3 and max(3, floor_top_y - 6) <= y < y_bot - 3:
                    tot += 1
                    hit += int(near_ridge[int(y), int(x)] > 0)
        line_sup = (hit / tot) if tot >= 12 else 0.0

        ank_score = 0.75
        if ankle_seeds:
            in_floor_anks, in_wts = [], []
            for ax, ay, wt, _ in ankle_seeds:
                ix, iy = int(np.clip(round(ax), 0, w - 1)), int(np.clip(round(ay), 0, h - 1))
                if floor_mask[iy, ix] > 0:
                    in_floor_anks.append([ax, ay])
                    in_wts.append(wt)
            if in_floor_anks:
                cm = project_pts(H_i2c, in_floor_anks)
                ok_a = (
                    np.isfinite(cm[:, 0]) & np.isfinite(cm[:, 1])
                    & (cm[:, 0] >= -0.6) & (cm[:, 0] <= COURT_WIDTH_M + 0.6)
                    & (cm[:, 1] >= -0.7) & (cm[:, 1] <= COURT_LENGTH_M + 0.7)
                )
                w_arr = np.array(in_wts, dtype=np.float64)
                ank_score = float((ok_a * w_arr).sum() / max(1e-9, w_arr.sum()))

        total_score = 0.30 * surf_precision + 0.28 * surf_recall + 0.18 * line_sup + 0.24 * ank_score
        conf_pct = float(np.clip(100.0 * (0.42 * min(1.0, surf_precision / 0.88)
                                          + 0.28 * min(1.0, surf_recall / 0.82)
                                          + 0.18 * min(1.0, max(line_sup, 0.45) / 0.75)
                                          + 0.12 * ank_score), 58.0, 99.2))
        meta = {
            "surface_agreement_pct": round(100.0 * surf_precision, 1),
            "surface_recall_pct": round(100.0 * surf_recall, 1),
            "line_support_pct": round(100.0 * line_sup, 1),
            "ankle_containment_pct": round(100.0 * ank_score, 1),
            "confidence_pct": round(conf_pct, 1),
            "solver_mode": mode,
        }
        return total_score, meta

    def _refine_with_ankle_hull(self, corners, floor_mask, ankle_seeds, frame_shape):
        h, w = frame_shape[:2]
        pts = corners.copy().astype(np.float32)

        ys_on = np.where(floor_mask > 0)[0]
        if len(ys_on) > 0:
            y_min_floor = float(ys_on.min())
            for k in (0, 1):
                if 0 <= pts[k, 0] < w and pts[k, 1] < y_min_floor - 4.0:
                    pts[k, 1] = y_min_floor

        H_i2c = compute_homography(pts)
        if H_i2c is None or not ankle_seeds:
            return pts

        in_anks = []
        for ax, ay, wt, bh in ankle_seeds:
            ix, iy = int(np.clip(round(ax), 0, w - 1)), int(np.clip(round(ay), 0, h - 1))
            y0, y1 = max(0, iy - 4), min(h, iy + 5)
            x0, x1 = max(0, ix - 4), min(w, ix + 5)
            if float(np.mean(floor_mask[y0:y1, x0:x1] > 0)) > 0.8 and bh >= 0.11 * h:
                in_anks.append([ax, ay])

        if len(in_anks) < 3:
            return pts

        cm = project_pts(H_i2c, in_anks)
        valid = np.isfinite(cm[:, 0]) & np.isfinite(cm[:, 1])
        cm = cm[valid]
        if len(cm) >= 3:
            xmin_a = float(np.percentile(cm[:, 0], 5))
            xmax_a = float(np.percentile(cm[:, 0], 95))
            x_lo = min(0.0, max(-0.40, xmin_a - 0.10))
            x_hi = max(COURT_WIDTH_M, min(COURT_WIDTH_M + 0.40, xmax_a + 0.10))
            if x_lo < -0.05 or x_hi > COURT_WIDTH_M + 0.05:
                H_c2i = invert_homography_c2i(H_i2c)
                if H_c2i is not None:
                    expanded = np.array([[x_lo, 0.0], [x_hi, 0.0], [x_hi, COURT_LENGTH_M], [x_lo, COURT_LENGTH_M]], np.float32)
                    cand_pts = project_pts(H_c2i, expanded)
                    if np.all(np.isfinite(cand_pts)) and compute_homography(cand_pts) is not None:
                        pts = cand_pts.astype(np.float32)
        return pts

    @staticmethod
    def classify_camera_angle(corners, vp_long, vp_trans, frame_shape):
        h, w = frame_shape[:2]
        top_dx = corners[1, 0] - corners[0, 0]
        top_dy = corners[1, 1] - corners[0, 1]
        top_slope = abs(top_dy / max(1.0, abs(top_dx)))

        vp_offset = 0.0
        if vp_long is not None and abs(vp_long[2]) > 1e-5:
            vx = vp_long[0] / vp_long[2]
            vp_offset = abs(vx - 0.5 * w) / w

        if top_slope > 0.042 or vp_offset > 0.13:
            return "Oblique"
        near_in = (-0.15 * w <= corners[2, 0] <= 1.15 * w and corners[2, 1] <= 1.05 * h)
        if near_in and corners[0, 1] < 0.34 * h:
            return "Elevated"
        return "Baseline"

    def _fallback_floor_quad(self, floor_mask, bg_bgr, tophat, floor_top_y, ankle_seeds):
        cnts, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        hull = cv2.convexHull(max(cnts, key=cv2.contourArea)).reshape(-1, 2).astype(np.float32)
        if len(hull) < 4:
            return None
        # Corners from the top / bottom bands of the hull (not x+y extremes): when the floor runs off the image
        # sides, the x+y extreme lands mid-height on the frame edge and the quad would lose the far court.
        dense = []
        for (x0, y0), (x1, y1) in zip(hull, np.roll(hull, -1, axis=0)):
            n = max(2, int(np.hypot(x1 - x0, y1 - y0) // 4))
            dense.append(np.column_stack([np.linspace(x0, x1, n), np.linspace(y0, y1, n)]))
        pts = np.vstack(dense)
        y_top, y_bot = float(pts[:, 1].min()), float(pts[:, 1].max())
        band = max(4.0, 0.06 * (y_bot - y_top))
        top = pts[pts[:, 1] <= y_top + band]
        bot = pts[pts[:, 1] >= y_bot - band]
        tl, tr = top[int(np.argmin(top[:, 0]))], top[int(np.argmax(top[:, 0]))]
        bl, br = bot[int(np.argmin(bot[:, 0]))], bot[int(np.argmax(bot[:, 0]))]
        if tr[0] - tl[0] < 8 or br[0] - bl[0] < 8:
            return None
        corners = np.array([tl, tr, br, bl], dtype=np.float32)
        H_i2c = compute_homography(corners)
        if H_i2c is None:
            return None
        H_c2i = invert_homography_c2i(H_i2c)
        if H_c2i is None:
            return None
        sc, meta = self._score_homography(bg_bgr, floor_mask, tophat, H_c2i, floor_top_y, ankle_seeds, mode="floor_hull")
        return (sc, corners, meta)

    def calibrate(self, frames, fast_single=False):
        """Run Stages A + B on `frames` (1 frame for instant seed, 15-30 frames for full lock)."""
        t0 = time.perf_counter()
        valid_frames = [f for f in frames if f is not None and getattr(f, "size", 0) > 0]
        if not valid_frames:
            return None, "no valid frames"
        h, w = valid_frames[0].shape[:2]
        valid_frames = [f for f in valid_frames if f.shape[:2] == (h, w)][:30]

        if len(valid_frames) >= 3 and not fast_single:
            sub = valid_frames[::max(1, len(valid_frames) // 15)][:15]
            bg = np.median(np.stack(sub, axis=0), axis=0).astype(np.uint8)
        else:
            bg = valid_frames[len(valid_frames) // 2].copy()

        # Fast single-frame provisional seed runs pure LAB+HSV + 2-VP in ~35ms (zero lock contention);
        # full 15-frame convergence runs 2 pose passes (~0.8-1.3s total!)
        if fast_single:
            ankle_seeds, person_boxes = [], []
        else:
            ankle_seeds, person_boxes = self.extract_player_ankle_seeds(valid_frames, max_pose_frames=min(2, len(valid_frames)))

        floor_mask, tophat, floor_meta = self.segment_primary_floor(bg, ankle_seeds)
        floor_top_y = floor_meta["floor_top_y"]
        if floor_meta["floor_area_pct"] < 4.0:
            return None, "no primary court floor surface visible in frame"
        occ_y = floor_meta.get("occluder_y")
        self._y_bot = int(occ_y) if occ_y is not None else h

        segs, _ = self._extract_segments(bg, floor_mask, tophat, y_bot=self._y_bot)
        vp_res = self._ransac_two_vps(segs, bg.shape)

        candidates = []
        vp_long, vp_trans = None, None

        if vp_res is not None:
            v1, in1, v2, in2, focal_px = vp_res
            ys_f, xs_f = np.where(floor_mask > 0)
            ref_pt = (float(np.mean(xs_f)), float(np.mean(ys_f)))

            for va, ina, vb, inb in ((v1, in1, v2, in2), (v2, in2, v1, in1)):
                segs_a = [segs[i] for i in np.where(ina)[0]]
                segs_b = [segs[i] for i in np.where(inb)[0]]
                cl_a = self._cluster_pencil(va, segs_a, ref_pt, merge_px=max(8.0, 11.0 * h / 720.0))
                cl_b = self._cluster_pencil(vb, segs_b, ref_pt, merge_px=max(8.0, 11.0 * h / 720.0))

                aspect_bonus = 0.0
                try:
                    H_rect = self._metric_rectify_H(vb, va, focal_px, bg.shape)
                    cnts, _ = cv2.findContours(floor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    hull_px = cv2.convexHull(max(cnts, key=cv2.contourArea)).reshape(-1, 2)
                    rect_pts = project_pts(H_rect, hull_px)
                    if np.all(np.isfinite(rect_pts)):
                        dx_r = float(np.ptp(rect_pts[:, 0]))
                        dy_r = float(np.ptp(rect_pts[:, 1]))
                        ratio = dy_r / max(1e-6, dx_r)
                        if 1.25 <= ratio <= 3.2:
                            aspect_bonus = 0.08
                        elif ratio < 0.85:
                            aspect_bonus = -0.10
                except Exception:
                    pass

                cand_full = self._solve_full_quad_candidate(
                    bg, floor_mask, tophat, va, cl_a, vb, cl_b, focal_px, floor_top_y, ankle_seeds
                )
                if cand_full is not None:
                    candidates.append((cand_full[0] + aspect_bonus, cand_full[1], cand_full[2], va, vb))

                cand_low = self._solve_low_baseline_candidate(
                    bg, floor_mask, tophat, va, cl_a, vb, cl_b, floor_top_y, ankle_seeds
                )
                if cand_low is not None:
                    candidates.append((cand_low[0] + aspect_bonus, cand_low[1], cand_low[2], va, vb))

        if not candidates:
            fb = self._fallback_floor_quad(floor_mask, bg, tophat, floor_top_y, ankle_seeds)
            if fb is None:
                return None, "insufficient court lines on segmented floor"
            candidates.append((fb[0], fb[1], fb[2], None, None))

        # Candidate gates + player-guided validation: the visible court must span >= 28 % of the frame height,
        # must not sit in the bottom 18 % (walkway under a railing), and the real court contains the players.
        feet = self._player_feet(person_boxes, h, w)
        gated = []
        for cand in candidates:
            vis = self.visible_court_extent(cand[1], (h, w), self._y_bot)
            if vis is None or vis[1] < 0.28 * h or vis[0] > 0.82 * h:
                continue
            frac = self._feet_inside(cand[1], feet)
            cand[2]["players_inside_pct"] = None if frac is None else round(100.0 * frac, 1)
            gated.append((cand[0] + (0.0 if frac is None else 0.22 * frac), cand[1], cand[2], cand[3], cand[4]))
        if not gated:
            return None, "no court-sized floor region (only strips at the frame edge)"
        candidates = gated

        best_sc, corners, meta, vp_long, vp_trans = max(candidates, key=lambda c: c[0])

        H_i2c_init = compute_homography(corners)
        if H_i2c_init is not None and meta.get("solver_mode") == "full_quad":
            H_c2i_init = invert_homography_c2i(H_i2c_init)
            if H_c2i_init is not None:
                H_c2i_ref = self._refine_icp_lines(
                    tophat, floor_mask, H_c2i_init, FULL_COURT_CORNERS_M, corners
                )
                c_ref = project_pts(H_c2i_ref, FULL_COURT_CORNERS_M)
                if ( np.all(np.isfinite(c_ref)) and compute_homography(c_ref) is not None
                     and min(c_ref[0, 1], c_ref[1, 1]) >= floor_top_y - 0.03 * h ):
                    corners = c_ref.astype(np.float32)

        corners = self._refine_with_ankle_hull(corners, floor_mask, ankle_seeds, bg.shape)
        H_i2c = compute_homography(corners)
        if H_i2c is None:
            return None, "degenerate court homography after refinement"

        # Update final top floor gate to never be above the calibrated far baseline on low/straight cameras
        eff_top_y = max(floor_top_y, int(min(corners[0, 1], corners[1, 1]) - 0.02 * h))
        angle_label = self.classify_camera_angle(corners, vp_long, vp_trans, bg.shape)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)

        info = dict(floor_meta)
        info.update(meta)
        band = self.foreground_occluder_band(bg) if occ_y is not None else None
        info.update({
            "floor_bottom_y": None if occ_y is None else int(occ_y),
            "occluder_bottom_y": None if band is None else int(band[1]),
            "method": f"court_geometry_ai_{meta.get('solver_mode', 'vp')}",
            "engine": "CourtGeometryAI (Surface + 2-VP RANSAC + Pose-Ankle Hull)",
            "angle": angle_label,
            "frames_used": len(valid_frames),
            "ankle_seeds": len(ankle_seeds),
            "solve_ms": elapsed_ms,
            "floor_top_y": eff_top_y,
            "vp_long": None if vp_long is None or abs(vp_long[2]) < 1e-5 else [round(float(vp_long[0] / vp_long[2]), 1), round(float(vp_long[1] / vp_long[2]), 1)],
            "vp_trans": None if vp_trans is None or abs(vp_trans[2]) < 1e-5 else [round(float(vp_trans[0] / vp_trans[2]), 1), round(float(vp_trans[1] / vp_trans[2]), 1)],
        })

        self.last_floor_mask = floor_mask
        self.last_floor_top_y = eff_top_y
        self.last_H_i2c = H_i2c
        self.last_H_c2i = invert_homography_c2i(H_i2c)
        self.last_info = info
        return corners.astype(np.float32), info

    @staticmethod
    def visible_court_extent(corners, frame_shape, y_bot=None):
        """(centroid_y, vertical_span) of the court polygon clipped to the visible frame (above y_bot)."""
        h, w = frame_shape[:2]
        yb = float(h if y_bot is None else y_bot)
        quad = np.asarray(corners, np.float32).reshape(-1, 2)
        if not np.all(np.isfinite(quad)):
            return None
        quad = np.clip(quad, -50.0 * max(h, w), 50.0 * max(h, w))
        if cv2.contourArea(quad) < 0:
            quad = quad[::-1].copy()
        frame = np.array([[0, 0], [w, 0], [w, yb], [0, yb]], np.float32)
        try:
            area, inter = cv2.intersectConvexConvex(np.ascontiguousarray(quad), frame)
        except cv2.error:
            return None
        if inter is None or area <= 1.0:
            return None
        pts = inter.reshape(-1, 2)
        m = cv2.moments(pts.astype(np.float32))
        cy = m["m01"] / m["m00"] if abs(m["m00"]) > 1e-6 else float(pts[:, 1].mean())
        return float(cy), float(np.ptp(pts[:, 1]))

    @staticmethod
    def _player_feet(person_boxes, h, w):
        """Foot points of person boxes from the calibration pass that sit in the playing band (y 18-75 %+)."""
        feet = []
        for _, boxes in person_boxes or []:
            for x1, y1, x2, y2 in boxes:
                bh = y2 - y1
                fy = y2
                if bh >= 0.08 * h and 0.18 * h <= fy <= 0.80 * h:
                    feet.append((0.5 * (x1 + x2), fy))
        return feet

    @staticmethod
    def _feet_inside(corners, feet):
        if len(feet) < 2:
            return None
        H = compute_homography(corners)
        if H is None:
            return None
        cm = project_pts(H, feet)
        ok = (np.isfinite(cm[:, 0]) & np.isfinite(cm[:, 1]) & (cm[:, 0] >= -0.8) & (cm[:, 0] <= COURT_WIDTH_M + 0.8)
              & (cm[:, 1] >= -1.0) & (cm[:, 1] <= COURT_LENGTH_M + 1.5))
        return float(np.mean(ok))

    # ------------------------------------------------------------------
    # Stage D: Strict Main-Court Pose-Ankle Player Filtering
    # ------------------------------------------------------------------
    # Court envelope for a tracked player's foot point (court metres): side / back glass rebound margins
    PLAYER_X_RANGE_M = (-1.2, COURT_WIDTH_M + 1.2)
    PLAYER_Y_RANGE_M = (-1.5, COURT_LENGTH_M + 1.5)
    PLAYER_ASPECT = (1.15, 4.2)            # box h / w without a skeleton (posts are thinner, benches squatter)

    @staticmethod
    def valid_human_skeleton(kp, box):
        """True / False when pose keypoints exist for this box, None when there is no skeleton to judge.
        A real person shows a shoulder above a hip with a plausible torso, plus a knee or ankle below the hip;
        a dark balcony post, glass strut or bag yields no such structure."""
        if kp is None:
            return None
        kp = np.asarray(kp, np.float32)
        if kp.shape[0] < 17 or float(np.max(kp[:, 2])) <= 0.0:
            return None
        bh = max(1.0, float(box[3] - box[1]))
        ok = kp[:, 2] > 0.35
        sh = [i for i in (5, 6) if ok[i]]
        hp = [i for i in (11, 12) if ok[i]]
        legs = [i for i in (13, 14, 15, 16) if ok[i]]
        if not sh or not hp:
            return False
        y_sh = float(np.mean(kp[sh, 1]))
        y_hp = float(np.mean(kp[hp, 1]))
        torso = y_hp - y_sh
        if torso < 0.12 * bh or torso > 0.62 * bh:
            return False
        if legs and float(np.max(kp[legs, 1])) < y_hp:
            return False
        if len(sh) == 2 and len(hp) == 2:      # left/right symmetry: shoulder and hip widths agree in scale
            ws, wh = abs(kp[5, 0] - kp[6, 0]), abs(kp[11, 0] - kp[12, 0])
            if max(ws, wh) > 6.0 * max(4.0, min(ws, wh)):
                return False
        return bool(legs) or int(np.count_nonzero(ok)) >= 7

    @staticmethod
    def filter_main_court_players(boxes, kps_list, H_i2c, frame_shape, floor_mask=None, floor_top_y=None,
                                  occluder_y=None, occluder_bottom_y=None):
        if H_i2c is None or boxes is None or len(boxes) == 0:
            return []
        h, w = frame_shape[:2]
        x_lo, x_hi = CourtGeometryAI.PLAYER_X_RANGE_M
        y_lo, y_hi = CourtGeometryAI.PLAYER_Y_RANGE_M
        a_lo, a_hi = CourtGeometryAI.PLAYER_ASPECT
        dil_floor = None
        if floor_mask is not None and floor_mask.shape[:2] == (h, w):
            k = max(15, int(round(23 * h / 720.0)) | 1)
            dil_floor = cv2.dilate(floor_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))

        top_gate = (floor_top_y - int(0.025 * h)) if floor_top_y is not None else int(0.15 * h)

        raw_cands = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            bh, bw = y2 - y1, x2 - x1
            if bh < 0.075 * h or bw < 8.0:
                continue
            if y2 < top_gate:
                continue

            kp = kps_list[i] if (kps_list is not None and i < len(kps_list)) else None
            # Real human structure: a skeleton must look human; without one the box shape must
            aspect = bh / max(1.0, bw)
            skel = CourtGeometryAI.valid_human_skeleton(kp, box)
            if skel is False or aspect > a_hi:
                continue
            if skel is None and aspect < a_lo:
                continue
            # In front of a foreground railing (balcony post, spectator on the walkway): a player behind the
            # railing has the feet hidden BY it, so the box never extends past the railing's bottom edge.
            if occluder_y is not None:
                rail_bot = occluder_bottom_y if occluder_bottom_y is not None else occluder_y + 0.06 * h
                if y2 > rail_bot + 0.015 * h:
                    continue

            foot_xy, method, sig_px, _ = pose_ground_point(box, kp, frame_shape)
            fx, fy = foot_xy
            feet_hidden = occluder_y is not None and fy >= occluder_y - 0.01 * h

            pos_m = project_pts(H_i2c, [[fx, fy]])[0]
            if not np.all(np.isfinite(pos_m)):
                continue
            xm, ym = float(pos_m[0]), float(pos_m[1])

            # Court envelope (`10m x 20m` + glass rebound margins). Sideways a player may run out through the side
            # openings (x -1.2 .. 11.2 m, foot on the turf); behind the BACK glass there is no playing area, so
            # lengthwise only the pose-ground tolerance applies (-0.95 m for an upright player at the glass).
            if not (x_lo <= xm <= x_hi and y_lo <= ym <= y_hi):
                continue
            min_ym = -0.95 if (0.35 <= xm <= COURT_WIDTH_M - 0.35 and aspect >= 1.55 and bh >= 0.105 * h) else -0.58
            if not (min_ym <= ym <= COURT_LENGTH_M + 1.15):
                continue
            inner = -0.60 <= xm <= COURT_WIDTH_M + 0.60

            # Primary floor mask containment check when foot is inside the image frame (and not behind a railing)
            on_floor = None
            if dil_floor is not None and 2 <= fx < w - 2 and 2 <= fy < h - 2 and not feet_hidden:
                ix, iy = int(round(fx)), int(round(fy))
                on_floor = dil_floor[iy, ix] > 0
                if not on_floor:
                    if not (inner and 0.25 <= xm <= COURT_WIDTH_M - 0.25 and min_ym <= ym <= 0.40):
                        continue
            if not inner and not on_floor:
                continue

            # Reject seated spectators behind back glass (squat aspect or tiny height near `ym ~ 0`)
            if ym < 1.35 and (bh / max(1.0, bw) < 1.32 or bh < 0.092 * h):
                continue

            p_dx = project_pts(H_i2c, [[fx + sig_px, fy]])[0]
            p_dy = project_pts(H_i2c, [[fx, fy + sig_px]])[0]
            sx = float(np.clip(np.linalg.norm(p_dx - pos_m), 0.03, 1.4)) if np.all(np.isfinite(p_dx)) else 0.25
            sy = float(np.clip(np.linalg.norm(p_dy - pos_m), 0.03, 1.4)) if np.all(np.isfinite(p_dy)) else 0.25

            far_side = bool(ym < NET_Y_M)
            raw_cands.append({
                "box": (x1, y1, x2, y2),
                "kps": kp,
                "foot_px": (fx, fy),
                "pos_m": np.array([xm, ym], dtype=np.float32),
                "sigma_m": (sx, sy),
                "far_side": far_side,
                "method": method,
                "bh": bh,
            })

        if len(raw_cands) > 1:
            for c in list(raw_cands):
                peers = [o["bh"] for o in raw_cands if o is not c and abs(o["pos_m"][1] - c["pos_m"][1]) < 4.5]
                if peers and c["bh"] < 0.48 * max(peers):
                    raw_cands.remove(c)

        def _rank(c):
            return -c["bh"] + 8.0 * max(0.0, abs(c["pos_m"][0] - 5.0) - 4.5)
        far_c = sorted([c for c in raw_cands if c["far_side"]], key=_rank)[:3]
        near_c = sorted([c for c in raw_cands if not c["far_side"]], key=_rank)[:3]
        return far_c + near_c
