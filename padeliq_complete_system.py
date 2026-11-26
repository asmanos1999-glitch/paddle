import os
import time
import json
import threading
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
from scipy.optimize import linear_sum_assignment

# =====================================================================
# 1. ستانداردی یاریگا و ڕەنگەکانی HUD
# =====================================================================
COURT_WIDTH_M = 10.0
COURT_LENGTH_M = 20.0
CALIBRATION_FILE = "live_court_calibration.json"

SLOT_COLORS = {
    1: (0, 110, 255),   # P1: پرتەقاڵی
    2: (50, 220, 50),   # P2: سەوز
    3: (0, 220, 220),   # P3: زەرد
    4: (180, 105, 255)  # P4: پەمەیی/مۆر
}

# =====================================================================
# 2. خوێنەری کامێرای زیندوو بەبێ دواکەوتن (Zero-Latency Threaded Camera)
# =====================================================================
class ThreadedLiveCamera:
    """
    ئەم کلاسە لە باکگراوند بەردەوام فڕەیمی کامێراکە (ESP32-CAM / RTSP / Webcam)
    دەخوێنێتەوە بۆ ئەوەی Buffer پڕ نەبێت و هەمیشە دیمەنی هەمان چرکە (Real-time) بدات.
    """
    def __init__(self, source):
        # ئەگەر ژمارە بوو (وەک "0" یان "1") بیکە بە int بۆ وێبکامی کۆمپیوتەر
        if isinstance(source, str) and source.isdigit():
            self.source = int(source)
        else:
            self.source = source

        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(f"نەتوانرا پەیوەندی بە کامێرای زیندووەوە بکرێت: {self.source}")

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

# =====================================================================
# 3. دیاریکردنی ٤ گۆشەی زەوی لەسەر کامێرای لایڤ بە ماوس
# =====================================================================
def load_or_default_corners(frame_shape):
    h, w = frame_shape[:2]
    if os.path.exists(CALIBRATION_FILE):
        try:
            with open(CALIBRATION_FILE, "r") as f:
                pts = json.load(f)
                if len(pts) == 4:
                    return np.array(pts, dtype=np.float32)
        except Exception:
            pass
    # چوار گۆشەی گریمانەیی لەسەر شاشەکە تا کاتێک بەکارهێنەر پیتی 'C' دادەگرێت
    return np.array([
        [w * 0.22, h * 0.25],
        [w * 0.78, h * 0.25],
        [w * 0.92, h * 0.92],
        [w * 0.08, h * 0.92]
    ], dtype=np.float32)

def interactive_live_calibration(current_frame):
    clicked = []
    win_name = "LIVE CALIBRATION: Click 4 Floor Corners (TL, TR, BR, BL) -> Press 'C' to Save"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < 4:
            clicked.append([float(x), float(y)])

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, on_mouse)
    labels = ["1: Top-Left", "2: Top-Right", "3: Bottom-Right", "4: Bottom-Left"]

    while True:
        disp = current_frame.copy()
        cv2.rectangle(disp, (10, 10), (760, 60), (0, 0, 0), -1)
        msg = labels[len(clicked)] if len(clicked) < 4 else "Press 'C' to Confirm or 'R' to Reset"
        cv2.putText(disp, f"Select Corner: {msg}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        for i, pt in enumerate(clicked):
            px, py = int(pt[0]), int(pt[1])
            cv2.circle(disp, (px, py), 7, (0, 255, 0), -1)
            cv2.putText(disp, labels[i], (px + 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            if i > 0:
                cv2.line(disp, (int(clicked[i-1][0]), int(clicked[i-1][1])), (px, py), (0, 255, 255), 2)
        if len(clicked) == 4:
            cv2.line(disp, (int(clicked[3][0]), int(clicked[3][1])), (int(clicked[0][0]), int(clicked[0][1])), (0, 255, 255), 2)

        cv2.imshow(win_name, disp)
        k = cv2.waitKey(20) & 0xFF
        if k in [ord('r'), ord('R')]:
            clicked.clear()
        elif k in [ord('c'), ord('C')] and len(clicked) == 4:
            break
        elif k == 27:
            cv2.destroyWindow(win_name)
            return None

    cv2.destroyWindow(win_name)
    with open(CALIBRATION_FILE, "w") as f:
        json.dump(clicked, f, indent=2)
    return np.array(clicked, dtype=np.float32)

# =====================================================================
# 4. مەکینەی شوێنپێهەڵگرتن و ماتماتیکی زیندوو (Real-Time Analytics)
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

    def update(self, bbox, foot_px, pos_m, now):
        dt = max(0.016, min(0.5, now - self.last_update_time))
        self.last_update_time = now

        bbox_arr = np.array(bbox, dtype=np.float32)
        if self.bbox is None or (now - self.last_seen_time) > 1.2:
            self.bbox = bbox_arr
            self.pixel_trail.clear()
        else:
            self.bbox = 0.55 * bbox_arr + 0.45 * self.bbox

        self.last_seen_time = now
        self.tracked_seconds_total += dt

        sb = self.bbox
        smooth_foot = (int((sb[0] + sb[2]) / 2), int(sb[3]))
        self.pixel_trail.append(smooth_foot)

        raw_m = np.array(pos_m, dtype=np.float32)
        if self.smoothed_pos_m is None:
            self.smoothed_pos_m = raw_m
        else:
            new_smoothed = 0.35 * raw_m + 0.65 * self.smoothed_pos_m
            step_dist = float(np.linalg.norm(new_smoothed - self.smoothed_pos_m))

            if 0.01 < step_dist < 1.8:
                self.distance_m += step_dist
                inst_spd = min(8.5, step_dist / dt)
                self.instant_speed_ms = 0.7 * self.instant_speed_ms + 0.3 * inst_spd
                self.speed_history.append(self.instant_speed_ms)
                if self.instant_speed_ms > 0.35:
                    self.active_seconds_total += dt
            else:
                self.instant_speed_ms *= 0.85
                self.speed_history.append(self.instant_speed_ms)

            self.smoothed_pos_m = new_smoothed

        self.last_pos_m = self.smoothed_pos_m.copy()
        self.meter_trail.append(self.smoothed_pos_m.copy())

        if self.speed_history:
            self.avg_speed_ms = float(np.mean(self.speed_history))

        gx = int(np.clip(self.smoothed_pos_m[0], 0, 9.99))
        half_y = self.smoothed_pos_m[1] if self.slot_id in [1, 2] else (self.smoothed_pos_m[1] - 10.0)
        gy = int(np.clip(half_y, 0, 9.99))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
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


class LiveSlotManager:
    def __init__(self, homography_matrix):
        self.H = homography_matrix
        self.reset_players()

    def reset_players(self):
        anchors = {
            1: (2.5, 4.5),
            2: (7.5, 4.5),
            3: (2.5, 15.5),
            4: (7.5, 15.5)
        }
        self.players = {sid: LivePlayerState(sid, anchors[sid]) for sid in [1, 2, 3, 4]}

    def set_homography(self, H):
        self.H = H

    def pixel_to_meters(self, px_pt):
        pt = np.array([[[float(px_pt[0]), float(px_pt[1])]]], dtype=np.float32)
        return cv2.perspectiveTransform(pt, self.H)[0][0]

    def update_detections(self, boxes):
        now = time.time()
        candidates = []
        for box in boxes:
            x1, y1, x2, y2 = box
            foot_px = (int((x1 + x2) / 2), int(y2))
            pos_m = self.pixel_to_meters(foot_px)
            if -2.5 <= pos_m[0] <= COURT_WIDTH_M + 2.5 and -3.0 <= pos_m[1] <= COURT_LENGTH_M + 3.0:
                candidates.append((box, foot_px, pos_m))

        if not candidates:
            return

        if len(candidates) > 4:
            center_m = np.array([5.0, 10.0])
            candidates.sort(key=lambda c: np.linalg.norm(c[2] - center_m))
            candidates = candidates[:4]

        slot_ids = [1, 2, 3, 4]
        cost_matrix = np.zeros((4, len(candidates)), dtype=np.float32)
        for r, sid in enumerate(slot_ids):
            pl = self.players[sid]
            for c, (_, _, pos_m) in enumerate(candidates):
                dist = float(np.linalg.norm(pos_m - pl.last_pos_m))
                penalty = 0.0
                if sid in [1, 2] and pos_m[1] > 11.5:
                    penalty = 18.0
                elif sid in [3, 4] and pos_m[1] < 8.5:
                    penalty = 18.0
                cost_matrix[r, c] = dist + penalty

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 20.0:
                sid = slot_ids[r]
                box, foot_px, pos_m = candidates[c]
                self.players[sid].update(box, foot_px, pos_m, now)

# =====================================================================
# 5. کێشانی ڕووکاری گرافیکی زیندوو (Live HUD, Minimap, AI Coach)
# =====================================================================
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
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 1, cv2.LINE_AA)

def render_live_hud(frame, slot_mgr, corners_px, fps_val):
    now = time.time()

    # 1. کێشانی چوارچێوەی زەوی یاریگاکە بە هێڵێکی سەوزی کاڵ
    pts = corners_px.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [pts], isClosed=True, color=(80, 255, 80), thickness=2, lineType=cv2.LINE_AA)

    # 2. کێشانی یاریزانەکان و کارتەکانی سەر سەریان
    for sid, p in slot_mgr.players.items():
        color = SLOT_COLORS[sid]
        if len(p.pixel_trail) > 1:
            t_pts = list(p.pixel_trail)
            for i in range(1, len(t_pts)):
                cv2.line(frame, t_pts[i-1], t_pts[i], color, max(1, int((i / len(t_pts)) * 3)), cv2.LINE_AA)

        if p.bbox is not None and (now - p.last_seen_time) < 0.8:
            x1, y1, x2, y2 = map(int, p.bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            cw, ch = 185, 98
            cx1 = int((x1 + x2) / 2 - cw / 2)
            cy1 = y1 - ch - 8
            if cy1 < 10:
                cy1 = y1 + 6

            draw_glass_box(frame, cx1, cy1, cx1 + cw, cy1 + ch, bg=(15, 15, 15), alpha=0.72, border=color)
            lines = [
                f"P{sid} (Coverage: {p.coverage_pct:.1f}%)",
                f"Distance: {p.distance_m:.1f}m",
                f"AvgSpeed: {p.avg_speed_ms:.2f}m/s",
                f"Active: {int(p.active_seconds_total)}s ({p.active_pct}%)",
                f"Stability: {int(p.stability_pct)}%"
            ]
            for idx, txt in enumerate(lines):
                ty = max(20, cy1 + 17 + idx * 17)
                c = color if idx == 0 else (240, 240, 240)
                cv2.putText(frame, txt, (max(5, cx1 + 8), ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, c, 2 if idx == 0 else 1, cv2.LINE_AA)

    # 3. کێشانی 2D Minimap لە گۆشەی سەرەوەی ڕاست
    h, w = frame.shape[:2]
    mw, mh = 160, 310
    mx1, my1 = w - mw - 20, 20
    draw_glass_box(frame, mx1, my1, mx1 + mw, my1 + mh, bg=(160, 85, 30), alpha=0.80, border=(255, 255, 255))

    pad = 12
    ix1, iy1 = mx1 + pad, my1 + pad
    iw, ih = mw - 2 * pad, mh - 2 * pad
    cv2.rectangle(frame, (ix1, iy1), (ix1 + iw, iy1 + ih), (245, 245, 245), 2, cv2.LINE_AA)
    cv2.line(frame, (ix1, iy1 + ih // 2), (ix1 + iw, iy1 + ih // 2), (255, 255, 255), 2, cv2.LINE_AA)

    for sid, p in slot_mgr.players.items():
        if not p.meter_trail or (now - p.last_seen_time) > 1.5:
            continue
        color = SLOT_COLORS[sid]
        m_pts = [
            (int(ix1 + (np.clip(m[0], 0, COURT_WIDTH_M) / COURT_WIDTH_M) * iw),
             int(iy1 + (np.clip(m[1], 0, COURT_LENGTH_M) / COURT_LENGTH_M) * ih))
            for m in p.meter_trail
        ]
        for i in range(1, len(m_pts)):
            cv2.line(frame, m_pts[i-1], m_pts[i], color, 2, cv2.LINE_AA)
        cv2.circle(frame, m_pts[-1], 6, color, -1, cv2.LINE_AA)
        cv2.circle(frame, m_pts[-1], 7, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. کێشانی پەنجەرەی AI Coach و زانیاری کۆنتڕۆڵ لە سەرەوەی چەپ
    draw_glass_box(frame, 20, 20, 480, 145, bg=(18, 18, 18), alpha=0.70, border=(90, 90, 90))
    cv2.putText(frame, f"PadelIQ LIVE STREAM | {fps_val:.1f} FPS", (32, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 2, cv2.LINE_AA)
    cv2.putText(frame, "Controls: [C] Calibrate 4 Corners | [R] Reset Stats | [Q] Quit", (32, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

    active_players = [p for p in slot_mgr.players.values() if (now - p.last_seen_time) < 1.5]
    if active_players:
        p = active_players[0]
        tip = "Maintain balanced base & recover to center axis" if p.stability_pct > 80 else "Split-step before directional change to improve stability"
        cv2.putText(frame, f"AI Coach (P{p.slot_id}):", (32, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.48, SLOT_COLORS[p.slot_id], 2, cv2.LINE_AA)
        cv2.putText(frame, f" - {tip}", (32, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "AI Coach: Waiting for players inside calibrated zone...", (32, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (180, 180, 180), 1, cv2.LINE_AA)

# =====================================================================
# 6. لووپی سەرەکی کامێرای زیندوو
# =====================================================================
def run_live_padeliq(source, model_path="yolov8n.pt"):
    print(f"[*] بارکردنی مۆدێلی YOLO ({model_path})...")
    model = YOLO(model_path)

    print(f"[*] بەستنەوەی پەخشی زیندووی کامێرا لە سەرچاوەی: {source} ...")
    cam = ThreadedLiveCamera(source)

    # چاوەڕوانکردن تا یەکەم فڕەیمی زیندوو دێت
    for _ in range(50):
        ret, frame = cam.read()
        if ret and frame is not None:
            break
        time.sleep(0.05)
    else:
        cam.release()
        raise RuntimeError("هیچ فڕەیمێک لە کامێراکەوە نەگەیشت!")

    corners_px = load_or_default_corners(frame.shape)
    dst_pts = np.array([[0, 0], [COURT_WIDTH_M, 0], [COURT_WIDTH_M, COURT_LENGTH_M], [0, COURT_LENGTH_M]], dtype=np.float32)
    H, _ = cv2.findHomography(corners_px, dst_pts)

    slot_mgr = LiveSlotManager(H)
    win_name = "PadelIQ Live Camera Engine"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    fps_smooth = 25.0

    print("\n[+] پەخشی زیندوو دەستی پێکرد!")
    print("    -> لەسەر کیبۆرد پیتی 'C' دابگرە بۆ دیاریکردنی ٤ گۆشەی زەوییەکە بە ماوس.")
    print("    -> پیتی 'R' دابگرە بۆ سفرکردنەوەی ئامارەکان.")
    print("    -> پیتی 'Q' دابگرە بۆ داخستن.\n")

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                continue

            now = time.time()
            dt = max(0.001, now - prev_time)
            prev_time = now
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)

            # ناسینەوەی مرۆڤەکان لەسەر کامێرا زیندووەکە
            results = model.predict(frame, classes=[0], conf=0.30, imgsz=640, verbose=False)
            boxes = results[0].boxes.xyxy.cpu().numpy() if (results and results[0].boxes is not None) else []

            slot_mgr.update_detections(boxes)
            render_live_hud(frame, slot_mgr, corners_px, fps_smooth)

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in [ord('q'), ord('Q'), 27]:
                break
            elif key in [ord('r'), ord('R')]:
                slot_mgr.reset_players()
                print("[*] هەموو داتا و مەترەکان سفر کرانەوە!")
            elif key in [ord('c'), ord('C')]:
                new_corners = interactive_live_calibration(frame)
                if new_corners is not None:
                    corners_px = new_corners
                    H, _ = cv2.findHomography(corners_px, dst_pts)
                    slot_mgr.set_homography(H)
                    slot_mgr.reset_players()
                    print("[+] ٤ گۆشە نوێیەکە خەزن کرا و ستاندەرەکە نوێ کرایەوە!")
    finally:
        cam.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PadelIQ Real-Time Live Camera Engine (ESP32-CAM / RTSP / Webcam)")
    parser.add_argument("--source", type=str, default="0",
                        help="0 بۆ وێبکام، یان لینکى ESP32-CAM وەک http://192.168.1.50:81/stream یان RTSP")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="مۆدێلی YOLO بۆ خێرایی زیندوو")
    args = parser.parse_args()

    run_live_padeliq(source=args.source, model_path=args.model)