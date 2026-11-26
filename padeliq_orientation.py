# -*- coding: utf-8 -*-
"""
PadelIQ orientation helpers - shared by the desktop engine, the web engine and the mobile bridge.

    rotate(frame, deg)                 0 / 90 (clockwise) / 180 / 270 (= 90 counter-clockwise), lossless cv2.rotate
    landscape(frame, deg)              rotate, then NEVER hand a portrait buffer to the 16:9 pipeline: a portrait
                                       result is uniformly scaled and pillarboxed onto a 16:9 canvas (no stretching)
    orientation_score(img)             0-100: how "upright padel court" a frame looks (court floor low in the
                                       frame and widening toward the camera + horizontal net / court lines)
    detect_orientation(raw, current)   keep `current` if it scores >= 55 %, else test all four rotations
    canvas_to_raw / raw_to_canvas      normalised point mapping between the analysed canvas and the camera frame
                                       (keeps the phone's touch corner handles aligned after a server rotation)

The court AI's own confidence cannot tell orientations apart (a 90 deg rotated court still scores ~87 %), which
is why orientation uses this dedicated, cheap (~7 ms) geometric score instead.
"""
import cv2
import numpy as np

ROTATIONS = (0, 90, 180, 270)
_CODES = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
MIN_CONF = 55.0
PILLAR_H = 720                     # portrait results are pillarboxed at this height (16:9 canvas)


def norm_deg(deg):
    return int(round((int(deg or 0) % 360) / 90.0)) * 90 % 360


def rotate(frame, deg):
    code = _CODES.get(norm_deg(deg))
    return frame if code is None else cv2.rotate(frame, code)


def _geometry(raw_w, raw_h, deg):
    """-> rotated (w, h), canvas (w, h), scale, x offset of the rotated image inside the canvas."""
    deg = norm_deg(deg)
    rw, rh = (raw_h, raw_w) if deg in (90, 270) else (raw_w, raw_h)
    if rh <= rw:
        return rw, rh, rw, rh, 1.0, 0
    s = min(1.0, PILLAR_H / float(rh))
    ch = int(round(rh * s))
    cw = int(round(ch * 16 / 9.0))
    return rw, rh, cw, ch, s, (cw - int(round(rw * s))) // 2


def landscape(frame, deg):
    """Rotate by deg; a portrait result is pillarboxed (uniform scale), never squashed into 16:9."""
    out = rotate(frame, deg)
    h, w = out.shape[:2]
    if h <= w:
        return out
    rw, rh, cw, ch, s, ox = _geometry(frame.shape[1], frame.shape[0], deg)
    canvas = np.zeros((ch, cw, 3), np.uint8)
    nw = int(round(rw * s))
    img = cv2.resize(out, (nw, ch), interpolation=cv2.INTER_AREA) if s != 1.0 else out
    canvas[:, ox:ox + img.shape[1]] = img[:, :cw - ox]
    return canvas


def _raw_to_rot(x, y, deg):
    if deg == 90:
        return 1.0 - y, x
    if deg == 180:
        return 1.0 - x, 1.0 - y
    if deg == 270:
        return y, 1.0 - x
    return x, y


def _rot_to_raw(x, y, deg):
    if deg == 90:
        return y, 1.0 - x
    if deg == 180:
        return 1.0 - x, 1.0 - y
    if deg == 270:
        return 1.0 - y, x
    return x, y


def canvas_to_raw(pts, raw_wh, deg):
    """Normalised canvas points -> normalised camera-frame points."""
    deg = norm_deg(deg)
    rw, rh, cw, ch, s, ox = _geometry(raw_wh[0], raw_wh[1], deg)
    out = []
    for u, v in np.asarray(pts, np.float64).reshape(-1, 2):
        x = (u * cw - ox) / (rw * s)
        out.append(_rot_to_raw(x, v, deg))
    return np.asarray(out, np.float64)


def raw_to_canvas(pts, raw_wh, deg):
    """Normalised camera-frame points -> normalised canvas points."""
    deg = norm_deg(deg)
    rw, rh, cw, ch, s, ox = _geometry(raw_wh[0], raw_wh[1], deg)
    out = []
    for x0, y0 in np.asarray(pts, np.float64).reshape(-1, 2):
        x, y = _raw_to_rot(x0, y0, deg)
        out.append(((x * rw * s + ox) / cw, y))
    return np.asarray(out, np.float64)


def orientation_score(img):
    """-> (score 0-100, parts). Upright court: floor (blue/green, saturated) sits low and widens downward, and the
    net / service lines make horizontal edges dominate. Validated on match footage: upright ~88 vs <= 44 otherwise."""
    h, w = img.shape[:2]
    s = 320.0 / max(h, w)
    sm = cv2.resize(img, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    hh, ww = sm.shape[:2]
    hsv = cv2.cvtColor(sm, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    floor = (H >= 35) & (H <= 130) & (S > 70) & (V > 50)
    fy = np.nonzero(floor)[0]
    bottom = float(fy.mean() / hh) if fy.size > 50 else 0.5
    rows = floor.mean(axis=1)
    widen = float(np.corrcoef(np.arange(hh), rows)[0, 1]) if rows.std() > 1e-6 else 0.0
    edges = cv2.Canny(cv2.cvtColor(sm, cv2.COLOR_BGR2GRAY), 60, 160)
    segs = cv2.HoughLinesP(edges, 1, np.pi / 180, 30, minLineLength=int(0.15 * ww), maxLineGap=4)
    hl = vl = 1e-6
    if segs is not None:
        for x1, y1, x2, y2 in segs.reshape(-1, 4):
            L = float(np.hypot(x2 - x1, y2 - y1))
            a = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
            if a < 20 or a > 160:
                hl += L
            elif 70 < a < 110:
                vl += L
    horiz = hl / (hl + vl)
    score = 100.0 * (0.35 * horiz + 0.35 * bottom + 0.30 * (widen + 1.0) / 2.0)
    return round(score, 1), {"horizontal_lines": round(horiz, 2), "floor_low": round(bottom, 2),
                             "floor_widens": round(widen, 2), "floor_frac": round(float(floor.mean()), 2)}


def detect_orientation(raw, current=None, min_conf=MIN_CONF):
    """raw = un-rotated camera frame. current None -> 90 (clockwise) for portrait buffers, else 0.
    Keeps `current` when it scores >= min_conf, otherwise tests every rotation and returns the best.
    -> (deg, score, {deg: score})"""
    if current is None:
        current = 90 if raw.shape[0] > raw.shape[1] else 0
    current = norm_deg(current)
    scores = {current: orientation_score(landscape(raw, current))[0]}
    if scores[current] >= min_conf:
        return current, scores[current], scores
    for d in (current + 90, current + 270, current + 180):     # clockwise, counter-clockwise, flip
        d = norm_deg(d)
        scores[d] = orientation_score(landscape(raw, d))[0]
    best = max(scores, key=scores.get)
    return best, scores[best], scores
