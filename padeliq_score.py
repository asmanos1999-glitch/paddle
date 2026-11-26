"""PadelIQ live padel score engine (shared by the desktop HUD and the web command center).

Official padel game scoring 0 -> 15 -> 30 -> 40 -> GAME, games and total points per team, with automatic point
detection from the ball tracker plus instant manual overrides ([1] / [2] / [0] and the web buttons).

Teams follow the player slots: Team A = P1 / P2 (near half, court y >= 10 m), Team B = P3 / P4 (far half).

Automatic rally logic (single camera, so it is a heuristic - the presenter can always correct with [1] / [2]):
  * a rally becomes active after the first shot or the first confirmed net crossing;
  * `owner` = the team that has to play the ball next: the ball entering a half hands it to that half's team,
    a shot from half X hands it to the other team, and a ball that stays on the hitter's own side (net) returns
    it to the hitter;
  * if the owner does not get the ball back across the net within RETURN_TIMEOUT_S (no crossing, no shot:
    ball stopped, lost, out or into the net), the point goes to the other team;
  * after a point the engine waits COOLDOWN_S before it watches the next rally.
"""
import threading
import time

NET_Y_M = 10.0
POINT_NAMES = ("0", "15", "30", "40")
TEAMS = ("A", "B")
TEAM_LABEL = {"A": "TEAM A", "B": "TEAM B"}
TEAM_PLAYERS = {"A": "P1-P2", "B": "P3-P4"}


def other(team):
    return "B" if team == "A" else "A"


def half_of(court_y, hysteresis=0.4):
    """'A' near half (y >= 10 m), 'B' far half, None inside the +-hysteresis band around the net."""
    if court_y is None:
        return None
    if court_y >= NET_Y_M + hysteresis:
        return "A"
    if court_y <= NET_Y_M - hysteresis:
        return "B"
    return None


class PadelScoreEngine:
    RETURN_TIMEOUT_S = 2.2
    BOUNCE_BACK_S = 1.0             # ball still on the hitter's side this long after the shot -> hitter's ball
    COOLDOWN_S = 3.5
    HALF_CONFIRM = 3                # consecutive visible samples needed to accept a half change
    MIN_VISIBLE = 6                 # visible samples a rally needs before it can produce an automatic point
    FLASH_S = 2.6

    def __init__(self, golden_point=True, auto=True):
        self.golden_point = bool(golden_point)
        self.auto = bool(auto)
        self.lock = threading.RLock()
        self.listeners = []         # fn(event dict) on every point (toasts, coach feed)
        self.reset()

    # ------------------------------------------------------------------ scoring
    def reset(self):
        with self.lock:
            self.points = {"A": 0, "B": 0}          # index into POINT_NAMES (4 = advantage when not golden point)
            self.games = {"A": 0, "B": 0}
            self.total = {"A": 0, "B": 0}
            self.history = []                        # snapshots for undo
            self.last_point = None                   # {"team", "t", "reason", "auto"}
            self._reset_rally(time.time(), cooldown=False)

    def _snapshot(self):
        return ({**self.points}, {**self.games}, {**self.total}, self.last_point)

    def point(self, team, reason="manual", now=None, auto=False):
        """Award one point to `team` ('A' / 'B'). Returns the event dict."""
        team = str(team).upper()
        if team not in TEAMS:
            raise ValueError("team must be 'A' or 'B'")
        now = time.time() if now is None else float(now)
        with self.lock:
            self.history.append(self._snapshot())
            del self.history[:-50]
            opp = other(team)
            self.total[team] += 1
            game = False
            p, q = self.points[team], self.points[opp]
            if p <= 2:
                self.points[team] = p + 1
            elif self.golden_point:
                game = True                          # 40 wins the game (golden point at 40 : 40)
            elif p == 3 and q < 3:
                game = True
            elif p == 3 and q == 3:
                self.points[team] = 4                # advantage
            elif p == 3 and q == 4:
                self.points[opp] = 3                 # back to deuce
            else:                                    # p == 4 (advantage) -> game
                game = True
            if game:
                self.games[team] += 1
                self.points = {"A": 0, "B": 0}
            rally = self.rally_shots
            self.last_point = {"team": team, "t": now, "reason": reason, "auto": bool(auto), "game": game,
                               "rally_shots": rally}
            self._reset_rally(now, cooldown=True)
            ev = dict(self.last_point, score=self.state(now))
        for fn in list(self.listeners):
            try:
                fn(ev)
            except Exception:
                pass
        return ev

    def undo(self):
        with self.lock:
            if not self.history:
                return False
            self.points, self.games, self.total, self.last_point = self.history.pop()
            self._reset_rally(time.time(), cooldown=True)
            return True

    def point_label(self, team):
        p, q = self.points[team], self.points[other(team)]
        if p == 4:
            return "AD"
        if q == 4:
            return "40"
        return POINT_NAMES[min(p, 3)]

    # ------------------------------------------------------------------ automatic rally tracking
    def _reset_rally(self, now, cooldown):
        self.rally_active = False
        self.rally_shots = 0
        self.crossings = 0
        self.visible_samples = 0
        self.owner = None
        self.owner_since = now
        self.half = None
        self._cand_half, self._cand_n = None, 0
        self._last_total_shots = None
        self.cooldown_until = now + self.COOLDOWN_S if cooldown else 0.0

    def update(self, now, visible, court_pos, total_shots):
        """Feed one ball-tracker sample (call at display / telemetry rate). Returns a point event or None."""
        now = float(now)
        team_won = None
        with self.lock:
            if self._last_total_shots is None:
                self._last_total_shots = total_shots
            new_shots = max(0, int(total_shots) - int(self._last_total_shots))
            self._last_total_shots = total_shots
            if now < self.cooldown_until:
                return None
            h = half_of(court_pos[1]) if (visible and court_pos is not None) else None
            if visible and court_pos is not None:
                self.visible_samples += 1
            # stable half with hysteresis + N-sample confirmation (single noisy detections never flip sides)
            if h is not None:
                if h == self._cand_half:
                    self._cand_n += 1
                else:
                    self._cand_half, self._cand_n = h, 1
                if self._cand_n >= self.HALF_CONFIRM and h != self.half:
                    if self.half is not None:
                        self.crossings += 1
                    self.half = h
                    self.owner, self.owner_since = h, now          # ball arrived: this team must play it
            if new_shots:
                self.rally_shots += new_shots
                hitter = h or self.half
                if hitter is not None:
                    self.owner, self.owner_since = other(hitter), now
            elif (h is not None and self.owner is not None and h != self.owner
                  and now - self.owner_since > self.BOUNCE_BACK_S):
                self.owner, self.owner_since = h, now              # ball stayed on the hitter's side (net)
            if not self.rally_active and (self.rally_shots >= 1 or self.crossings >= 1):
                self.rally_active = True
            if (self.auto and self.rally_active and self.owner is not None
                    and self.visible_samples >= self.MIN_VISIBLE
                    and now - self.owner_since > self.RETURN_TIMEOUT_S):
                team_won = other(self.owner)
                reason = (f"no return from {TEAM_LABEL[self.owner]} within {self.RETURN_TIMEOUT_S:.1f}s "
                          f"(rally {self.rally_shots} shots)")
        if team_won is not None:
            return self.point(team_won, reason=reason, now=now, auto=True)
        return None

    def update_from_ball(self, ball, now=None):
        """Convenience for `PadelBallTracker` instances (reads the live attributes, no copy)."""
        now = time.time() if now is None else now
        cp = ball.court_pos if ball.visible else None
        return self.update(now, bool(ball.visible), None if cp is None else (float(cp[0]), float(cp[1])),
                           int(ball.total_shots))

    # ------------------------------------------------------------------ views
    def flash(self, now=None):
        """The last point while its on-screen badge should show, else None."""
        now = time.time() if now is None else now
        lp = self.last_point
        if lp and 0 <= now - lp["t"] < self.FLASH_S:
            return lp
        return None

    def state(self, now=None):
        now = time.time() if now is None else now
        with self.lock:
            return {
                "points": {t: self.point_label(t) for t in TEAMS},
                "games": dict(self.games),
                "total": dict(self.total),
                "rally_active": bool(self.rally_active),
                "rally_shots": int(self.rally_shots),
                "owner": self.owner,
                "auto": self.auto,
                "golden_point": self.golden_point,
                "teams": {t: {"label": TEAM_LABEL[t], "players": TEAM_PLAYERS[t]} for t in TEAMS},
                "last_point": None if self.last_point is None else {
                    k: (round(v, 3) if k == "t" else v) for k, v in self.last_point.items()},
                "flash": self.flash(now) is not None,
                "can_undo": bool(self.history),
            }

    def banner_text(self):
        st = self.state()
        return (f"{TEAM_LABEL['A']} [{TEAM_PLAYERS['A']}]  {st['points']['A']} : {st['points']['B']}  "
                f"{TEAM_LABEL['B']} [{TEAM_PLAYERS['B']}] | Games {st['games']['A']}-{st['games']['B']} | "
                f"Rally: {st['rally_shots']} shots")
