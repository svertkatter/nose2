# nose_logic.py — しっかり笑った時だけ増える（キャリブ＋相対&絶対ゲート＋連続フレーム＋減衰）
import time
import math

# ---- スケール設定 ----
SCALE_MIN_BASE   = 2.0   # 最低倍率（初期）
HARD_MAX_SCALE   = 3.8   # ★最終上限（大きすぎるなら 3.6 などへ）

# ---- 笑顔ゲート（絶対/相対の両方を満たしたら候補ON）----
SMILE_ON_THRESH  = 0.25  # ★絶対ONしきい（上げるほど厳しい）
SMILE_OFF_THRESH = 0.18  # 絶対OFFしきい（ONより低く）
DELTA_ON         = 0.05  # ★中立（個人）よりどれだけ上がればONか
DELTA_OFF        = 0.01  # OFFの相対しきい（ONより低く）
K_SIGMA_ON       = 0.7   # ★ノイズσに対するONの係数（上げるほど厳しい）
K_SIGMA_OFF      = 0.1   # OFF側の係数

# ---- 平滑化・確定までの猶予 ----
S_EMA_ALPHA      = 0.25  # スコアEMA（0.2〜0.5）
BASELINE_ALPHA   = 0.05  # 中立EMA（遅めが◎）
NOISE_ALPHA      = 0.05  # ノイズ量EMA（|s-baseline| のEMA）
MIN_ON_FRAMES    = 7     # ★連続このフレーム数候補ONになって初めて本ON

# ---- 速度（fps非依存：秒ベース）----
ADD_PER_SEC_K    = 4.00  # ★笑っている間の増加係数（下げると伸びが穏やか）
DECAY_PER_SEC    = 0.60  # ★笑っていない間の減衰係数（上げると元に戻りやすい）

# ---- キャリブレーション ----
CALIB_SECS       = 2.0   # ★各人が見え始めてからこの秒数は絶対に増やさない

def _clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def compute_smile_score(pts):
    """
    0.0〜1.0の笑顔スコア（口の横幅/縦幅の比）。
    環境で高めに出るなら (ratio - 2.0)/3.8 などに調整してください。
    """
    try:
        xL, yL, _ = pts[61]; xR, yR, _ = pts[291]
        xU, yU, _ = pts[13]; xD, yD, _ = pts[14]
    except (IndexError, TypeError):
        return 0.0
    mouth_w = math.hypot(xR - xL, yR - yL)
    mouth_h = math.hypot(xD - xU, yD - yU)
    if mouth_w <= 1e-6 or mouth_h <= 1e-6:
        return 0.0
    ratio = mouth_w / mouth_h
    score = (ratio - 1.8) / 3.5
    return _clamp(score, 0.0, 1.0)

def compute_nose_base_size(pts):
    """基準サイズ：目外側(33,263)×0.45。フォールバックあり。"""
    try:
        xL, yL, _ = pts[33]; xR, yR, _ = pts[263]
    except (IndexError, TypeError):
        try:
            xL, yL, _ = pts[234]; xR, yR, _ = pts[454]
        except (IndexError, TypeError):
            return 120
    return max(40, int(math.hypot(xR - xL, yR - yL) * 0.45))

class NoseLogic:
    """
    update(landmarks_by_id, smile_by_id) -> {id: scale}
      - 出現直後 CALIB_SECS は学習のみ（絶対に増やさない）
      - 候補ON（絶対/相対）かつ MIN_ON_FRAMES 連続で本ONになり加算
      - ★ON中でも候補ONでなくなった瞬間から減衰（無表情で確実に小さくなる）
      - 倍率は [SCALE_MIN_BASE, HARD_MAX_SCALE] にクランプ
    """
    def __init__(self):
        self.scales     = {}   # id -> 現在倍率
        self.smile_ema  = {}   # id -> 平滑後スコア
        self.baseline   = {}   # id -> 中立EMA（笑っていない時のみ更新）
        self.noise_ema  = {}   # id -> ノイズ量EMA（|s-baseline|）
        self.on_frames  = {}   # id -> 連続候補ONフレーム数
        self.is_on      = {}   # id -> 今ONか（本ON）
        self.first_ts   = {}   # id -> 観測開始時刻
        self.last_ts    = {}   # id -> 前回更新時刻
        self.ids_live   = set()

    def _now(self): return time.monotonic()

    def _reset_people(self, ids_now):
        if ids_now != self.ids_live:
            now = self._now()
            self.scales    = {pid: self.scales.get(pid, SCALE_MIN_BASE) for pid in ids_now}
            self.smile_ema = {pid: 0.0 for pid in ids_now}
            self.baseline  = {pid: 0.0 for pid in ids_now}
            self.noise_ema = {pid: 0.0 for pid in ids_now}
            self.on_frames = {pid: 0   for pid in ids_now}
            self.is_on     = {pid: False for pid in ids_now}
            self.first_ts  = {pid: self.first_ts.get(pid, now) for pid in ids_now}
            self.last_ts   = {pid: now for pid in ids_now}
            self.ids_live  = set(ids_now)

    def _candidate_on(self, pid, s, base, sigma):
        # 絶対＆相対（中立+ノイズ×係数）を両方満たしたら候補ON
        thr_abs_on = SMILE_ON_THRESH
        thr_rel_on = base + max(DELTA_ON, K_SIGMA_ON * sigma)
        return (s >= thr_abs_on) and (s >= thr_rel_on)

    def _candidate_off(self, pid, s, base, sigma):
        thr_abs_off = SMILE_OFF_THRESH
        thr_rel_off = base + max(DELTA_OFF, K_SIGMA_OFF * sigma)
        return (s < thr_abs_off) or (s < thr_rel_off)

    def update(self, landmarks_by_id: dict, smile_by_id: dict) -> dict:
        ids = list(landmarks_by_id.keys())
        self._reset_people(set(ids))
        if not ids:
            return {}

        now = self._now()
        out = {}

        for pid in ids:
            # 経過時間
            prev_ts = self.last_ts.get(pid, now)
            dt = max(1/120.0, now - prev_ts)  # 極端な0除け
            self.last_ts[pid] = now

            # スコアEMA
            s_raw = float(smile_by_id.get(pid, 0.0))
            s_prev = self.smile_ema.get(pid, s_raw)
            s = s_prev * (1.0 - S_EMA_ALPHA) + s_raw * S_EMA_ALPHA
            self.smile_ema[pid] = s

            # キャリブ期間中は学習のみ（増やさない）
            first = self.first_ts.get(pid, now)
            in_calib = (now - first) < CALIB_SECS

            # 中立＆ノイズEMA更新（本ONでない時のみ更新して中立を保つ）
            if not self.is_on.get(pid, False):
                b_prev = self.baseline.get(pid, s)
                base = b_prev * (1.0 - BASELINE_ALPHA) + s * BASELINE_ALPHA
                self.baseline[pid] = base
                n_prev = self.noise_ema.get(pid, 0.0)
                self.noise_ema[pid] = n_prev * (1.0 - NOISE_ALPHA) + abs(s - base) * NOISE_ALPHA
            else:
                base = self.baseline.get(pid, 0.0)

            sigma = self.noise_ema.get(pid, 0.0)

            # 候補ON/OFF判定
            cand_on  = self._candidate_on(pid, s, base, sigma)
            cand_off = self._candidate_off(pid, s, base, sigma)

            if self.is_on.get(pid, False):
                if cand_off:
                    self.is_on[pid] = False
                    self.on_frames[pid] = 0
            else:
                if cand_on:
                    self.on_frames[pid] = self.on_frames.get(pid, 0) + 1
                    if self.on_frames[pid] >= MIN_ON_FRAMES and not in_calib:
                        self.is_on[pid] = True
                        self.on_frames[pid] = 0
                else:
                    self.on_frames[pid] = 0

            # スケール更新（★ここを修正）
            prev_scale = self.scales.get(pid, SCALE_MIN_BASE)

            if self.is_on.get(pid, False):
                # ★ON中でも cand_on を満たしていない間は“加算しない”で減衰する
                if cand_on:
                    thr_rel_on = base + max(DELTA_ON, K_SIGMA_ON * sigma)
                    s_eff = max(0.0, s - thr_rel_on)
                    new_scale = prev_scale + s_eff * ADD_PER_SEC_K * dt
                else:
                    new_scale = prev_scale - DECAY_PER_SEC * dt
            else:
                # 笑っていない：減衰
                new_scale = prev_scale - DECAY_PER_SEC * dt

            new_scale = _clamp(new_scale, SCALE_MIN_BASE, HARD_MAX_SCALE)
            self.scales[pid] = new_scale
            out[pid] = new_scale

        return out
