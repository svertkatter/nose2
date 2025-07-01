import numpy as np
import time


def compute_smile_score(landmarks):
    """
    改良版：口角の上下位置の変化も考慮
    """
    # 顔幅 (ランドマーク 234 - 454)
    x_cl, y_cl, _ = landmarks[234]
    x_cr, y_cr, _ = landmarks[454]
    face_width = np.linalg.norm([x_cr - x_cl, y_cr - y_cl])

    # 口の横幅 (ランドマーク 61 - 291)
    x_l, y_l, _ = landmarks[61]
    x_r, y_r, _ = landmarks[291]
    mouth_width = np.linalg.norm([x_r - x_l, y_r - y_l])

    # 口の開閉 (ランドマーク 13 - 14)
    _, y_top, _ = landmarks[13]
    _, y_bot, _ = landmarks[14]
    mouth_open = abs(y_bot - y_top)

    # 頬の持ち上がり (234,61) と (454,291)
    _, y_lc, _ = landmarks[61]
    _, y_rc, _ = landmarks[291]
    _, y_lcheek, _ = landmarks[234]
    _, y_rcheek, _ = landmarks[454]
    lift_left  = y_lcheek - y_lc
    lift_right = y_rcheek - y_rc

    score = mouth_width / face_width
    score += (lift_left + lift_right) / (2 * face_width)
    score += mouth_open / face_width
    score *= 0.5
    return np.clip(score, 0.0, 1.0)


def compute_nose_base_size(landmarks):
    """
    顔幅の20%を基本サイズとする
    """
    x_cl, y_cl, _ = landmarks[234]
    x_cr, y_cr, _ = landmarks[454]
    face_width = np.linalg.norm([x_cr - x_cl, y_cr - y_cl])
    base = int(face_width * 0.2)
    return max(base, 1)


class NoseLogic:
    """
    鼻スケール算出ロジック
    - 一人モード：時間経過で徐々にスケールアップ
    - 二人モード：前後どちらも自分の笑顔に応じて変形
                   90秒ごとに入れ替えフェーズ
    - 多人数モード：前の人または全員変形フェーズ
    """
    FLIP_INTERVAL = 90.0
    ONE_DELAY      =  3.0
    BASE_SCALE     =  3.0
    MAX_SCALE      =  8.0

    def __init__(self):
        self.last_flip_time = time.time()
        self.invert_phase   = False
        self.one_enter_time = None
        self.scales         = {}

    def update(self, landmarks_by_id, smiles_by_id):
        now = time.time()
        if now - self.last_flip_time >= self.FLIP_INTERVAL:
            self.invert_phase   = not self.invert_phase
            self.last_flip_time = now
            self.one_enter_time = None
            self.scales.clear()

        count = len(landmarks_by_id)
        if count <= 1:
            return self._one_mode(landmarks_by_id, smiles_by_id, now)
        if count == 2:
            return self._two_mode(landmarks_by_id, smiles_by_id)
        return self._multi_mode(landmarks_by_id, smiles_by_id)

    def _one_mode(self, landmarks, smiles, now):
        if self.one_enter_time is None:
            self.one_enter_time = now
        elapsed = now - self.one_enter_time
        if elapsed >= self.ONE_DELAY:
            t = min((elapsed - self.ONE_DELAY) / self.FLIP_INTERVAL, 1.0)
            base = self.BASE_SCALE + t * 5.0
        else:
            base = self.BASE_SCALE

        scales = {}
        if landmarks:
            i = next(iter(landmarks))
            prev  = self.scales.get(i, base)
            smile = smiles.get(i, 0.0)
            if smile > 0.05:
                updated = min(prev + smile * 0.5, self.MAX_SCALE)
                self.scales[i] = max(prev, updated)
            else:
                self.scales[i] = max(prev, base)
            scales[i] = self.scales[i]
        return scales

    def _two_mode(self, landmarks, smiles):
        # 前後判定（面積ベース）
        areas = []
        for i, lm in landmarks.items():
            xs = [p[0] for p in lm]; ys = [p[1] for p in lm]
            areas.append((i, (max(xs)-min(xs))*(max(ys)-min(ys))))
        areas.sort(key=lambda x: x[1], reverse=True)
        front, back = areas[0][0], areas[1][0]

        scales = {}
        # 通常フェーズ：前後どちらも自分の笑顔に応じて変形
        if not self.invert_phase:
            for i in (front, back):
                prev  = self.scales.get(i, self.BASE_SCALE)
                smile = smiles.get(i, 0.0)
                if smile > 0.05:
                    updated = min(prev + smile * 0.4, self.MAX_SCALE)
                    self.scales[i] = max(prev, updated)
                else:
                    self.scales[i] = max(prev, self.BASE_SCALE)
                scales[i] = self.scales[i]
        else:
            # 反転フェーズ：後ろだけ変形、前はノーマル
            prev  = self.scales.get(back, self.BASE_SCALE)
            smile = smiles.get(back, 0.0)
            if smile > 0.05:
                updated = min(prev + smile * 0.4, self.MAX_SCALE)
                self.scales[back] = max(prev, updated)
            else:
                self.scales[back] = max(prev, self.BASE_SCALE)
            scales[back]  = self.scales[back]
            scales[front] = 1.0
        return scales

    def _multi_mode(self, landmarks, smiles):
        ids = list(landmarks.keys())
        areas = []
        for i in ids:
            xs = [p[0] for p in landmarks[i]]; ys = [p[1] for p in landmarks[i]]
            areas.append((i, (max(xs)-min(xs))*(max(ys)-min(ys))))
        areas.sort(key=lambda x: x[1], reverse=True)
        front = areas[0][0]

        scales = {}
        if not self.invert_phase:
            others = [i for i in ids if i != front]
            avg_smile = sum(smiles.get(j,0.0) for j in others)/len(others) if others else 0.0
            prev = self.scales.get(front, self.BASE_SCALE)
            if avg_smile > 0.05:
                updated = min(prev + avg_smile * 0.4, self.MAX_SCALE)
                self.scales[front] = max(prev, updated)
            else:
                self.scales[front] = max(prev, self.BASE_SCALE)
            scales[front] = self.scales[front]
            for i in others:
                scales[i] = 1.0
        else:
            for i in ids:
                others = [j for j in ids if j != i]
                avg_smile = sum(smiles.get(j,0.0) for j in others)/len(others) if others else 0.0
                prev = self.scales.get(i, self.BASE_SCALE)
                if avg_smile > 0.05:
                    updated = min(prev + avg_smile * 0.4, self.MAX_SCALE)
                    self.scales[i] = max(prev, updated)
                else:
                    self.scales[i] = max(prev, self.BASE_SCALE)
                scales[i] = self.scales[i]
        return scales
