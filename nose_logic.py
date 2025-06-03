# nose_logic.py

import numpy as np
import time

def compute_smile_score(landmarks):
    """
    landmarks: 468 点の face mesh (x_px, y_px, z_rel) のリスト
    return: smile_score (0.0 ～ 1.0)
    """
    # 頬の左右: index 234, 454
    x_cl, y_cl, _ = landmarks[234]
    x_cr, y_cr, _ = landmarks[454]
    face_width = np.linalg.norm([x_cr - x_cl, y_cr - y_cl])

    # 口角: index 61, 291
    x_l, y_l, _ = landmarks[61]
    x_r, y_r, _ = landmarks[291]
    mouth_width = np.linalg.norm([x_r - x_l, y_r - y_l])

    if face_width > 0:
        score = mouth_width / face_width
    else:
        score = 0.0

    return np.clip(score, 0.0, 1.0)

def compute_nose_base_size(landmarks):
    """
    landmarks: 468 点の face mesh (x_px, y_px, z_rel) のリスト
    return: nose_width (最終的にリサイズする鼻画像のベース幅 pixel)
    """
    # 頬の左右を基準に顔幅を算出
    x_cl, y_cl, _ = landmarks[234]
    x_cr, y_cr, _ = landmarks[454]
    face_width = np.linalg.norm([x_cr - x_cl, y_cr - y_cl])
    # 顔幅の 20% を鼻ベース幅とする
    nose_width = int(face_width * 0.2)
    return max(nose_width, 1)

class NoseLogic:
    def __init__(self):
        self.one_enter_time = None       # 一人モード開始時刻
        self.scales = {}                 # ID -> 現在の鼻スケール
        self.invert_phase = False        # 90秒反転フラグ
        self.last_flip_time = time.time()

    def update(self, landmarks_by_id, smile_scores_by_id):
        """
        landmarks_by_id: { ID: [(x0,y0,z0), ... (x467,y467,z467)] }
        smile_scores_by_id: { ID: smile_score }
        return: { ID: nose_scale } （1.0 ～ 3.0）
        """
        cur_time = time.time()
        # 90秒反転トリガー
        if cur_time - self.last_flip_time >= 90.0:
            self.invert_phase = not self.invert_phase
            self.last_flip_time = cur_time
            self.one_enter_time = None
            self.scales = {}

        ids = list(landmarks_by_id.keys())
        num_faces = len(ids)
        nose_scales = {}

        if num_faces <= 1:
            # 一人モード
            if self.one_enter_time is None:
                self.one_enter_time = cur_time
            elapsed = cur_time - self.one_enter_time
            if elapsed >= 3.0:
                t = min((elapsed - 3.0) / 90.0, 1.0)
                base_scale = 1.0 + t * 2.0  # 1.0→3.0
            else:
                base_scale = 1.0

            if ids:
                id0 = ids[0]
                s = smile_scores_by_id.get(id0, 0.0)
                if s > 0.1:
                    base_scale = min(base_scale + 0.02, 3.0)

            if ids:
                nose_scales[ids[0]] = base_scale

        elif num_faces == 2:
            # 二人モード：バウンディングボックス面積で前後を判定
            areas = []
            for i in ids:
                lm = landmarks_by_id[i]
                xs = [p[0] for p in lm]
                ys = [p[1] for p in lm]
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                areas.append((i, area))
            areas_sorted = sorted(areas, key=lambda x: x[1], reverse=True)
            front_id = areas_sorted[0][0]
            back_id  = areas_sorted[1][0]

            front_scale = self.scales.get(front_id, 1.0)
            back_smile = smile_scores_by_id.get(back_id, 0.0)
            if back_smile > 0.1:
                front_scale = min(front_scale + back_smile * 0.1, 3.0)
            else:
                front_scale = max(front_scale - 0.01, 1.0)

            self.scales[front_id] = front_scale
            nose_scales[front_id] = front_scale
            nose_scales[back_id] = 1.0

        else:
            # 多人数モード (3人以上)
            areas = []
            for i in ids:
                lm = landmarks_by_id[i]
                xs = [p[0] for p in lm]
                ys = [p[1] for p in lm]
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                areas.append((i, area))
            areas_sorted = sorted(areas, key=lambda x: x[1], reverse=True)
            target_id = areas_sorted[0][0]
            laugher_ids = [x[0] for x in areas_sorted[1:]]

            if laugher_ids:
                avg_smile = sum(smile_scores_by_id.get(i, 0.0) for i in laugher_ids) / len(laugher_ids)
            else:
                avg_smile = 0.0

            target_scale = self.scales.get(target_id, 1.0)
            if avg_smile > 0.1:
                target_scale = min(target_scale + avg_smile * 0.15, 3.0)
            else:
                target_scale = max(target_scale - 0.01, 1.0)

            self.scales[target_id] = target_scale
            nose_scales[target_id] = target_scale

            for i in laugher_ids:
                nose_scales[i] = 1.0

        # 反転フェーズが True のときは、ここにモードを反転するロジックを追加可

        return nose_scales
