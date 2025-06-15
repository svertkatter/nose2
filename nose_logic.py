# nose_logic.py

import numpy as np
import time

def compute_smile_score(landmarks):
    """
    改良版：口角の上下位置の変化も考慮
    """
    x_cl, y_cl, _ = landmarks[234]
    x_cr, y_cr, _ = landmarks[454]
    face_width = np.linalg.norm([x_cr - x_cl, y_cr - y_cl])

    x_l, y_l, _ = landmarks[61]
    x_r, y_r, _ = landmarks[291]
    mouth_width = np.linalg.norm([x_r - x_l, y_r - y_l])

    _, y_top, _ = landmarks[13]
    _, y_bot, _ = landmarks[14]
    mouth_open = abs(y_bot - y_top)

    _, y_lc, _ = landmarks[61]
    _, y_rc, _ = landmarks[291]
    _, y_lcheek, _ = landmarks[234]
    _, y_rcheek, _ = landmarks[454]
    lift_left  = y_lcheek - y_lc
    lift_right = y_rcheek - y_rc

    score = (mouth_width / face_width) + (lift_left + lift_right) / (2 * face_width) + (mouth_open / face_width)
    score *= 0.5
    return np.clip(score, 0.0, 1.0)

def compute_nose_base_size(landmarks):
    x_cl, y_cl, _ = landmarks[234]
    x_cr, y_cr, _ = landmarks[454]
    face_width = np.linalg.norm([x_cr - x_cl, y_cr - y_cl])
    nose_width = int(face_width * 0.2)
    return max(nose_width, 1)

class NoseLogic:
    def __init__(self):
        self.one_enter_time = None
        self.scales = {}  # ID -> 最大スケール
        self.invert_phase = False
        self.last_flip_time = time.time()

    def update(self, landmarks_by_id, smile_scores_by_id):
        cur_time = time.time()
        if cur_time - self.last_flip_time >= 90.0:
            self.invert_phase = not self.invert_phase
            self.last_flip_time = cur_time
            self.one_enter_time = None
            self.scales = {}

        ids = list(landmarks_by_id.keys())
        num_faces = len(ids)
        nose_scales = {}

        if num_faces <= 1:
            if self.one_enter_time is None:
                self.one_enter_time = cur_time
            elapsed = cur_time - self.one_enter_time
            if elapsed >= 3.0:
                t = min((elapsed - 3.0) / 90.0, 1.0)
                base_scale = 3.0 + t * 5.0  # 最小3.0 → 最大8.0
            else:
                base_scale = 3.0

            if ids:
                id0 = ids[0]
                s = smile_scores_by_id.get(id0, 0.0)
                prev = self.scales.get(id0, base_scale)
                if s > 0.05: # 笑顔のしきい値
                    updated = min(prev + s * 0.5, 8.0)
                    self.scales[id0] = max(prev, updated)
                else:
                    self.scales[id0] = max(prev, base_scale)
                nose_scales[id0] = self.scales[id0]

        elif num_faces == 2:
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

            prev = self.scales.get(front_id, 3.0)
            back_smile = smile_scores_by_id.get(back_id, 0.0)
            if back_smile > 0.05: #笑顔のしきい値
                updated = min(prev + back_smile * 0.4, 8.0)
                self.scales[front_id] = max(prev, updated)
            else:
                self.scales[front_id] = max(prev, 3.0)

            nose_scales[front_id] = self.scales[front_id]
            nose_scales[back_id] = 1.0

        else:
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

            prev = self.scales.get(target_id, 3.0)
            if avg_smile > 0.05: #笑顔のしきい値
                updated = min(prev + avg_smile * 0.4, 8.0)
                self.scales[target_id] = max(prev, updated)
            else:
                self.scales[target_id] = max(prev, 3.0)

            nose_scales[target_id] = self.scales[target_id]
            for i in laugher_ids:
                nose_scales[i] = 1.0

        return nose_scales
