# main.py — 設定UI組み込み版（Camera/SwapSec/MaxScale/DebugをGUI調整＆保存）
import os
import cv2
import sys
import mediapipe as mp
import numpy as np
import pygame
import time
import random
from glob import glob
from centroid_tracker import CentroidTracker
from nose_logic import NoseLogic, compute_smile_score, compute_nose_base_size
from utils import overlay_image_alpha
from mediapipe.python.solutions.pose import PoseLandmark

# 追加：設定UIと保存/復元
from app_config import load_config, save_config
from settings_ui import SettingsUI

# ---- デバッグ互換ユーティリティ（落ちないように） ----
def _ensure_debug_info(nl):
    if not hasattr(nl, 'debug_mode'):
        try: nl.debug_mode = False
        except: pass
    if not hasattr(nl, 'debug_info'):
        try: nl.debug_info = {}
        except: pass
# ----------------------------------------------------

# ──────────────────────────────────────────────
# 設定の読み込み & UI 準備
# ──────────────────────────────────────────────
CFG = load_config()
ui = SettingsUI(CFG)  # 別ウィンドウでトラックバー表示

# ──────────────────────────────────────────────
# 定数・モデル初期化
# ──────────────────────────────────────────────
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

MIN_BODY_BOX_AREA           = 5000
VISIBILITY_THRESH           = 0.5
IOU_THRESH                  = 0.3
FALLBACK_MIN_AREA           = MIN_BODY_BOX_AREA

# 設定値から初期化
TWO_PERSON_SWITCH_INTERVAL  = float(CFG.get("swap_sec", 15.0))
MAX_SCALE_CLAMP             = float(CFG.get("max_scale", 4.5))
DEBUG_OVERLAY               = bool(int(CFG.get("debug_overlay", 1)))

prev_count = 0

# MediaPipe Pose（キーワードで）
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    smooth_segmentation=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# Face Detection & Mesh
mp_fd = mp.solutions.face_detection
fd_model = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_fm = mp.solutions.face_mesh
fm_model = mp_fm.FaceMesh(
    static_image_mode=False,
    max_num_faces=6,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# PyGame
pygame.mixer.init()
pygame.display.set_mode((1, 1), pygame.NOFRAME)
screen_w, screen_h = 1920, 1080

# サウンド
use_audio = True
try:
    sound_giggle = pygame.mixer.Sound("assets/laugh_giggle.wav")
    sound_chuckle = pygame.mixer.Sound("assets/laugh_chuckle.wav")
    sound_big     = pygame.mixer.Sound("assets/laugh_big.wav")
    for s in (sound_giggle, sound_chuckle, sound_big):
        s.play(loops=-1); s.set_volume(0.0)
except:
    print("Warning: 音声がロードできませんでした。")
    use_audio = False

def update_sound_volumes(smile_score):
    if not use_audio: return
    sound_giggle.set_volume(0.0)
    sound_chuckle.set_volume(0.0)
    sound_big.set_volume(0.0)
    if smile_score < 0.1:   sound_big.set_volume(1.0)
    elif smile_score < 0.25:sound_giggle.set_volume(1.0)
    else:                   sound_chuckle.set_volume(1.0)

# 鼻画像
nose_images, nose_alphas = [], []
for path in sorted(glob("assets/nose_*.png")):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.shape[2] == 4:
        nose_images.append(img[:, :, :3])
        nose_alphas.append(img[:, :, 3] / 255.0)
if not nose_images:
    print("Warning: 鼻画像が見つかりません。")

# トラッカー＆ロジック
ct         = CentroidTracker(max_disappeared=300)
nose_logic = NoseLogic()

# 割当ステート
assigned_id            = None
assigned_img_idx       = None
two_person_last_switch = None

# カメラ（設定から）
cam_index = int(CFG.get("camera_index", 2))
def open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"Webカメラ {index} を開けませんでした。")
        sys.exit(1)
    return cap

cap = open_camera(cam_index)

# フルスクリーン
cv2.namedWindow("Nose Mirror", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Nose Mirror", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

def bbox_iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[0]+a[2], b[0]+b[2]); yB = min(a[1]+a[3], b[1]+b[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interA = interW * interH
    union = a[2]*a[3] + b[2]*b[3] - interA
    return interA / union if union > 0 else 0

frame_count = 0
try:
    while True:
        # === 設定UIの反映（毎フレーム/軽い） ===
        new_cfg = ui.read()
        # カメラ切り替え
        if new_cfg["camera_index"] != cam_index:
            cam_index = int(new_cfg["camera_index"])
            cap.release()
            cap = open_camera(cam_index)
        # 他の設定
        TWO_PERSON_SWITCH_INTERVAL = float(new_cfg["swap_sec"])
        MAX_SCALE_CLAMP = float(new_cfg["max_scale"])
        DEBUG_OVERLAY = bool(int(new_cfg["debug_overlay"]))

        # === いつも通りの処理 ===
        ret, frame = cap.read()
        if not ret: break

        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Pose → pose_bbox
        pose_bbox = None
        pose_res = pose_model.process(frame_rgb)
        if pose_res.pose_landmarks:
            lm = pose_res.pose_landmarks.landmark
            key_ids = [PoseLandmark.LEFT_SHOULDER.value, PoseLandmark.RIGHT_SHOULDER.value,
                       PoseLandmark.LEFT_HIP.value, PoseLandmark.RIGHT_HIP.value]
            avg_vis = sum(lm[i].visibility for i in key_ids) / len(key_ids)
            if avg_vis > VISIBILITY_THRESH:
                coords = [(int(l.x*w), int(l.y*h)) for l in lm]
                xs, ys = zip(*coords)
                x0, x1 = max(min(xs), 0), min(max(xs), w)
                y0, y1 = max(min(ys), 0), min(max(ys), h)
                bw, bh = x1-x0, y1-y0
                if bw*bh >= MIN_BODY_BOX_AREA:
                    pose_bbox = (x0, y0, bw, bh)

        # 顔検出ボックス
        boxes = []
        face_res = fd_model.process(frame_rgb)
        if face_res.detections:
            for det in face_res.detections:
                bb = det.location_data.relative_bounding_box
                x1 = int(bb.xmin*w); y1 = int(bb.ymin*h)
                bw = int(bb.width*w); bh = int(bb.height*h)
                if bw*bh >= MIN_BODY_BOX_AREA:
                    boxes.append((x1, y1, bw, bh))

        # HOG補完
        if not boxes:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rects, _ = hog.detectMultiScale(gray, winStride=(8,8), padding=(16,16), scale=1.05)
            for x, y, bw, bh in rects:
                if bw*bh < MIN_BODY_BOX_AREA: continue
                if pose_bbox and bbox_iou((x,y,bw,bh), pose_bbox) > IOU_THRESH:
                    boxes.append((x, y, bw, bh))
                elif not pose_bbox and bw*bh >= FALLBACK_MIN_AREA:
                    boxes.append((x, y, bw, bh))

        # トラッカー
        objects = ct.update(boxes)

        # FaceMesh → ランドマーク / 笑顔
        fm_res = fm_model.process(frame_rgb)
        landmarks_by_id, smile_by_id = {}, {}
        if fm_res.multi_face_landmarks:
            for face_lms in fm_res.multi_face_landmarks:
                pts = [(int(lm.x*w), int(lm.y*h), lm.z) for lm in face_lms.landmark]
                nx, ny, _ = pts[1]
                best_id, min_d = None, float("inf")
                for oid, (cx, cy) in objects.items():
                    d = (nx-cx)**2 + (ny-cy)**2
                    if d < min_d: min_d, best_id = d, oid
                if best_id is not None:
                    landmarks_by_id[best_id] = pts
                    # ※ compute_smile_score は nose_logic.py 側の実装を使用
                    smile_by_id[best_id]     = compute_smile_score(pts)

        # 割当（元の流れ）
        cur_time = time.time()
        current_faces = list(landmarks_by_id.keys())

        if assigned_id is None:
            if current_faces:
                if len(current_faces) == 1:
                    assigned_id = current_faces[0]
                    two_person_last_switch = None
                else:
                    assigned_id = random.choice(current_faces)
                    two_person_last_switch = cur_time
                assigned_img_idx = random.randint(0, len(nose_images)-1)

        elif assigned_id not in current_faces:
            if not current_faces:
                assigned_id = None; assigned_img_idx = None
                two_person_last_switch = None
            elif len(current_faces) == 1:
                assigned_id = current_faces[0]
                assigned_img_idx = random.randint(0, len(nose_images)-1)
                two_person_last_switch = None
            else:
                assigned_id = random.choice(current_faces)
                assigned_img_idx = random.randint(0, len(nose_images)-1)
                two_person_last_switch = cur_time

        elif len(current_faces) == 2:
            if two_person_last_switch is None:
                two_person_last_switch = cur_time
            elif cur_time - two_person_last_switch >= TWO_PERSON_SWITCH_INTERVAL:
                other = [i for i in current_faces if i != assigned_id]
                if other: assigned_id = other[0]
                two_person_last_switch = cur_time

        # nose_logic で各人の倍率を更新
        nose_scales = nose_logic.update(landmarks_by_id, smile_by_id)

        # サウンド
        if smile_by_id:
            avg_smile = np.mean(list(smile_by_id.values()))
            update_sound_volumes(avg_smile)
        else:
            if use_audio:
                sound_giggle.set_volume(0.0)
                sound_chuckle.set_volume(0.0)
                sound_big.set_volume(0.0)

        # ---- デバッグ表示 ----
        if DEBUG_OVERLAY:
            _ensure_debug_info(nose_logic)
            if len(current_faces) == 2 and two_person_last_switch is not None:
                remaining = max(0.0, TWO_PERSON_SWITCH_INTERVAL - (cur_time - two_person_last_switch))
            else:
                # 1人時の残りは未使用（nose_logic にAPIがあれば利用）
                remaining = 0.0
            cv2.putText(frame, f"MODE: {len(current_faces)}人", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Flip in: {remaining:.1f}s", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

            # 簡易スコア・スケール
            panel_x, panel_y = 10, 100
            cv2.putText(frame, "Smile Debug:", (panel_x, panel_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
            y = panel_y + 28
            for pid in sorted(landmarks_by_id.keys()):
                s_val = float(smile_by_id.get(pid, 0.0))
                sc    = float(nose_scales.get(pid, 0.0)) if nose_scales else 0.0
                mark  = "*" if pid == assigned_id else " "
                cv2.putText(frame, f"{mark}ID {pid}: s={s_val:.3f} sc={sc:.2f}",
                            (panel_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,200), 2)
                y += 22

        # ---- 鼻オーバーレイ（元の参照方法のまま）----
        if assigned_id in landmarks_by_id and nose_images:
            pts = landmarks_by_id[assigned_id]
            x_n, y_n, _ = pts[1]
            base = compute_nose_base_size(pts)

            # 1人：自分、2人：相手、3人以上：相手の最大
            scale = nose_scales.get(assigned_id, 3.0)
            if len(landmarks_by_id) == 2:
                other_id = next(i for i in landmarks_by_id.keys() if i != assigned_id)
                scale = nose_scales.get(other_id, 3.0)
            elif len(landmarks_by_id) >= 3:
                others = [i for i in landmarks_by_id.keys() if i != assigned_id]
                scale = max(nose_scales.get(i, 3.0) for i in others)

            # ★ UIの上限クランプを適用
            scale = min(MAX_SCALE_CLAMP, float(scale))

            size  = max(8, int(base * float(scale)))

            img   = nose_images[assigned_img_idx]
            alpha = nose_alphas[assigned_img_idx]
            rgb_r = cv2.resize(img,   (size, size))
            a_r   = cv2.resize(alpha, (size, size))
            tx = int(x_n - size/2); ty = int(y_n - size*0.7)
            overlay_image_alpha(frame, rgb_r, (tx, ty), a_r)

        # フルスクリーン
        fh, fw = frame.shape[:2]
        fa, sa = fw/fh, screen_w/screen_h
        if fa > sa: nw, nh = screen_w, int(screen_w/fa)
        else:       nh, nw = screen_h, int(screen_h*fa)
        rf = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        ox = (screen_w - nw)//2; oy = (screen_h - nh)//2
        canvas[oy:oy+nh, ox:ox+nw] = rf

        cv2.imshow("Nose Mirror", canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    # ループ終了
finally:
    # 最終設定を保存
    final_cfg = ui.read()
    save_config(final_cfg)
    cap.release()
    cv2.destroyAllWindows()
