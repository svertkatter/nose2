import os
import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
import random
from glob import glob
from centroid_tracker import CentroidTracker
from nose_logic import NoseLogic, compute_smile_score, compute_nose_base_size
from utils import overlay_image_alpha

# ------------------------------
# 1) MediaPipe 初期化
# ------------------------------
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

fd_model = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
fm_model = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=6,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------------------
# 2) Pygame 初期化 + スクリーンサイズ取得
# ------------------------------
pygame.mixer.init()
pygame.display.set_mode((1, 1), pygame.NOFRAME)
info = pygame.display.Info()
# screen_w, screen_h = info.current_w, info.current_h
screen_w, screen_h = 1920, 1080

# ------------------------------
# 3) 音声読み込み
# ------------------------------
use_audio = True
try:
    sound_giggle = pygame.mixer.Sound("assets/laugh_giggle.wav")
    sound_chuckle = pygame.mixer.Sound("assets/laugh_chuckle.wav")
    sound_big = pygame.mixer.Sound("assets/laugh_big.wav")
    for s in [sound_giggle, sound_chuckle, sound_big]:
        s.play(loops=-1)
        s.set_volume(0.0)
except:
    print("Warning: 音声がロードできなかったため、スキップします。")
    use_audio = False

def update_sound_volumes(smile_score):
    if not use_audio: return
    sound_giggle.set_volume(0.0)
    sound_chuckle.set_volume(0.0)
    sound_big.set_volume(0.0)
    if smile_score < 0.1:
        sound_big.set_volume(1.0)
    elif smile_score < 0.25:
        sound_giggle.set_volume(1.0)
    else:
        sound_chuckle.set_volume(1.0)

# ------------------------------
# 4) 鼻画像読み込み（複数）
# ------------------------------
nose_images = []
nose_alphas = []
nose_paths = sorted(glob("assets/nose_*.png"))

for path in nose_paths:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.shape[2] == 4:
        nose_images.append(img[:, :, :3])
        nose_alphas.append(img[:, :, 3] / 255.0)

if not nose_images:
    print("Warning: 鼻画像が見つかりません。鼻オーバーレイはスキップされます。")

# ------------------------------
# 5) その他初期化
# ------------------------------
ct = CentroidTracker(max_disappeared=50)
nose_logic = NoseLogic()
nose_image_by_id = {}

# ------------------------------
# 6) カメラ開始
# ------------------------------
cap = cv2.VideoCapture(2, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Webカメラを開けませんでした。")
    exit(1)

# ------------------------------
# 7) ウィンドウ設定（フルスクリーン）
# ------------------------------
cv2.namedWindow("Nose Mirror", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Nose Mirror", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ------------------------------
# 8) メインループ
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 顔検出
    fd_results = fd_model.process(frame_rgb)
    face_boxes = []
    if fd_results.detections:
        for det in fd_results.detections:
            rb = det.location_data.relative_bounding_box
            x = int(rb.xmin * w)
            y = int(rb.ymin * h)
            bw = int(rb.width * w)
            bh = int(rb.height * h)
            x = max(x, 0)
            y = max(y, 0)
            bw = min(bw, w - x)
            bh = min(bh, h - y)
            face_boxes.append((x, y, bw, bh))

    # トラッキング
    objects = ct.update(face_boxes)
    for objectID in objects:
        if objectID not in nose_image_by_id and nose_images:
            idx = random.randint(0, len(nose_images) - 1)
            nose_image_by_id[objectID] = (nose_images[idx], nose_alphas[idx])

    # ランドマーク取得
    fm_results = fm_model.process(frame_rgb)
    landmarks_by_id = {}
    smile_by_id = {}

    if fm_results.multi_face_landmarks:
        for face_lms in fm_results.multi_face_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in face_lms.landmark]
            nose_tip = pts[1]
            lx, ly, _ = nose_tip
            min_dist = float('inf')
            matched_id = None
            for objectID, (cx, cy) in objects.items():
                d = np.hypot(lx - cx, ly - cy)
                if d < min_dist:
                    min_dist = d
                    matched_id = objectID
            if matched_id is not None:
                landmarks_by_id[matched_id] = pts
                smile_by_id[matched_id] = compute_smile_score(pts)

    # 鼻スケール計算
    nose_scales = nose_logic.update(landmarks_by_id, smile_by_id)

    # 音声更新
    # avg_smile = np.mean(list(smile_by_id.values())) if smile_by_id else 0.0
    # update_sound_volumes(avg_smile)
    if smile_by_id:
        avg_smile = np.mean(list(smile_by_id.values()))
        update_sound_volumes(avg_smile)
    else:
        sound_giggle.set_volume(0.0)
        sound_chuckle.set_volume(0.0)
        sound_big.set_volume(0.0)

    # 鼻描画
    for ID, pts in landmarks_by_id.items():
        if ID not in nose_image_by_id: continue
        x_n, y_n, _ = pts[1]
        base_size = compute_nose_base_size(pts)
        scale = nose_scales.get(ID, 1.0)
        final_size = int(base_size * scale)

        img_bgr, alpha_mask = nose_image_by_id[ID]
        resized_bgr = cv2.resize(img_bgr, (final_size, final_size))
        resized_alpha = cv2.resize(alpha_mask, (final_size, final_size))

        top_left_x = int(x_n - final_size / 2)
        top_left_y = int(y_n - final_size * 0.7)
        overlay_image_alpha(frame, resized_bgr, (top_left_x, top_left_y), resized_alpha)

    # smile_score を表示（デバッグ用）
    # for ID, score in smile_by_id.items():
    #     cx, cy = objects[ID]
    #     cv2.putText(frame, f"smile: {score:.2f}", (cx, cy - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # アスペクト比を保って黒背景に描画
    frame_h, frame_w = frame.shape[:2]
    frame_aspect = frame_w / frame_h
    screen_aspect = screen_w / screen_h

    if frame_aspect > screen_aspect:
        new_w = screen_w
        new_h = int(screen_w / frame_aspect)
    else:
        new_h = screen_h
        new_w = int(screen_h * frame_aspect)

    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x_offset = (screen_w - new_w) // 2
    y_offset = (screen_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame

    # ウィンドウ表示
    cv2.imshow("Nose Mirror", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
