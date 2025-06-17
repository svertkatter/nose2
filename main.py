# main.py

import os
import cv2
import mediapipe as mp
import numpy as np
import pygame
import time
from centroid_tracker import CentroidTracker
from nose_logic import NoseLogic, compute_smile_score, compute_nose_base_size
from utils import overlay_image_alpha

# ------------------------------
# 1) MediaPipe セットアップ
# ------------------------------
mp_face_detection = mp.solutions.face_detection
mp_face_mesh      = mp.solutions.face_mesh

fd_model = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5
)

fm_model = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=6,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ------------------------------
# 2) Pygame (音声) 初期化 (存在チェック付き)
# ------------------------------
# assets に WAV ファイルがあればロード、なければスキップ
use_audio = True
pygame.mixer.init()
sound_giggle  = None
sound_chuckle = None
sound_big     = None

giggle_path  = "assets/laugh_giggle.wav"
chuckle_path = "assets/laugh_chuckle.wav"
big_path     = "assets/laugh_big.wav"

if os.path.isfile(giggle_path) and os.path.isfile(chuckle_path) and os.path.isfile(big_path):
    try:
        sound_giggle  = pygame.mixer.Sound(giggle_path)
        sound_chuckle = pygame.mixer.Sound(chuckle_path)
        sound_big     = pygame.mixer.Sound(big_path)
        sound_giggle.play(loops=-1)
        sound_chuckle.play(loops=-1)
        sound_big.play(loops=-1)
        sound_giggle.set_volume(0.0)
        sound_chuckle.set_volume(0.0)
        sound_big.set_volume(0.0)
    except Exception as e:
        print(f"Warning: 音声ファイルがロードできませんでした: {e}")
        use_audio = False
else:
    print("Warning: assets フォルダ内に .wav ファイルが見つかりません。音声はスキップします。")
    use_audio = False

def update_sound_volumes(smile_score):
    """
    改訂版：smile_score に応じて1つの音だけを鳴らす
    """
    if not use_audio:
        return

    # 全て無音に
    sound_giggle.set_volume(0.0)
    sound_chuckle.set_volume(0.0)
    sound_big.set_volume(0.0)

    # 条件に応じて一つだけ音を出す
    if smile_score < 0.1:
        sound_big.set_volume(1.0)
    elif smile_score < 0.25:
        sound_giggle.set_volume(1.0)
    else:
        sound_chuckle.set_volume(1.0)

# ------------------------------
# 3) CentroidTracker と NoseLogic 初期化
# ------------------------------
ct = CentroidTracker(max_disappeared=50)
nose_logic = NoseLogic()

# ------------------------------
# 4) カメラキャプチャ開始
# ------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Web カメラを開けませんでした。ID を確認してください。")
    exit(1)

# ------------------------------
# 5) 使用する鼻画像の確認
# ------------------------------
nose_img = None
nose_alpha = None
nose_path = "assets/nose.png"
if os.path.isfile(nose_path):
    img = cv2.imread(nose_path, cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] != 4:
        print("Warning: nose.png が正しい透過 PNG ではありません。鼻オーバーレイはスキップします。")
    else:
        nose_img = img
else:
    print("Warning: assets/nose.png が見つかりません。鼻オーバーレイはスキップします。")

# ------------------------------
# 6) メインループ
# ------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # (1) Face Detection で矩形を取得
    fd_results = fd_model.process(frame_rgb)
    face_boxes = []
    if fd_results.detections:
        for det in fd_results.detections:
            rb = det.location_data.relative_bounding_box
            x_min = int(rb.xmin * w)
            y_min = int(rb.ymin * h)
            bb_w  = int(rb.width * w)
            bb_h  = int(rb.height * h)
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            bb_w  = min(bb_w, w - x_min)
            bb_h  = min(bb_h, h - y_min)
            face_boxes.append((x_min, y_min, bb_w, bb_h))

    # (2) CentroidTracker で ID 管理
    objects = ct.update(face_boxes)  # {ID: (cX,cY), ...}

    # (3) Face Mesh でランドマークを取得
    fm_results = fm_model.process(frame_rgb)
    landmarks_by_id = {}
    smile_by_id = {}

    if fm_results.multi_face_landmarks:
        for face_lms in fm_results.multi_face_landmarks:
            pts = []
            for lm in face_lms.landmark:
                x_px = int(lm.x * w)
                y_px = int(lm.y * h)
                z_rel = lm.z
                pts.append((x_px, y_px, z_rel))
            # 鼻先ポジションで centroid とマッチング
            nose_tip = pts[1]
            lx, ly, _ = nose_tip
            min_dist = float('inf')
            matched_id = None
            for objectID, (cX, cY) in objects.items():
                d = np.hypot(lx - cX, ly - cY)
                if d < min_dist:
                    min_dist = d
                    matched_id = objectID
            if matched_id is not None:
                landmarks_by_id[matched_id] = pts
                smile_by_id[matched_id] = compute_smile_score(pts)

    # (4) 鼻スケール計算
    nose_scales = nose_logic.update(landmarks_by_id, smile_by_id)

    # (5) 音声レイヤーの音量更新
    if use_audio and smile_by_id:
        avg_smile_all = np.mean(list(smile_by_id.values()))
    else:
        avg_smile_all = 0.0
    update_sound_volumes(avg_smile_all)

    # (6) フレームに鼻画像をオーバーレイ
    if nose_img is not None:
        for ID, pts in landmarks_by_id.items():
            x_n, y_n, _ = pts[1]  # 鼻先
            base_size = compute_nose_base_size(pts)
            scale = nose_scales.get(ID, 1.0)
            final_size = int(base_size * scale)

            resized = cv2.resize(nose_img, (final_size, final_size), interpolation=cv2.INTER_AREA)
            alpha_mask = resized[:, :, 3] / 255.0
            overlay_bgr = resized[:, :, 0:3]

            top_left_x = int(x_n - final_size / 2)
            top_left_y = int(y_n - final_size / 2)
            top_left_y = int(y_n - final_size * 0.5) - int(final_size * 0.2)
            overlay_image_alpha(frame, overlay_bgr, (top_left_x, top_left_y), alpha_mask)

    # (7) 顔ボックスを描画 (デバッグ用)
    for (x, y, w_bb, h_bb) in face_boxes:
        cv2.rectangle(frame, (x, y), (x + w_bb, y + h_bb), (0, 255, 0), 1)

    pygame.mixer.init

    # 追加：pygameでスクリーンサイズ取得
    pygame.display.set_mode((1, 1), pygame.NOFRAME)
    info = pygame.display.Info()
    # screen_w, screen_h = info.current_w, info.current_h
    screen_w, screen_h = 1920, 1080

    # (8) ウィンドウ表示の前に以下を追加
    cv2.namedWindow("Nose Mirror", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Nose Mirror", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN) #古ウインドウ
    # cv2.resizeWindow("Nose Mirror", 1920, 1080) #フルHD固定

    # (8) アスペクト比を保って黒背景に描画
    frame_h, frame_w = frame.shape[:2]
    frame_aspect = frame_w / frame_h
    screen_aspect = screen_w / screen_h
    
    if frame_aspect > screen_aspect:
    # 横長の場合：幅基準
        new_w = screen_w
        new_h = int(screen_w / frame_aspect)
    else:
    # 縦長の場合：高さ基準
        new_h = screen_h
        new_w = int(screen_h * frame_aspect)

    # resized_frame = cv2.resize(frame, (new_w, new_h))
    resized_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x_offset = (screen_w - new_w) // 2
    y_offset = (screen_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame


    # (8) ウィンドウ表示
    cv2.imshow("Nose Mirror", canvas)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC で終了
        break
    # (9) smile_score を表示（デバッグ用）
    for ID, score in smile_by_id.items():
        cx, cy = objects[ID]
        cv2.putText(frame, f"smile: {score:.2f}", (cx, cy - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

cap.release()
cv2.destroyAllWindows()
