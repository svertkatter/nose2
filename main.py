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

# HOG + SVMを使った複数人用ボディ検出器初期化
#（OpenCVにバンドルされているものを使用）
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# ボックスフィルター用のしきい値
MIN_BODY_BOX_AREA = 5000 #これ以下の小さなボックスは捨てる
VISIBILITY_THRESH = 0.5
IOU_THRESH = 0.3
FALLBACK_MIN_AREA = MIN_BODY_BOX_AREA

# ------------------------------
# 1) MediaPipe 初期化
# ------------------------------
mp_pose = mp.solutions.pose
pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_face_detection = mp.solutions.face_detection
mp_face_mesh      = mp.solutions.face_mesh

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
screen_w, screen_h = 1920, 1080  # Full HD固定

# ------------------------------
# 3) 音声読み込み
# ------------------------------
use_audio = True
try:
    sound_giggle = pygame.mixer.Sound("assets/laugh_giggle.wav")
    sound_chuckle = pygame.mixer.Sound("assets/laugh_chuckle.wav")
    sound_big     = pygame.mixer.Sound("assets/laugh_big.wav")
    for s in (sound_giggle, sound_chuckle, sound_big):
        s.play(loops=-1)
        s.set_volume(0.0)
except:
    print("Warning: 音声がロードできなかったため、スキップします。")
    use_audio = False

def update_sound_volumes(smile_score):
    if not use_audio:
        return
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
for path in sorted(glob("assets/nose_*.png")):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.shape[2] == 4:
        nose_images.append(img[:, :, :3])
        nose_alphas.append(img[:, :, 3] / 255.0)
if not nose_images:
    print("Warning: 鼻画像が見つかりません。鼻オーバーレイはスキップされます。")

# ------------------------------
# 5) その他初期化
# ------------------------------
ct = CentroidTracker(max_disappeared=300)
nose_logic       = NoseLogic()
nose_image_by_id = {}

assigned_id = None
assigned_img_idx = None
two_person_last_switch = None
TWO_PERSON_SWITCH_INTERVAL = 90.0

# ------------------------------
# 6) カメラ開始
# ------------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
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

    h, w, _   = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pose_res = pose_model.process(frame_rgb)
    pose_bbox = None
    if pose_res.pose_landmarks:
        lm = pose_res.pose_landmarks.landmark
        key_ids = [
            PoseLandmark.LEFT_SHOULDER.value,
            PoseLandmark.RIGHT_SHOULDER.value,
            PoseLandmark.LEFT_HIP.value,
            PoseLandmark.RIGHT_HIP.value,
        ]

        avg_vis = sum(lm[i].visibility for i in key_ids) / len(key_ids)
        if avg_vis > VISIBILITY_THRESH:
            coords = [(int(l.x * w), int(l.y * h)) for l in lm]
            xs, ys = zip(*coords)
            p_xmin, p_xmax = max(min(xs), 0), min(max(xs), w)
            p_ymin, p_ymax = max(min(ys), 0), min(max(ys), h)
            p_bw, p_bh = p_xmax - p_xmin, p_ymax - p_ymin
            if p_bw * p_bh >= MIN_BODY_BOX_AREA:
                pose_bbox = (p_xmin, p_ymin, p_bw, p_bh)

    # 〈置換〉体トラッキング（Pose → CentroidTracker）
    # body_boxes = []
    # pose_res = pose_model.process(frame_rgb)
    # if pose_res.pose_landmarks:
    #     coords = [(int(lm.x * w), int(lm.y * h))
    #               for lm in pose_res.pose_landmarks.landmark]
    #     xs, ys = zip(*coords)
    #     x_min, x_max = max(min(xs), 0), min(max(xs), w)
    #     y_min, y_max = max(min(ys), 0), min(max(ys), h)
    #     body_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
    # objects = ct.update(body_boxes)

    # 体トラッキング（HOG＋SVMによる複数人検出）
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects, _ = hog.detectMultiScale(
        gray,
        winStride=(8, 8),
        padding=(16, 16),
        scale=1.05
    )

    #IoU計算
    def bbox_iou(a, b):
        xA = max(a[0], b[0]); yA = max(a[1], b[1])
        xB = min(a[0]+a[2], b[0]+b[2]); yB = min(a[1]+a[3], b[1]+b[3])
        interW = max(0, xB - xA); interH = max(0, yB - yA)
        interA = interW * interH
        union = a[2]*a[3] + b[2]*b[3] - interA
        return interA / union if union > 0 else 0

    body_boxes = []
    for (x, y, bw, bh) in rects:
        if bw * bh < MIN_BODY_BOX_AREA:
            continue
        if pose_bbox:
            # px, py, pbw, pbh = pose_bbox
            # if x <= px and y <= py and x + bw >= px + pbw and y + bh >= py + pbh:
            #     body_boxes.append((x, y, bw, bh))
            if bbox_iou((x, y, bw, bh), pose_bbox) > IOU_THRESH:
                body_boxes.append((x, y, bw, bh))
        else:
            if bw * bh >= FALLBACK_MIN_AREA:
                body_boxes.append((x, y, bw, bh))

    # CentroidTrackerにわたすリスト形式にそのままつめる
    # body_boxes = [(x, y, w, h) for (x, y, w, h) in rects]
    objects = ct.update(body_boxes)

    # 〈追加 デバッグ〉トラッキング枠とIDを表示
    for box in body_boxes:
        x, y, bw, bh = box
        # ボックス中心を計算
        cx_box, cy_box = x + bw // 2, y + bh // 2
        # 最も近い centroid の objectID を探す
        best_id, min_d = None, float('inf')
        for objectID, (cx, cy) in objects.items():
            d = (cx_box - cx)**2 + (cy_box - cy)**2
            if d < min_d:
                min_d, best_id = d, objectID
        if best_id is not None:
            # 緑色の枠を描画
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            # 左上にIDを表示
            cv2.putText(
                frame,
                f'ID:{best_id}',
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # 新規IDには鼻画像ランダム割当
    for objectID in objects:
        if objectID not in nose_image_by_id and nose_images:
            idx = random.randint(0, len(nose_images) - 1)
            nose_image_by_id[objectID] = (nose_images[idx],
                                          nose_alphas[idx])

    # 〈既存〉FaceMesh でランドマーク取得 → body-IDに紐づけ
    fm_results = fm_model.process(frame_rgb)
    landmarks_by_id = {}
    smile_by_id     = {}
    if fm_results.multi_face_landmarks:
        for face_lms in fm_results.multi_face_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h), lm.z)
                   for lm in face_lms.landmark]
            nx, ny, _ = pts[1]  # 鼻先
            best_id, min_d = None, float('inf')
            for objectID, (cx, cy) in objects.items():
                d = (nx - cx)**2 + (ny - cy)**2
                if d < min_d:
                    min_d, best_id = d, objectID
            if best_id is not None:
                landmarks_by_id[best_id] = pts
                smile_by_id[best_id]     = compute_smile_score(pts)
    
    # 鼻画像割当ロジック
    cur_time = time.time()
    current_faces = list(landmarks_by_id.keys())

    # 1.新規割当
    if assigned_id is None:
        if len(current_faces) >= 1:
            if len(current_faces) == 1:
                assigned_id = current_faces[0]
                two_person_last_switch = None
            else:
                assigned_id = random.choice(current_faces)
                two_person_last_switch = cur_time
            assigned_img_idx = random.randint(0, len(nose_images) - 1)

    elif assigned_id not in current_faces:
        if len(current_faces) == 0:
            assigned_id = None
            assigned_img_idx = None
            two_person_last_switch = None
        elif len(current_faces) == 1:
            assigned_id = current_faces[0]
            assigned_img_idx = random.randint(0, len(nose_images) - 1)
            two_person_last_switch = None
        else:
            assigned_id = random.choice(current_faces)
            assigned_img_idx = random.randint(0, len(nose_images) - 1)
            two_person_last_switch = cur_time
    elif len(current_faces) == 2:
        if two_person_last_switch is None:
            two_person_last_switch = cur_time
        elif cur_time - two_person_last_switch >= TWO_PERSON_SWITCH_INTERVAL:
            other = [i for i in current_faces if i != assigned_id]
            if other:
                assigned_id = other[0]
            two_person_last_switch = cur_time
    

    # 鼻スケール計算
    nose_scales = nose_logic.update(landmarks_by_id, smile_by_id)

    # 音声更新
    if smile_by_id:
        avg_smile = np.mean(list(smile_by_id.values()))
        update_sound_volumes(avg_smile)
    else:
        sound_giggle.set_volume(0.0)
        sound_chuckle.set_volume(0.0)
        sound_big.set_volume(0.0)

    # 鼻描画
    # for ID, pts in landmarks_by_id.items():
    #     if ID not in nose_image_by_id:
    #         continue
    #     x_n, y_n, _     = pts[1]
    #     base_size       = compute_nose_base_size(pts)
    #     scale           = nose_scales.get(ID, 1.0)
    #     final_size      = int(base_size * scale)
    #     img_bgr, alpha  = nose_image_by_id[ID]
    #     resized_bgr     = cv2.resize(img_bgr, (final_size, final_size))
    #     resized_alpha   = cv2.resize(alpha,   (final_size, final_size))
    #     top_left_x      = int(x_n - final_size/2)
    #     top_left_y      = int(y_n - final_size*0.7)
    #     overlay_image_alpha(frame, resized_bgr,
    #                         (top_left_x, top_left_y),
    #                         resized_alpha)

    for ID, pts in landmarks_by_id.items():
        if ID != assigned_id:
            continue

        x_n, y_n, _ = pts[1]
        base_size = compute_nose_base_size(pts)
        scale = nose_scales.get(ID, 1.0)
        final_size = int(base_size * scale)
        img_bgr = nose_images[assigned_img_idx]
        alpha = nose_alphas[assigned_img_idx]
        resized_bgr = cv2.resize(img_bgr, (final_size, final_size))
        resized_alpha = cv2.resize(alpha, (final_size, final_size))
        top_left_x = int(x_n - final_size/2)
        top_left_y = int(y_n - final_size*0.7)
        overlay_image_alpha(frame, resized_bgr,
                            (top_left_x, top_left_y),
                            resized_alpha)

    # アスペクト比を保って黒背景に描画
    frame_h, frame_w = frame.shape[:2]
    frame_aspect     = frame_w / frame_h
    screen_aspect    = screen_w / screen_h
    if frame_aspect > screen_aspect:
        new_w = screen_w
        new_h = int(screen_w / frame_aspect)
    else:
        new_h = screen_h
        new_w = int(screen_h * frame_aspect)
    resized_frame = cv2.resize(frame, (new_w, new_h),
                               interpolation=cv2.INTER_AREA)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    x_offset = (screen_w - new_w) // 2
    y_offset = (screen_h - new_h) // 2
    canvas[y_offset:y_offset+new_h,
           x_offset:x_offset+new_w] = resized_frame

    # 画面表示
    cv2.imshow("Nose Mirror", canvas)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
