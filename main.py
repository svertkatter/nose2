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

# ──────────────────────────────────────────────
# 定数・モデル初期化
# ──────────────────────────────────────────────
# HOG + SVM（OpenCV内蔵）による複数人検出
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# フィルター用しきい値
MIN_BODY_BOX_AREA        = 5000
VISIBILITY_THRESH        = 0.5
IOU_THRESH               = 0.3
FALLBACK_MIN_AREA        = MIN_BODY_BOX_AREA
TWO_PERSON_SWITCH_INTERVAL = 90.0  # 二人モードで入れ替え

# MediaPipe Pose
mp_pose    = mp.solutions.pose
pose_model = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
# MediaPipe Face Detection & Face Mesh
mp_fd    = mp.solutions.face_detection
fd_model = mp_fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)
mp_fm    = mp.solutions.face_mesh
fm_model = mp_fm.FaceMesh(
    static_image_mode=False,
    max_num_faces=6,
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# PyGame オーディオ／画面
pygame.mixer.init()
pygame.display.set_mode((1, 1), pygame.NOFRAME)
screen_w, screen_h = 1920, 1080

# サウンドロード
use_audio = True
try:
    sound_giggle = pygame.mixer.Sound("assets/laugh_giggle.wav")
    sound_chuckle = pygame.mixer.Sound("assets/laugh_chuckle.wav")
    sound_big     = pygame.mixer.Sound("assets/laugh_big.wav")
    for s in (sound_giggle, sound_chuckle, sound_big):
        s.play(loops=-1)
        s.set_volume(0.0)
except:
    print("Warning: 音声がロードできませんでした。")
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

# 鼻画像読み込み
nose_images = []
nose_alphas = []
for path in sorted(glob("assets/nose_*.png")):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is not None and img.shape[2] == 4:
        nose_images.append(img[:, :, :3])
        nose_alphas.append(img[:, :, 3] / 255.0)
if not nose_images:
    print("Warning: 鼻画像が見つかりません。")

# トラッカー＆ロジック
ct          = CentroidTracker(max_disappeared=300)
nose_logic  = NoseLogic()

# 鼻画像割当ステート
assigned_id             = None
assigned_img_idx        = None
two_person_last_switch  = None

# カメラ開始
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Webカメラを開けませんでした。")
    sys.exit(1)

# フルスクリーンウィンドウ
cv2.namedWindow("Nose Mirror", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Nose Mirror", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# IoU 計算関数
def bbox_iou(a, b):
    xA = max(a[0], b[0]);    yA = max(a[1], b[1])
    xB = min(a[0]+a[2], b[0]+b[2]); yB = min(a[1]+a[3], b[1]+b[3])
    interW = max(0, xB - xA); interH = max(0, yB - yA)
    interA = interW * interH
    union = a[2]*a[3] + b[2]*b[3] - interA
    return interA / union if union > 0 else 0

# メインループ
while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _   = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # PoseLandmark→pose_bboxフィルター
    pose_bbox = None
    pose_res = pose_model.process(frame_rgb)
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
            x0, x1 = max(min(xs), 0), min(max(xs), w)
            y0, y1 = max(min(ys), 0), min(max(ys), h)
            bw, bh = x1 - x0, y1 - y0
            if bw * bh >= MIN_BODY_BOX_AREA:
                pose_bbox = (x0, y0, bw, bh)

    # １）優先：顔検出ボックス
    boxes = []
    face_res = fd_model.process(frame_rgb)
    if face_res.detections:
        for det in face_res.detections:
            bb = det.location_data.relative_bounding_box
            x1 = int(bb.xmin * w); y1 = int(bb.ymin * h)
            bw = int(bb.width * w); bh = int(bb.height * h)
            if bw * bh >= MIN_BODY_BOX_AREA:
                boxes.append((x1, y1, bw, bh))

    # ２）顔がなければ HOG＋Pose/IoU 補完
    if not boxes:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects, _ = hog.detectMultiScale(gray, winStride=(8,8), padding=(16,16), scale=1.05)
        for x, y, bw, bh in rects:
            if bw*bh < MIN_BODY_BOX_AREA:
                continue
            if pose_bbox and bbox_iou((x,y,bw, bh), pose_bbox) > IOU_THRESH:
                boxes.append((x, y, bw, bh))
            elif not pose_bbox and bw*bh >= FALLBACK_MIN_AREA:
                boxes.append((x, y, bw, bh))

    # トラッカー更新（必ず一度だけ）
    objects = ct.update(boxes)

    # FaceMesh→鼻先ランドマーク＆笑顔スコア
    fm_res = fm_model.process(frame_rgb)
    landmarks_by_id = {}
    smile_by_id     = {}
    if fm_res.multi_face_landmarks:
        for face_lms in fm_res.multi_face_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h), lm.z) for lm in face_lms.landmark]
            nx, ny, _ = pts[1]
            best_id, min_d = None, float("inf")
            for oid, (cx, cy) in objects.items():
                d = (nx-cx)**2 + (ny-cy)**2
                if d < min_d:
                    min_d, best_id = d, oid
            if best_id is not None:
                landmarks_by_id[best_id] = pts
                smile_by_id[best_id]     = compute_smile_score(pts)

    # 鼻画像割当ロジック
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
            assigned_id = None
            assigned_img_idx = None
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
            if other:
                assigned_id = other[0]
            two_person_last_switch = cur_time

    # 鼻スケール更新・音声更新
    nose_scales = nose_logic.update(landmarks_by_id, smile_by_id)
    if smile_by_id:
        avg_smile = np.mean(list(smile_by_id.values()))
        update_sound_volumes(avg_smile)
    else:
        sound_giggle.set_volume(0.0)
        sound_chuckle.set_volume(0.0)
        sound_big.set_volume(0.0)

    # 鼻オーバーレイ
    if assigned_id in landmarks_by_id:
        pts = landmarks_by_id[assigned_id]
        x_n, y_n, _ = pts[1]
        base = compute_nose_base_size(pts)
        scale = nose_scales.get(assigned_id, 1.0)
        size  = int(base * scale)
        img   = nose_images[assigned_img_idx]
        alpha = nose_alphas[assigned_img_idx]
        rgb_r = cv2.resize(img,   (size, size))
        a_r   = cv2.resize(alpha, (size, size))
        tx = int(x_n - size/2)
        ty = int(y_n - size*0.7)
        overlay_image_alpha(frame, rgb_r, (tx, ty), a_r)

    # フルスクリーン描画
    fh, fw = frame.shape[:2]
    fa, sa = fw/fh, screen_w/screen_h
    if fa > sa:
        nw, nh = screen_w, int(screen_w/fa)
    else:
        nh, nw = screen_h, int(screen_h*fa)
    rf = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    ox = (screen_w - nw)//2; oy = (screen_h - nh)//2
    canvas[oy:oy+nh, ox:ox+nw] = rf

    cv2.imshow("Nose Mirror", canvas)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
