# utils.py

import cv2
import numpy as np

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """
    img: BGR 8bit 背景フレーム (numpy array)
    img_overlay: BGR 8bit オーバーレイする鼻画像 (アルファ抜き)
    pos: (x, y) のタプル。 背景上の左上に配置する座標
    alpha_mask: float マスク (0.0～1.0) 同じサイズの行列
    """
    x, y = pos
    h, w = img_overlay.shape[:2]

    # オーバーレイ領域が画面に完全に収まらない場合のクリップ処理
    if x < 0 or y < 0 or x + w > img.shape[1] or y + h > img.shape[0]:
        x1 = max(x, 0)
        y1 = max(y, 0)
        x2 = min(x + w, img.shape[1])
        y2 = min(y + h, img.shape[0])

        if x1 >= x2 or y1 >= y2:
            return

        ex1 = x1 - x
        ey1 = y1 - y
        ex2 = ex1 + (x2 - x1)
        ey2 = ey1 + (y2 - y1)

        img_crop = img[y1:y2, x1:x2]
        overlay_crop = img_overlay[ey1:ey2, ex1:ex2]
        mask_crop = alpha_mask[ey1:ey2, ex1:ex2]

        if overlay_crop.size == 0 or mask_crop.size == 0:
            return
        
        h_crop, w_crop = overlay_crop.shape[:2]
        mask_crop = mask_crop[:h_crop, :w_crop]

        inv_mask = 1.0 - mask_crop[..., None]
        img[y1:y2, x1:x2] = (overlay_crop * mask_crop[..., None] + img_crop * inv_mask).astype("uint8")
    else:
        # 完全に画像内に収まる場合
        roi = img[y:y+h, x:x+w]
        inv_mask = 1.0 - alpha_mask[..., None]
        img[y:y+h, x:x+w] = (img_overlay * alpha_mask[..., None] + roi * inv_mask).astype("uint8")
