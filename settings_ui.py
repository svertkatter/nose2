# settings_ui.py — OpenCVトラックバーで設定UI
import cv2

class SettingsUI:
    """
    OpenCV Trackbar を使った簡易設定UI。
    - Camera: 0..5
    - SwapSec: 5..60
    - MaxScale: 3.0..6.0（内部は×10のintで扱う）
    - Debug: 0/1
    """
    def __init__(self, initial: dict):
        self._win = "Settings"
        cv2.namedWindow(self._win)
        cv2.resizeWindow(self._win, 420, 180)

        cam = int(initial.get("camera_index", 0))
        swap = float(initial.get("swap_sec", 15.0))
        msc = float(initial.get("max_scale", 4.5))
        dbg = int(initial.get("debug_overlay", 1))

        def _noop(x): pass

        cv2.createTrackbar("Camera",  self._win, max(0, min(5, cam)), 5, _noop)
        cv2.createTrackbar("SwapSec", self._win, int(max(5, min(60, swap))), 60, _noop)
        cv2.createTrackbar("MaxScale(x10)", self._win, int(max(30, min(60, msc*10))), 60, _noop)
        cv2.createTrackbar("Debug(0/1)", self._win, 1 if dbg else 0, 1, _noop)

    def read(self) -> dict:
        cam  = cv2.getTrackbarPos("Camera",  self._win)
        swap = cv2.getTrackbarPos("SwapSec", self._win)
        mscx = cv2.getTrackbarPos("MaxScale(x10)", self._win)
        dbg  = cv2.getTrackbarPos("Debug(0/1)", self._win)
        swap = max(5, min(60, int(swap)))
        max_scale = max(3.0, min(6.0, mscx / 10.0))
        return {
            "camera_index": cam,
            "swap_sec": float(swap),
            "max_scale": float(max_scale),
            "debug_overlay": 1 if dbg else 0
        }
