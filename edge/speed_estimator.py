"""
SpeedEstimator: แปลงตำแหน่ง (pixel) ของรถแต่ละคันในแต่ละเฟรม ให้เป็นความเร็ว (km/h)

รองรับ 2 วิธี calibration:
    - simple_ratio: คูณระยะ pixel ด้วยค่าคงที่ meters_per_pixel (ง่าย เหมาะกล้องมุมตรง)
    - homography:   แปลงพิกัด pixel เป็นพิกัดโลกจริงผ่าน perspective transform
                     (แม่นกว่า เหมาะกล้องมุมเอียง/มองจากที่สูง)
"""

from collections import defaultdict, deque

import cv2
import numpy as np


class SpeedEstimator:
    def __init__(
        self,
        calibration: dict,
        frame_window: int = 10,
        smoothing_alpha: float = 0.3,
        fps: float = 30.0,
    ):
        self.method = calibration.get("method", "simple_ratio")
        self.frame_window = frame_window
        self.smoothing_alpha = smoothing_alpha
        self.fps = fps

        if self.method == "homography":
            pixel_pts = np.array(calibration["pixel_points"], dtype=np.float32)
            real_pts = np.array(calibration["real_points"], dtype=np.float32)
            if pixel_pts.shape != (4, 2) or real_pts.shape != (4, 2):
                raise ValueError("homography ต้องการจุดอ้างอิงอย่างละ 4 จุด (pixel_points, real_points)")
            self.H = cv2.getPerspectiveTransform(pixel_pts, real_pts)
        elif self.method == "simple_ratio":
            self.meters_per_pixel = calibration["meters_per_pixel"]
        else:
            raise ValueError(f"ไม่รู้จัก calibration method: {self.method}")

        # track_id -> deque ของ (frame_idx, x_pixel, y_pixel)
        self.history = defaultdict(lambda: deque(maxlen=frame_window + 1))
        # track_id -> ความเร็วที่ผ่าน smoothing แล้ว (km/h)
        self.smoothed_speed = {}

    def _pixel_to_world(self, x: float, y: float):
        """แปลงพิกัด pixel เป็นพิกัดโลกจริง (หน่วยเมตร)"""
        if self.method == "homography":
            pt = np.array([[[x, y]]], dtype=np.float32)
            world = cv2.perspectiveTransform(pt, self.H)
            return float(world[0][0][0]), float(world[0][0][1])
        return x * self.meters_per_pixel, y * self.meters_per_pixel

    def update(self, track_id: int, frame_idx: int, cx: float, cy: float):
        """
        ป้อนตำแหน่งล่าสุดของรถคันนี้ในเฟรมนี้
        คืนค่าความเร็ว (km/h แบบ smoothed) หรือ None ถ้ายังมีข้อมูลไม่พอคำนวณ
        """
        self.history[track_id].append((frame_idx, cx, cy))
        hist = self.history[track_id]

        if len(hist) < 2:
            return None

        f_old, x_old, y_old = hist[0]
        f_new, x_new, y_new = hist[-1]
        frame_delta = f_new - f_old

        if frame_delta <= 0:
            return None

        wx_old, wy_old = self._pixel_to_world(x_old, y_old)
        wx_new, wy_new = self._pixel_to_world(x_new, y_new)
        dist_m = ((wx_new - wx_old) ** 2 + (wy_new - wy_old) ** 2) ** 0.5

        time_s = frame_delta / self.fps
        if time_s <= 0:
            return None

        speed_kmh = (dist_m / time_s) * 3.6

        prev = self.smoothed_speed.get(track_id)
        smoothed = speed_kmh if prev is None else (
            self.smoothing_alpha * speed_kmh + (1 - self.smoothing_alpha) * prev
        )
        self.smoothed_speed[track_id] = smoothed
        return smoothed

    def get_speed(self, track_id: int):
        return self.smoothed_speed.get(track_id)

    def cleanup(self, active_ids):
        """ลบ track ที่หายไปจากเฟรม (รถออกจากภาพแล้ว) เพื่อไม่ให้ memory โตเรื่อยๆ"""
        active_set = set(active_ids)
        stale = set(self.history.keys()) - active_set
        for tid in stale:
            del self.history[tid]
            self.smoothed_speed.pop(tid, None)
