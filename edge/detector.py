"""
Edge Layer: YOLOv8 + ByteTrack + Speed Estimation สำหรับตรวจจับรถวิ่งเกินความเร็วกำหนด

วิธีใช้:
    python detector.py --config config.yaml
    python detector.py --config config.yaml --source path/to/video.mp4
    python detector.py --config config.yaml --source 0 --show   # webcam + แสดงผลสด

Output:
    - บันทึก event เป็น JSON ลงไฟล์ (config: output.events_log) ทุกครั้งที่เจอรถทำผิด
    - บันทึก log รถทุกคันลงไฟล์แยก โดยชื่อไฟล์เป็น timestamp
    - บันทึกภาพ snapshot ของเหตุการณ์ (config: output.snapshot_dir)
    - ถ้าเปิด mqtt.enabled ใน config จะ publish event ไปยัง MQTT broker ด้วย
    - ถ้าเปิด influxdb.enabled ใน config จะเขียน event เข้า InfluxDB v2 โดยตรง
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import yaml
from ultralytics import YOLO

from event_publisher import EventPublisher
from speed_estimator import SpeedEstimator

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# COCO class id ของยานพาหนะที่สนใจ (โมเดล pretrained ปกติของ ultralytics ใช้ id ชุดนี้)
VEHICLE_CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
THAILAND_TZ = ZoneInfo("Asia/Bangkok")


def thailand_now() -> datetime:
    """เวลาปัจจุบันตามเขตเวลาไทย (GMT+7)"""
    return datetime.now(THAILAND_TZ)


def parse_args():
    p = argparse.ArgumentParser(description="Speeding violation detector (Edge layer)")
    p.add_argument("--config", type=str, default="config.yaml", help="path ไปยังไฟล์ config")
    p.add_argument("--source", type=str, default=None, help="override video source ใน config (path หรือ 0=webcam)")
    p.add_argument("--show", action="store_true", help="เปิดหน้าต่างแสดงผลแบบ real-time")
    return p.parse_args()


def load_config(path: str) -> dict:
    """Load config and resolve project-relative paths from the config location."""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base_dir = config_path.parent
    for key in ("model_path", "source"):
        value = cfg.get(key)
        # Keep webcams (0), stream URLs and absolute paths unchanged.
        if isinstance(value, str) and value and not value.isdigit() and "://" not in value:
            value_path = Path(value)
            if not value_path.is_absolute():
                cfg[key] = str(base_dir / value_path)

    for key in ("snapshot_dir", "events_log", "vehicles_log_dir"):
        value = cfg.get("output", {}).get(key)
        if value and not Path(value).is_absolute():
            cfg["output"][key] = str(base_dir / value)
    return cfg


def resolve_source(raw_source):
    """แปลง '0' เป็น int 0 สำหรับ webcam ถ้าจำเป็น เก็บ path วิดีโอเป็น string ตามเดิม"""
    if isinstance(raw_source, int):
        return raw_source
    if isinstance(raw_source, str) and raw_source.isdigit():
        return int(raw_source)
    return raw_source


def save_snapshot(frame, box, snapshot_dir: str, track_id: int) -> str:
    os.makedirs(snapshot_dir, exist_ok=True)
    ts = thailand_now().strftime("%Y%m%dT%H%M%S+0700")
    path = os.path.join(snapshot_dir, f"viol_{track_id}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


def build_event(camera_id, track_id, vehicle_type, speed, speed_limit, x, y, snapshot_path) -> dict:
    return {
        "timestamp": thailand_now().isoformat(timespec="seconds"),
        "camera_id": camera_id,
        "vehicle_id": int(track_id),
        "type": vehicle_type,
        "speed_kmh": round(float(speed), 1),
        "speed_limit_kmh": speed_limit,
        "event_type": "SPEEDING_VIOLATION",
        "location": {"x": round(float(x), 1), "y": round(float(y), 1)},
        "snapshot_url": snapshot_path,
    }


def build_vehicle_log(camera_id, track_id, vehicle_type, speed, x, y, frame_idx) -> dict:
    """ข้อมูลรถหนึ่งคัน ณ เวลาที่ตรวจพบในแต่ละเฟรม (GMT+7)"""
    return {
        "timestamp": thailand_now().isoformat(timespec="seconds"),
        "camera_id": camera_id,
        "frame_index": frame_idx,
        "vehicle_id": int(track_id),
        "type": vehicle_type,
        "speed_kmh": round(float(speed), 1) if speed is not None else None,
        "location": {"x": round(float(x), 1), "y": round(float(y), 1)},
    }


class VehicleLogger:
    """บันทึกข้อมูลรถทุกคันเป็น JSON Lines ในไฟล์แยกจาก event log"""

    def __init__(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = thailand_now().strftime("%Y%m%dT%H%M%S+0700")
        self.log_path = os.path.join(output_dir, f"vehicles_{timestamp}.jsonl")

    def log(self, vehicle: dict):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(vehicle, ensure_ascii=False) + "\n")


def draw_overlay(frame, box, label, speed, color):
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    text = label if speed is None else f"{label} {speed:.0f} km/h"
    cv2.putText(frame, text, (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    source = resolve_source(args.source if args.source is not None else cfg["source"])
    camera_id = cfg.get("camera_id", "CAM_01")
    speed_limit = cfg["speed_limit_kmh"]
    fps_cfg = cfg.get("fps", 30)

    model = YOLO(cfg.get("model_path", "yolov8n.pt"))

    speed_estimator = SpeedEstimator(
        calibration=cfg["calibration"],
        frame_window=cfg["speed_calc"]["frame_window"],
        smoothing_alpha=cfg["speed_calc"]["smoothing_alpha"],
        fps=fps_cfg,
    )
    publisher = EventPublisher(
        cfg["mqtt"],
        cfg["output"]["events_log"],
        cfg.get("influxdb", {}),
        cfg.get("kafka", {}),
    )
    vehicle_logger = VehicleLogger(cfg["output"].get("vehicles_log_dir", "output"))

    alerted_ids = set()  # track_id ที่เคย trigger alert ไปแล้ว ป้องกันแจ้งซ้ำ
    frame_idx = 0
    event_count = 0

    logger.info(f"เริ่มประมวลผล source={source} camera_id={camera_id} speed_limit={speed_limit} km/h")

    results_gen = model.track(
        source=source,
        classes=list(VEHICLE_CLASS_NAMES.keys()),
        persist=True,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
    )

    try:
        for result in results_gen:
            frame = result.orig_img
            active_ids = []

            if result.boxes is not None and result.boxes.id is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls_id in zip(boxes, track_ids, cls_ids):
                    x1, y1, x2, y2 = box
                    # ใช้จุดกึ่งกลางขอบล่างของ box (จุดที่รถสัมผัสถนน) แม่นกว่าจุดกึ่งกลาง box เฉยๆ
                    cx, cy = (x1 + x2) / 2, y2
                    active_ids.append(tid)

                    speed = speed_estimator.update(tid, frame_idx, cx, cy)
                    label = VEHICLE_CLASS_NAMES.get(cls_id, "vehicle")
                    vehicle_logger.log(build_vehicle_log(camera_id, tid, label, speed, cx, cy, frame_idx))
                    color = (0, 255, 0)

                    if speed is not None and speed > speed_limit:
                        color = (0, 0, 255)
                        if tid not in alerted_ids:
                            alerted_ids.add(tid)
                            snapshot_path = save_snapshot(frame, box, cfg["output"]["snapshot_dir"], tid)
                            event = build_event(camera_id, tid, label, speed, speed_limit, cx, cy, snapshot_path)
                            publisher.publish(event)
                            event_count += 1
                            logger.info(f"[ALERT] vehicle_id={tid} type={label} speed={speed:.1f} km/h > limit {speed_limit}")

                    if args.show:
                        draw_overlay(frame, box, label, speed, color)

            speed_estimator.cleanup(active_ids)

            if args.show:
                cv2.imshow("Speeding Detector (q เพื่อออก)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
    finally:
        publisher.close()
        if args.show:
            cv2.destroyAllWindows()

    logger.info(f"เสร็จสิ้น ประมวลผลทั้งหมด {frame_idx} เฟรม พบเหตุการณ์ {event_count} ครั้ง")
    logger.info(f"บันทึก event ไว้ที่ {cfg['output']['events_log']}, snapshot ไว้ที่ {cfg['output']['snapshot_dir']}")
    logger.info(f"บันทึก log รถทุกคันไว้ที่ {vehicle_logger.log_path}")


if __name__ == "__main__":
    main()
