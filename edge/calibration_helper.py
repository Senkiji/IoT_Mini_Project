"""
Calibration Helper: เครื่องมือช่วยหาค่า calibration แบบ homography จากวิดีโอจริง

วิธีใช้:
    python calibration_helper.py --source path/to/video.mp4 --config config.yaml

วิธีทำ:
    1. โปรแกรมจะเปิดเฟรมแรกของวิดีโอขึ้นมา
    2. คลิกซ้าย เลือก 4 จุดอ้างอิงบนถนนตามลำดับ (แนะนำ: 4 มุมของสี่เหลี่ยมบนพื้นถนน
       ที่รู้ขนาดจริง เช่น มุมของเลนที่กว้าง 3.5 ม. ยาว 20 ม.)
    3. กด 'r' รีเซ็ตถ้าเลือกผิด, กด 'q' เมื่อเลือกครบ 4 จุดแล้วเพื่อยืนยัน
    4. ใส่พิกัดจริง (หน่วยเมตร) ของแต่ละจุดตามลำดับเดียวกัน
    5. โปรแกรมจะเขียนค่า calibration ลงใน config.yaml ให้อัตโนมัติ (method: homography)
"""

import argparse

import cv2
import yaml

points = []


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append([x, y])
        print(f"จุดที่ {len(points)}: ({x}, {y})")


def main():
    parser = argparse.ArgumentParser(description="Interactive homography calibration tool")
    parser.add_argument("--source", required=True, help="path วิดีโอ หรือ 0 สำหรับ webcam")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        print("อ่านเฟรมจากวิดีโอไม่สำเร็จ ตรวจสอบ path/source อีกครั้ง")
        return

    window_name = "Calibration - คลิกเลือก 4 จุด | r=รีเซ็ต | q=ยืนยัน"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        display = frame.copy()
        for i, (px, py) in enumerate(points):
            cv2.circle(display, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(display, str(i + 1), (px + 8, py), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if len(points) == 4:
            pts = points + [points[0]]
            for i in range(4):
                cv2.line(display, tuple(pts[i]), tuple(pts[i + 1]), (0, 255, 0), 1)

        cv2.imshow(window_name, display)
        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            points.clear()
            print("รีเซ็ตจุดแล้ว")
        elif key == ord("q"):
            if len(points) == 4:
                break
            print(f"ต้องเลือกให้ครบ 4 จุดก่อน (ตอนนี้มี {len(points)} จุด)")

    cv2.destroyAllWindows()

    print("\nใส่พิกัดโลกจริง (หน่วยเมตร) ของแต่ละจุด ตามลำดับเดียวกับที่คลิกไว้")
    print("ตัวอย่าง: ถ้าจุดที่ 1 คือมุมซ้ายบนของสี่เหลี่ยมอ้างอิง ให้ใส่ 0,0")
    real_points = []
    for i in range(4):
        while True:
            raw = input(f"จุดที่ {i + 1} - x,y (เมตร) เช่น 0,0 : ").strip()
            try:
                x_str, y_str = raw.split(",")
                real_points.append([float(x_str), float(y_str)])
                break
            except ValueError:
                print("รูปแบบไม่ถูกต้อง กรุณาใส่เป็น x,y เช่น 3.5,0")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["calibration"] = {
        "method": "homography",
        "pixel_points": points,
        "real_points": real_points,
    }

    with open(args.config, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"\nบันทึก calibration (homography) ลงใน {args.config} เรียบร้อยแล้ว")
    print("ลองรัน detector.py --show เพื่อดูผลได้เลย")


if __name__ == "__main__":
    main()
