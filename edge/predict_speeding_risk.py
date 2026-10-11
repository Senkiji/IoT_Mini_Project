"""
Predict Speeding Risk: ทำนายว่า "วัน+ชั่วโมง" ไหนมีความเสี่ยงรถวิ่งเร็วเกินกำหนดสูงสุด

ใช้ RandomForestRegressor เรียนรู้จากรูปแบบ (hour, day_of_week) ของ event ในอดีต
แล้วทำนายความเสี่ยงล่วงหน้าสำหรับทุกชั่วโมงของทุกวันในสัปดาห์ (7x24 = 168 ช่วง)
รวมถึงช่วงเวลาที่ไม่เคยมีข้อมูลจริงมาก่อนด้วย (interpolate จากรูปแบบที่เรียนรู้ได้)

ข้อมูลเข้าได้ 2 ทาง:
    1. ไฟล์ local .jsonl (--events)             - ใช้ทดสอบ/ยังไม่มี InfluxDB
    2. Query ตรงจาก InfluxDB (--influx-config)  - ใช้ข้อมูลจริงที่ระบบสะสมไว้

ผลลัพธ์ถูกเขียนกลับ:
    - ไฟล์ local: CSV + heatmap PNG + JSON สรุป (เสมอ)
    - InfluxDB (ถ้าตั้งค่า --influx-config และเปิด write): measurement
      "speeding_risk_prediction" เพื่อให้ Grafana ดึงไปแสดงเป็น heatmap/ตาราง

วิธีใช้:
    # โหมดทดสอบด้วยไฟล์ local
    python predict_speeding_risk.py --events output/events_history.jsonl

    # โหมดจริง อ่าน+เขียน InfluxDB จาก config.yaml หลักของโปรเจกต์
    python predict_speeding_risk.py --config ../config.yaml --write-influx
"""

import argparse
import json
import os
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score

try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
except ImportError:
    InfluxDBClient = None
    Point = None
    SYNCHRONOUS = None

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
LOCAL_TZ = "Asia/Bangkok"

# ความเร็วเกินนี้ถือว่าผิดปกติ (แทบเป็นไปไม่ได้ในทางกายภาพ) มักเกิดจาก tracking ID กระโดด
# ตัดทิ้งก่อนเทรนโมเดล ไม่งั้นโมเดลจะเรียนรู้ pattern ที่ผิดเพี้ยนจาก noise พวกนี้
DEFAULT_MAX_PLAUSIBLE_SPEED = 200


# ---------------------------------------------------------------------------
# โหลดและทำความสะอาดข้อมูล
# ---------------------------------------------------------------------------

def load_events_from_file(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)


def load_events_from_influx(influx_cfg: dict) -> pd.DataFrame:
    if InfluxDBClient is None:
        raise RuntimeError("ไม่พบ influxdb-client ติดตั้งด้วย pip install influxdb-client")

    client = InfluxDBClient(
        url=influx_cfg["url"], token=influx_cfg["token"], org=influx_cfg["organization"], timeout=20_000
    )
    query_api = client.query_api()
    measurement = influx_cfg.get("measurement", "speeding_events")
    lookback_days = influx_cfg.get("lookback_days", 90)

    flux = f'''
    from(bucket: "{influx_cfg["bucket"]}")
      |> range(start: -{lookback_days}d)
      |> filter(fn: (r) => r._measurement == "{measurement}")
      |> filter(fn: (r) => r._field == "speed_kmh" or r._field == "speed_limit_kmh")
      |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
    '''
    tables = query_api.query_data_frame(flux)
    client.close()

    if isinstance(tables, list):
        df = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()
    else:
        df = tables

    if df.empty:
        return df

    df = df.rename(columns={"_time": "timestamp"})
    return df[["timestamp", "speed_kmh", "speed_limit_kmh"]]


def clean_events(df: pd.DataFrame, max_plausible_speed: float, dedupe: bool) -> pd.DataFrame:
    n_before = len(df)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    if dedupe:
        dedup_cols = [c for c in ["vehicle_id", "speed_kmh", "type"] if c in df.columns]
        if dedup_cols:
            df = df.drop_duplicates(subset=dedup_cols + ["timestamp"], keep="first")

    n_after_dedup = len(df)

    outliers = df["speed_kmh"] > max_plausible_speed
    n_outliers = int(outliers.sum())
    df = df[~outliers].copy()

    print(f"ทำความสะอาดข้อมูล: {n_before} -> ลบซ้ำ {n_before - n_after_dedup} -> "
          f"ลบ outlier (>{max_plausible_speed} km/h) {n_outliers} -> เหลือ {len(df)} event")

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    local_ts = df["timestamp"].dt.tz_convert(LOCAL_TZ)
    df["hour"] = local_ts.dt.hour
    df["day_of_week"] = local_ts.dt.dayofweek  # 0=Mon .. 6=Sun
    return df


# ---------------------------------------------------------------------------
# เทรนโมเดลและทำนาย
# ---------------------------------------------------------------------------

def build_training_table(df: pd.DataFrame) -> pd.DataFrame:
    """รวม event เป็นตาราง (day_of_week, hour) -> count, avg_speed

    สำคัญ: ต้องเติมแถว count=0 สำหรับชั่วโมงที่ "มีการบันทึกจริงแต่ไม่มี violation"
    (เช่น ชั่วโมงเร่งด่วนรถติด แทบไม่มีใครขับเร็วเกิน) ไม่งั้นโมเดลจะไม่เห็นความต่าง
    ระหว่าง "ไม่มีข้อมูลเลย" กับ "มีข้อมูลแต่ความเสี่ยงเป็นศูนย์" แล้วเดาค่าสูงเกินจริง
    ในช่วงที่ไม่เคยมี violation เกิดขึ้น (เพราะ interpolate จากชั่วโมงข้างเคียงที่เสี่ยงสูงกว่า)

    วิธีทำ: เติม 0 เฉพาะชั่วโมงของ "วันที่ (date) ที่รู้แน่ชัดว่ามีการบันทึกจริง" (มี event
    อย่างน้อย 1 ครั้งในวันนั้น ไม่ว่าชั่วโมงไหน) เท่านั้น ไม่เติม 0 ให้วันที่ไม่มีการรันระบบเลย
    เพราะไม่รู้ว่าวันนั้น "ไม่มีใครขับเร็ว" หรือ "กล้องไม่ได้ทำงาน"
    """
    df = df.copy()
    df["date"] = df["timestamp"].dt.tz_convert(LOCAL_TZ).dt.date

    active_dates = df["date"].unique()
    full_grid = pd.MultiIndex.from_product(
        [active_dates, range(24)], names=["date", "hour"]
    ).to_frame(index=False)

    observed = df.groupby(["date", "hour"]).agg(
        count=("speed_kmh", "size"),
        avg_speed=("speed_kmh", "mean"),
    ).reset_index()

    merged = full_grid.merge(observed, on=["date", "hour"], how="left")
    merged["count"] = merged["count"].fillna(0)
    # avg_speed ไม่มีความหมายตอน count=0 (ไม่มีผู้ฝ่าฝืนให้เฉลี่ย) ใช้ speed_limit เป็น floor แทน NaN
    speed_limit_floor = df["speed_limit_kmh"].iloc[0] if "speed_limit_kmh" in df.columns and len(df) else 0
    merged["avg_speed"] = merged["avg_speed"].fillna(speed_limit_floor)

    merged["date"] = pd.to_datetime(merged["date"])
    merged["day_of_week"] = merged["date"].dt.dayofweek

    agg = merged.groupby(["day_of_week", "hour"]).agg(
        count=("count", "mean"),       # ค่าเฉลี่ยต่อครั้งที่เกิดวันนั้นๆ (แทนผลรวมดิบ กัน bias จากจำนวนวันไม่เท่ากัน)
        avg_speed=("avg_speed", "mean"),
    ).reset_index()
    return agg


def cyclical_features(day_of_week: np.ndarray, hour: np.ndarray) -> np.ndarray:
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    return np.column_stack([hour_sin, hour_cos, dow_sin, dow_cos])


def train_and_predict(agg: pd.DataFrame):
    X = cyclical_features(agg["day_of_week"].values, agg["hour"].values)
    y_count = agg["count"].values
    y_speed = agg["avg_speed"].values

    n_samples = len(agg)
    model_count = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=1, random_state=42)
    model_speed = RandomForestRegressor(n_estimators=300, max_depth=None, min_samples_leaf=1, random_state=42)

    # วัดผลโมเดลด้วย cross-validation เทียบกับ baseline (ค่าเฉลี่ยรวมทั้งหมด)
    metrics = {"n_training_rows": n_samples}
    if n_samples >= 5:
        k = min(5, n_samples)
        cv = KFold(n_splits=k, shuffle=True, random_state=42)
        mae_scores = -cross_val_score(model_count, X, y_count, cv=cv, scoring="neg_mean_absolute_error")
        baseline_mae = np.mean(np.abs(y_count - y_count.mean()))
        metrics["model_mae"] = float(mae_scores.mean())
        metrics["baseline_mae"] = float(baseline_mae)
        metrics["better_than_baseline"] = bool(mae_scores.mean() < baseline_mae)
    else:
        metrics["model_mae"] = None
        metrics["baseline_mae"] = None
        metrics["better_than_baseline"] = None
        print("คำเตือน: ข้อมูลน้อยเกินไป (<5 กลุ่ม day/hour) ข้ามการทำ cross-validation")

    model_count.fit(X, y_count)
    model_speed.fit(X, y_speed)

    # ทำนายครบทุก (day_of_week, hour) = 7x24 = 168 ช่วง แม้ไม่เคยมีข้อมูลจริงในช่วงนั้นมาก่อน
    grid_dow, grid_hour = np.meshgrid(range(7), range(24), indexing="ij")
    grid_dow = grid_dow.ravel()
    grid_hour = grid_hour.ravel()
    X_grid = cyclical_features(grid_dow, grid_hour)

    pred_count = model_count.predict(X_grid)
    pred_speed = model_speed.predict(X_grid)

    result = pd.DataFrame({
        "day_of_week": grid_dow,
        "hour": grid_hour,
        "predicted_count": pred_count,
        "predicted_avg_speed": pred_speed,
    })

    # risk_score รวมทั้งความถี่และความรุนแรง (normalize 0-1 แล้วถ่วงน้ำหนักเท่ากัน)
    norm_count = (result["predicted_count"] - result["predicted_count"].min()) / (
        result["predicted_count"].max() - result["predicted_count"].min() + 1e-9
    )
    norm_speed = (result["predicted_avg_speed"] - result["predicted_avg_speed"].min()) / (
        result["predicted_avg_speed"].max() - result["predicted_avg_speed"].min() + 1e-9
    )
    result["risk_score"] = (0.5 * norm_count + 0.5 * norm_speed).round(3)

    tiers = pd.qcut(result["risk_score"], q=3, labels=["Low", "Medium", "High"], duplicates="drop")
    result["risk_level"] = tiers.astype(str)

    return result, metrics


# ---------------------------------------------------------------------------
# Output: CSV, chart, JSON, InfluxDB
# ---------------------------------------------------------------------------

def save_outputs(result: pd.DataFrame, metrics: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "risk_prediction.csv")
    result.to_csv(csv_path, index=False)

    top10 = result.sort_values("risk_score", ascending=False).head(10)
    summary = {
        "metrics": metrics,
        "top_risk_periods": [
            {
                "day": DAY_NAMES[int(r.day_of_week)],
                "hour": f"{int(r.hour):02d}:00-{int(r.hour)+1:02d}:00",
                "risk_score": r.risk_score,
                "risk_level": r.risk_level,
                "predicted_count": round(r.predicted_count, 1),
                "predicted_avg_speed": round(r.predicted_avg_speed, 1),
            }
            for r in top10.itertuples()
        ],
    }
    json_path = os.path.join(output_dir, "risk_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    heatmap_path = os.path.join(output_dir, "risk_heatmap.png")
    plot_heatmap(result, heatmap_path)

    print(f"\nบันทึกผลลัพธ์ไว้ที่:\n  {csv_path}\n  {json_path}\n  {heatmap_path}")
    return summary


def plot_heatmap(result: pd.DataFrame, output_path: str):
    pivot = result.pivot(index="day_of_week", columns="hour", values="risk_score")
    fig, ax = plt.subplots(figsize=(14, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(24))
    ax.set_xticklabels(range(24))
    ax.set_yticks(range(7))
    ax.set_yticklabels(DAY_NAMES)
    ax.set_xlabel("Hour of Day")
    ax.set_title("Predicted Speeding Risk Score (Day x Hour)")
    plt.colorbar(im, ax=ax, label="Risk Score (0-1)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)


def write_to_influx(result: pd.DataFrame, influx_cfg: dict):
    if InfluxDBClient is None:
        print("คำเตือน: ไม่พบ influxdb-client ข้ามการเขียนผลกลับ InfluxDB")
        return

    client = InfluxDBClient(
        url=influx_cfg["url"], token=influx_cfg["token"], org=influx_cfg["organization"], timeout=20_000
    )
    write_api = client.write_api(write_options=SYNCHRONOUS)
    measurement = influx_cfg.get("prediction_measurement", "traffic_ML_6610301010")

    # เขียน timestamp เป็น "ครั้งถัดไป" ของแต่ละ day_of_week/hour ในสัปดาห์นี้
    # เพื่อให้ Grafana panel แบบ time-series ปกติแสดงผลได้โดยไม่ต้องตั้งค่าพิเศษ
    today = datetime.utcnow()
    points = []
    for row in result.itertuples():
        days_ahead = (row.day_of_week - today.weekday()) % 7
        target_date = today + timedelta(days=days_ahead)
        ts = target_date.replace(hour=int(row.hour), minute=0, second=0, microsecond=0)

        point = (
            Point(measurement)
            .tag("day_of_week", DAY_NAMES[int(row.day_of_week)])
            .tag("hour", str(int(row.hour)))
            .tag("risk_level", row.risk_level)
            .field("predicted_count", float(row.predicted_count))
            .field("predicted_avg_speed", float(row.predicted_avg_speed))
            .field("risk_score", float(row.risk_score))
            .time(ts)
        )
        points.append(point)

    write_api.write(bucket=influx_cfg["bucket"], org=influx_cfg["organization"], record=points)
    client.close()
    print(f"เขียนผลทำนาย {len(points)} จุดกลับเข้า InfluxDB measurement '{measurement}' แล้ว")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Predict speeding risk by day/hour")
    parser.add_argument("--events", type=str, default=None, help="path ไปยัง events.jsonl (local mode)")
    parser.add_argument("--config", type=str, default=None,
                        help="path ไปยัง config.yaml หลักของโปรเจกต์ (อ่าน section influxdb)")
    parser.add_argument("--influx-config", type=str, default=None,
                        help="path ไปยัง YAML ที่มีค่า InfluxDB โดยตรง (รองรับรูปแบบเดิม)")
    parser.add_argument("--max-speed", type=float, default=DEFAULT_MAX_PLAUSIBLE_SPEED,
                         help="ตัด event ที่ความเร็วเกินนี้ทิ้งก่อนเทรน (ถือว่าผิดปกติ)")
    parser.add_argument("--no-dedupe", action="store_true", help="ปิดการลบ event ที่ซ้ำกันเป๊ะ")
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--write-influx", action="store_true", help="เขียนผลทำนายกลับ InfluxDB (ต้องมี --influx-config)")
    args = parser.parse_args()

    if not args.events and not args.config and not args.influx_config:
        parser.error("ต้องระบุ --events, --config หรือ --influx-config อย่างใดอย่างหนึ่ง")

    influx_cfg = None
    config_path = args.config or args.influx_config
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f) or {}
        # --config reads the project's main config; --influx-config keeps backward compatibility.
        influx_cfg = raw_config.get("influxdb", raw_config) if args.config else raw_config
        if not influx_cfg.get("enabled", True):
            print("คำเตือน: influxdb.enabled เป็น false แต่ยังจะลองอ่านข้อมูลตามค่าที่ตั้งไว้")

    if args.events:
        print(f"โหลดข้อมูลจากไฟล์ local: {args.events}")
        df = load_events_from_file(args.events)
    else:
        print(f"ดึงข้อมูลจาก InfluxDB bucket: {influx_cfg['bucket']}")
        df = load_events_from_influx(influx_cfg)

    if df.empty:
        print("ไม่พบข้อมูล event เลย ไม่สามารถเทรนโมเดลได้")
        return

    df = clean_events(df, args.max_speed, dedupe=not args.no_dedupe)
    if df.empty:
        print("ไม่มี event ที่ใช้ได้หลังทำความสะอาดข้อมูล จึงยังเทรนโมเดลไม่ได้")
        return
    df = add_time_features(df)

    n_distinct_dates = df["timestamp"].dt.date.nunique()
    if n_distinct_dates < 7:
        print(f"\nคำเตือน: ข้อมูลมีแค่ {n_distinct_dates} วันที่แตกต่างกัน "
              f"การทำนายตอนนี้ยังความน่าเชื่อถือต่ำ ควรรันระบบสะสมข้อมูลจริงต่อเนื่องอย่างน้อย 1-2 สัปดาห์ก่อนใช้ผลจริงจัง\n")

    agg = build_training_table(df)
    result, metrics = train_and_predict(agg)

    print("\n=== ผลการวัดประสิทธิภาพโมเดล (cross-validation) ===")
    if metrics["model_mae"] is not None:
        print(f"Model MAE   : {metrics['model_mae']:.2f}")
        print(f"Baseline MAE: {metrics['baseline_mae']:.2f} (ทำนายด้วยค่าเฉลี่ยรวมเฉยๆ)")
        print(f"โมเดลดีกว่า baseline: {'ใช่' if metrics['better_than_baseline'] else 'ไม่ (ข้อมูลอาจน้อย/สุ่มเกินไป)'}")

    print("\n=== ช่วงเวลาเสี่ยงสูงสุด 10 อันดับ ===")
    top10 = result.sort_values("risk_score", ascending=False).head(10)
    for r in top10.itertuples():
        print(f"  {DAY_NAMES[int(r.day_of_week)]} {int(r.hour):02d}:00  "
              f"risk={r.risk_score:.2f} ({r.risk_level})  "
              f"predicted_count={r.predicted_count:.1f}  predicted_speed={r.predicted_avg_speed:.1f} km/h")

    save_outputs(result, metrics, args.output_dir)

    if args.write_influx:
        if not influx_cfg:
            print("ต้องระบุ --influx-config ด้วยถ้าจะใช้ --write-influx")
        else:
            write_to_influx(result, influx_cfg)


if __name__ == "__main__":
    main()
