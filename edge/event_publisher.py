"""
EventPublisher: ส่ง event ที่ตรวจพบ (violation) ออกไป 4 ทาง
    1. บันทึกลงไฟล์ local (.jsonl) - ใช้เป็น backup และทดสอบได้แม้ไม่มี MQTT broker
    2. publish ผ่าน MQTT ถ้าเปิดใช้งานและเชื่อมต่อได้ (สำหรับต่อเข้า Kafka/Docker stack ภายหลัง)
    3. ส่งเข้า Kafka โดยตรง (ถ้าเปิดใช้งาน)
    4. เขียนเข้า InfluxDB v2 โดยตรงผ่าน influxdb-client (ไม่ต้องใช้ Telegraf)
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

try:
    from influxdb_client import InfluxDBClient, Point
    from influxdb_client.client.write_api import SYNCHRONOUS
except ImportError:
    InfluxDBClient = None
    Point = None
    SYNCHRONOUS = None

try:
    import paho.mqtt.client as mqtt
except ImportError:
    mqtt = None

try:
    from kafka import KafkaProducer
except ImportError:
    KafkaProducer = None

logger = logging.getLogger(__name__)


class InfluxDBWriter:
    """Write speeding event records through the official InfluxDB v2 Python client."""

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.url = config.get("url", "").rstrip("/")
        self.organization = config.get("organization", "")
        self.bucket = config.get("bucket", "")
        self.measurement = config.get("measurement", "speeding_events")
        self.token = config.get("token", "")
        self.client = None
        self.write_api = None

        if self.enabled and InfluxDBClient is None:
            logger.warning("ไม่พบ influxdb-client — ติดตั้งด้วย pip install -r requirements.txt")
            self.enabled = False
        elif self.enabled and not all((self.url, self.organization, self.bucket, self.token)):
            logger.warning(
                "ตั้งค่า InfluxDB ไม่ครบ — ต้องมี url, organization, bucket และ token",
            )
            self.enabled = False
        elif self.enabled:
            try:
                self.client = InfluxDBClient(url=self.url, token=self.token, org=self.organization, timeout=10_000)
                self.write_api = self.client.write_api(write_options=SYNCHRONOUS)
            except Exception as error:
                logger.warning("สร้าง InfluxDB client ไม่สำเร็จ: %s", error)
                self.enabled = False

    def _point(self, event: dict):
        location = event.get("location", {})
        point = Point(self.measurement)
        for key in ("camera_id", "type", "event_type"):
            point.tag(key, str(event.get(key, "unknown")))
        point.field("vehicle_id", int(event["vehicle_id"]))
        point.field("speed_kmh", float(event["speed_kmh"]))
        point.field("speed_limit_kmh", float(event["speed_limit_kmh"]))
        point.field("location_x", float(location["x"]))
        point.field("location_y", float(location["y"]))
        if event.get("snapshot_url"):
            point.field("snapshot_url", str(event["snapshot_url"]))

        try:
            dt = datetime.fromisoformat(str(event.get("timestamp")).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
        except (TypeError, ValueError):
            dt = datetime.now(timezone.utc)
        return point.time(dt)

    def write(self, event: dict):
        if not self.enabled or self.write_api is None:
            return
        try:
            self.write_api.write(bucket=self.bucket, org=self.organization, record=self._point(event))
        except Exception as error:
            logger.warning("เขียนข้อมูลไป InfluxDB ไม่สำเร็จ: %s", error)

    def close(self):
        if self.client is not None:
            self.client.close()


class KafkaWriter:
    """Publish event JSON to a Kafka topic without authentication by default."""

    def __init__(self, config: dict):
        self.enabled = config.get("enabled", False)
        self.topic = config.get("topic", "traffic/events")
        self.producer = None

        if self.enabled and KafkaProducer is None:
            logger.warning("ไม่พบ kafka-python — ติดตั้งด้วย pip install -r requirements.txt")
            self.enabled = False
        elif self.enabled and not config.get("bootstrap_servers"):
            logger.warning("ตั้งค่า Kafka ไม่ครบ — ต้องมี bootstrap_servers")
            self.enabled = False
        elif self.enabled:
            try:
                self.producer = KafkaProducer(
                    bootstrap_servers=config["bootstrap_servers"],
                    # Kafka ปลายทางนี้ไม่ใช้ username/password จึงใช้ PLAINTEXT
                    security_protocol=config.get("security_protocol", "PLAINTEXT"),
                    acks=config.get("acks", "all"),
                    client_id=config.get("client_id", "car-detect-speed"),
                )
                logger.info("เชื่อมต่อ Kafka bootstrap server %s สำเร็จ", config["bootstrap_servers"])
            except Exception as error:
                logger.warning("เชื่อมต่อ Kafka ไม่สำเร็จ (%s) — จะบันทึกลง local log อย่างเดียว", error)
                self.enabled = False
                self.producer = None

    @staticmethod
    def _report_error(error):
        logger.warning("Kafka publish ไม่สำเร็จ: %s", error)

    def write(self, payload: str):
        if not self.enabled or self.producer is None:
            return
        try:
            future = self.producer.send(self.topic, payload.encode("utf-8"))
            future.add_errback(self._report_error)
        except Exception as error:
            logger.warning("Kafka publish ไม่สำเร็จ: %s", error)

    def close(self):
        if self.producer is not None:
            try:
                self.producer.flush(timeout=10)
            except Exception as error:
                logger.warning("ส่งข้อมูล Kafka ที่ค้างอยู่ไม่สำเร็จ: %s", error)
            finally:
                self.producer.close(timeout=10)


class EventPublisher:
    def __init__(
        self,
        mqtt_config: dict,
        local_log_path: str,
        influxdb_config: Optional[dict] = None,
        kafka_config: Optional[dict] = None,
    ):
        self.topic = mqtt_config.get("topic", "traffic/events")
        self.local_log_path = local_log_path
        self.client = None
        self.enabled = mqtt_config.get("enabled", False)
        self.influxdb = InfluxDBWriter(influxdb_config or {})
        self.kafka = KafkaWriter(kafka_config or {})

        os.makedirs(os.path.dirname(local_log_path) or ".", exist_ok=True)

        if self.enabled and mqtt is None:
            logger.warning("เปิด mqtt.enabled ไว้แต่ไม่ได้ติดตั้ง paho-mqtt (pip install paho-mqtt) — จะบันทึกลง local log อย่างเดียว")
            self.enabled = False

        if self.enabled:
            try:
                try:
                    # paho-mqtt >= 2.0 ต้องระบุ callback API version
                    self.client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
                except AttributeError:
                    # paho-mqtt เวอร์ชันเก่ากว่า 2.0
                    self.client = mqtt.Client()
                self.client.connect(mqtt_config["host"], mqtt_config.get("port", 1883), keepalive=30)
                self.client.loop_start()
                logger.info(f"เชื่อมต่อ MQTT broker {mqtt_config['host']}:{mqtt_config.get('port', 1883)} สำเร็จ")
            except Exception as e:
                logger.warning(f"เชื่อมต่อ MQTT ไม่สำเร็จ ({e}) — จะบันทึกลง local log อย่างเดียว")
                self.enabled = False
                self.client = None

    def publish(self, event: dict):
        payload = json.dumps(event, ensure_ascii=False)

        # บันทึก local เสมอ ไม่ว่า MQTT จะพร้อมหรือไม่
        with open(self.local_log_path, "a", encoding="utf-8") as f:
            f.write(payload + "\n")

        if self.enabled and self.client is not None:
            try:
                self.client.publish(self.topic, payload, qos=1)
            except Exception as e:
                logger.warning(f"MQTT publish ไม่สำเร็จ: {e}")

        self.kafka.write(payload)
        self.influxdb.write(event)

    def close(self):
        if self.client is not None:
            self.client.loop_stop()
            self.client.disconnect()
        self.kafka.close()
        self.influxdb.close()
