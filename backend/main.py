from pathlib import Path
from collections import Counter
from uuid import uuid4
from datetime import datetime
import json
import shutil
import sqlite3
import threading
import time

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from algorithm.plate_recognizer import recognize_plate_real
from algorithm.owner_gesture_recognizer import recognize_owner_gesture_image
from algorithm.traffic_gesture_recognizer import recognize_traffic_gesture_image


app = FastAPI(title="智能车载视觉感知与告警系统")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
DATA_DIR = BASE_DIR / "data"
DB_PATH = DATA_DIR / "app.db"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")


RTSP_SOURCES = [
    {"id": "live1", "name": "桥面", "url": "rtsp://10.126.59.120:8554/live/live1"},
    {"id": "live2", "name": "停车场出口", "url": "rtsp://10.126.59.120:8554/live/live2"},
    {"id": "live3", "name": "行人检测", "url": "rtsp://10.126.59.120:8554/live/live3"},
    {"id": "live4", "name": "消防车识别", "url": "rtsp://10.126.59.120:8554/live/live4"},
    {"id": "live5", "name": "桥出口", "url": "rtsp://10.126.59.120:8554/live/live5"},
    {"id": "live6", "name": "桥入口", "url": "rtsp://10.126.59.120:8554/live/live6"},
    {"id": "live7", "name": "道路2", "url": "rtsp://10.126.59.120:8554/live/live7"},
    {"id": "live8", "name": "隧道事故识别", "url": "rtsp://10.126.59.120:8554/live/live8"},
    {"id": "live9", "name": "隧道车辆数量", "url": "rtsp://10.126.59.120:8554/live/live9"},
    {"id": "live10", "name": "道路3", "url": "rtsp://10.126.59.120:8554/live/live10"},
    {"id": "live11", "name": "停车场入口", "url": "rtsp://10.126.59.120:8554/live/live11"},
    {"id": "live12", "name": "道路1", "url": "rtsp://10.126.59.120:8554/live/live12"},
]


class GestureRequest(BaseModel):
    gesture: str


class RtspRecognizeRequest(BaseModel):
    source_id: str
    use_mock_frame: bool = True


class StreamRecognizeRequest(BaseModel):
    source_id: str
    task_type: str = "plate"
    frame_count: int = 50
    sample_interval: int = 5
    use_mock_frame: bool = False
    custom_rtsp_url: str | None = None


class MonitorStartRequest(BaseModel):
    task_type: str = "all"
    interval_seconds: int = 30
    frame_count: int = 20
    sample_interval: int = 5
    use_mock_frame: bool = False
    source_ids: list[str] | None = None


MONITOR_LOCK = threading.RLock()
MONITOR_STOP_EVENT = threading.Event()
MONITOR_THREAD: threading.Thread | None = None
MONITOR_STATE = {
    "running": False,
    "task_type": "all",
    "interval_seconds": 30,
    "frame_count": 20,
    "sample_interval": 5,
    "use_mock_frame": False,
    "source_ids": [],
    "started_at": "",
    "stopped_at": "",
    "last_round_at": "",
    "next_round_after": "",
    "rounds_completed": 0,
    "total_records_created": 0,
    "total_alerts_created": 0,
    "source_status": {},
    "recent_events": [],
}


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def ensure_column_exists(
    conn: sqlite3.Connection,
    table_name: str,
    column_name: str,
    column_definition: str,
):
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row[1] for row in cursor.fetchall()]

    if column_name not in columns:
        cursor.execute(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}"
        )
        conn.commit()


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS recognition_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_type TEXT NOT NULL,
            input_type TEXT NOT NULL,
            original_filename TEXT,
            saved_filename TEXT,
            image_url TEXT,
            output_image_url TEXT,
            result_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS alert_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level TEXT NOT NULL,
            event_type TEXT NOT NULL,
            summary TEXT NOT NULL,
            reason TEXT,
            suggestion TEXT,
            status TEXT NOT NULL,
            related_record_id INTEGER,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS operation_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            detail TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS vehicle_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            system_awake INTEGER NOT NULL,
            current_function TEXT NOT NULL,
            volume INTEGER NOT NULL,
            temperature INTEGER NOT NULL,
            phone_status TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        INSERT OR IGNORE INTO vehicle_state (
            id,
            system_awake,
            current_function,
            volume,
            temperature,
            phone_status,
            updated_at
        )
        VALUES (
            1,
            0,
            'home',
            50,
            24,
            '空闲',
            ?
        )
        """,
        (now_text(),),
    )

    conn.commit()

    ensure_column_exists(
        conn=conn,
        table_name="recognition_records",
        column_name="output_image_url",
        column_definition="TEXT",
    )

    conn.close()


def insert_operation_log(action: str, detail: dict | str | None = None) -> int:
    if isinstance(detail, dict):
        detail_text = json.dumps(detail, ensure_ascii=False)
    elif detail is None:
        detail_text = ""
    else:
        detail_text = str(detail)

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO operation_logs (
            action,
            detail,
            created_at
        )
        VALUES (?, ?, ?)
        """,
        (
            action,
            detail_text,
            now_text(),
        ),
    )

    conn.commit()
    log_id = cursor.lastrowid
    conn.close()

    return log_id


def insert_alert_event(
    level: str,
    event_type: str,
    summary: str,
    reason: str,
    suggestion: str,
    related_record_id: int | None = None,
    status: str = "未处理",
) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO alert_events (
            level,
            event_type,
            summary,
            reason,
            suggestion,
            status,
            related_record_id,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            level,
            event_type,
            summary,
            reason,
            suggestion,
            status,
            related_record_id,
            now_text(),
        ),
    )

    conn.commit()
    alert_id = cursor.lastrowid
    conn.close()

    return alert_id


def insert_recognition_record(
    task_type: str,
    input_type: str,
    original_filename: str,
    saved_filename: str,
    image_url: str,
    output_image_url: str,
    result: dict,
) -> int:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO recognition_records (
            task_type,
            input_type,
            original_filename,
            saved_filename,
            image_url,
            output_image_url,
            result_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            task_type,
            input_type,
            original_filename,
            saved_filename,
            image_url,
            output_image_url,
            json.dumps(result, ensure_ascii=False),
            now_text(),
        ),
    )

    conn.commit()
    record_id = cursor.lastrowid
    conn.close()

    return record_id


def create_annotated_plate_image(
    input_path: Path,
    output_path: Path,
    confidence: float = 0.92,
) -> dict:
    """
    真实车牌识别版本：
    使用 HyperLPR3 完成车牌检测 + OCR 识别。
    confidence 参数只用于“模拟低置信度告警”测试：
    - 如果 confidence < 0.6，则强制把模型输出置信度改低，用于触发告警。
    - 正常识别时使用模型自己的置信度。
    """
    try:
        force_confidence = confidence if confidence < 0.6 else None

        return recognize_plate_real(
            input_path=input_path,
            output_path=output_path,
            force_confidence=force_confidence,
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"真实车牌识别失败：{error}"
        )
    image = cv2.imread(str(input_path))

    if image is None:
        raise HTTPException(status_code=400, detail="图片读取失败，请检查图片格式是否正确")

    height, width = image.shape[:2]

    x1 = int(width * 0.35)
    y1 = int(height * 0.65)
    x2 = int(width * 0.65)
    y2 = int(height * 0.75)

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    plate_number = "京A12345"
    plate_number_for_image = "JingA12345"
    plate_color = "蓝牌"

    box_color = (0, 255, 0) if confidence >= 0.6 else (0, 165, 255)

    cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 3)

    label = f"{plate_number_for_image} {confidence:.2f}"

    cv2.putText(
        image,
        label,
        (x1, max(y1 - 10, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        box_color,
        2,
        cv2.LINE_AA,
    )

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise HTTPException(status_code=500, detail="标注结果图保存失败")

    return {
        "plates": [
            {
                "plate_number": plate_number,
                "plate_color": plate_color,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2],
            }
        ]
    }


def create_mock_rtsp_frame(source_name: str, output_path: Path):
    image = np.zeros((720, 1280, 3), dtype=np.uint8)
    image[:] = (35, 35, 35)

    cv2.rectangle(image, (0, 420), (1280, 720), (55, 55, 55), -1)
    cv2.line(image, (0, 570), (1280, 570), (255, 255, 255), 5)
    cv2.rectangle(image, (480, 430), (800, 560), (20, 20, 160), -1)
    cv2.rectangle(image, (560, 535), (720, 565), (255, 255, 255), -1)

    cv2.putText(
        image,
        f"RTSP Source: {source_name}",
        (40, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        image,
        "Mock frame for demo",
        (40, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    success = cv2.imwrite(str(output_path), image)

    if not success:
        raise HTTPException(status_code=500, detail="模拟视频帧保存失败")


def capture_rtsp_frame(source: dict, output_path: Path):
    cap = cv2.VideoCapture(source["url"], cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise HTTPException(
            status_code=400,
            detail=f"无法打开 RTSP 视频流：{source['name']}。可勾选模拟帧用于离线演示。",
        )

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise HTTPException(
            status_code=400,
            detail=f"读取 RTSP 视频帧失败：{source['name']}",
        )

    success = cv2.imwrite(str(output_path), frame)

    if not success:
        raise HTTPException(status_code=500, detail="RTSP 视频帧保存失败")




def clamp_int(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(int(value), max_value))


def get_stream_source(source_id: str, custom_rtsp_url: str | None = None) -> dict:
    if custom_rtsp_url:
        return {
            "id": "custom",
            "name": "自定义 RTSP 视频源",
            "url": custom_rtsp_url,
        }

    source = next((item for item in RTSP_SOURCES if item["id"] == source_id), None)

    if source is None:
        raise HTTPException(status_code=404, detail="未找到指定 RTSP 视频源")

    return source


def save_frame_image(frame, filename_prefix: str) -> dict:
    saved_filename = f"{filename_prefix}_{uuid4().hex}.jpg"
    saved_path = UPLOAD_DIR / saved_filename

    success = cv2.imwrite(str(saved_path), frame)

    if not success:
        raise HTTPException(status_code=500, detail="视频帧保存失败")

    return {
        "saved_filename": saved_filename,
        "saved_path": saved_path,
        "image_url": f"/uploads/{saved_filename}",
    }


def capture_stream_sampled_frames(
    source: dict,
    frame_count: int,
    sample_interval: int,
    use_mock_frame: bool,
) -> tuple[list[dict], int]:
    """
    从 RTSP 视频流中连续读取 frame_count 帧，并每隔 sample_interval 帧保存一张抽样帧。
    返回：抽样帧列表、实际读取帧数。
    """
    frame_count = clamp_int(frame_count, 5, 300)
    sample_interval = clamp_int(sample_interval, 1, 60)

    sampled_frames: list[dict] = []

    if use_mock_frame:
        mock_count = max(1, min(10, frame_count // sample_interval))

        for index in range(mock_count):
            saved_filename = f"stream_mock_{source['id']}_{index:03d}_{uuid4().hex}.jpg"
            saved_path = UPLOAD_DIR / saved_filename
            create_mock_rtsp_frame(source["name"], saved_path)
            sampled_frames.append(
                {
                    "frame_index": index * sample_interval,
                    "saved_filename": saved_filename,
                    "saved_path": saved_path,
                    "image_url": f"/uploads/{saved_filename}",
                }
            )

        return sampled_frames, frame_count

    cap = cv2.VideoCapture(source["url"], cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        raise HTTPException(
            status_code=400,
            detail=f"无法打开 RTSP 视频流：{source['name']}。请确认沙盘推流已开启、RTSP 地址正确，或勾选模拟视频帧。",
        )

    frames_read = 0

    for frame_index in range(frame_count):
        ret, frame = cap.read()

        if not ret or frame is None:
            continue

        frames_read += 1

        if frame_index % sample_interval == 0:
            frame_info = save_frame_image(
                frame=frame,
                filename_prefix=f"stream_{source['id']}_{frame_index:03d}",
            )
            frame_info["frame_index"] = frame_index
            sampled_frames.append(frame_info)

    cap.release()

    if not sampled_frames:
        raise HTTPException(
            status_code=400,
            detail=f"已尝试读取 RTSP 视频流，但没有得到可用抽样帧：{source['name']}",
        )

    return sampled_frames, frames_read


def get_result_confidence(task_type: str, result: dict) -> float:
    if task_type == "plate":
        plates = result.get("plates", []) or []
        if not plates:
            return 0.0
        return max(float(plate.get("confidence", 0) or 0) for plate in plates)

    return float(result.get("confidence", 0) or 0)


def recognize_single_sampled_frame(frame_info: dict, task_type: str) -> dict:
    input_path = frame_info["saved_path"]
    output_filename = f"annotated_{task_type}_{frame_info['saved_filename']}"
    output_path = OUTPUT_DIR / output_filename

    if task_type == "plate":
        result = create_annotated_plate_image(
            input_path=input_path,
            output_path=output_path,
            confidence=0.92,
        )
    elif task_type == "owner_gesture":
        result = recognize_owner_gesture_image(
            input_path=input_path,
            output_path=output_path,
        )
    elif task_type == "traffic_gesture":
        result = recognize_traffic_gesture_image(
            input_path=input_path,
            output_path=output_path,
        )
    else:
        raise HTTPException(status_code=400, detail="不支持的连续帧识别任务类型")

    confidence = get_result_confidence(task_type, result)

    return {
        "frame_index": frame_info["frame_index"],
        "saved_filename": frame_info["saved_filename"],
        "image_url": frame_info["image_url"],
        "output_filename": output_filename,
        "output_image_url": f"/outputs/{output_filename}",
        "confidence": confidence,
        "result": result,
    }


def compact_frame_result(task_type: str, item: dict) -> dict:
    result = item.get("result", {})
    compact = {
        "frame_index": item.get("frame_index"),
        "image_url": item.get("image_url"),
        "output_image_url": item.get("output_image_url"),
        "confidence": item.get("confidence", 0),
    }

    if task_type == "plate":
        plates = result.get("plates", []) or []
        compact["plates_count"] = len(plates)
        compact["best_plate"] = max(
            plates,
            key=lambda plate: float(plate.get("confidence", 0) or 0),
            default=None,
        )
        compact["plates"] = [
            {
                "plate_number": plate.get("plate_number", ""),
                "plate_color": plate.get("plate_color", ""),
                "confidence": plate.get("confidence", 0),
                "bbox": plate.get("bbox"),
            }
            for plate in plates
        ]
    else:
        compact["gesture"] = result.get("gesture")
        compact["gesture_name"] = result.get("gesture_name")

    return compact


def aggregate_plate_frame_results(frame_results: list[dict]) -> dict:
    """
    多帧、多车牌聚合：
    - 每个抽样帧可能识别出 0 到多个车牌。
    - 按 plate_number 聚合同一车牌在多帧中的结果。
    - 同一车牌保留最高置信度结果，同时记录出现次数和出现帧序号。
    - 最终 plates 字段保留多个车牌，不再只返回单帧最高置信度结果。
    """
    best = max(
        frame_results,
        key=lambda item: (
            len(item.get("result", {}).get("plates", []) or []),
            float(item.get("confidence", 0) or 0),
        ),
    )

    aggregated: dict[str, dict] = {}

    for item in frame_results:
        frame_index = item.get("frame_index")
        image_url = item.get("image_url", "")
        output_image_url = item.get("output_image_url", "")
        plates = item.get("result", {}).get("plates", []) or []

        for plate in plates:
            plate_number = str(plate.get("plate_number") or "").strip()
            if not plate_number:
                # 未得到车牌号时，按定位框近似区分，避免把多个未知车牌误合并。
                plate_number = f"UNKNOWN_{frame_index}_{len(aggregated) + 1}"

            confidence = float(plate.get("confidence", 0) or 0)

            if plate_number not in aggregated:
                item_copy = dict(plate)
                item_copy["plate_number"] = plate_number
                item_copy["confidence"] = confidence
                item_copy["appear_count"] = 1
                item_copy["frame_indices"] = [frame_index]
                item_copy["best_frame_index"] = frame_index
                item_copy["best_image_url"] = image_url
                item_copy["best_output_image_url"] = output_image_url
                aggregated[plate_number] = item_copy
                continue

            existing = aggregated[plate_number]
            existing["appear_count"] = int(existing.get("appear_count", 0)) + 1
            existing.setdefault("frame_indices", []).append(frame_index)

            if confidence > float(existing.get("confidence", 0) or 0):
                existing.update(dict(plate))
                existing["plate_number"] = plate_number
                existing["confidence"] = confidence
                existing["best_frame_index"] = frame_index
                existing["best_image_url"] = image_url
                existing["best_output_image_url"] = output_image_url

    aggregated_plates = sorted(
        aggregated.values(),
        key=lambda plate: (
            int(plate.get("appear_count", 0)),
            float(plate.get("confidence", 0) or 0),
        ),
        reverse=True,
    )

    final_result = dict(best["result"])
    final_result["model"] = final_result.get("model", "HyperLPR3")
    final_result["plates"] = aggregated_plates
    final_result["plate_count"] = len(aggregated_plates)
    final_result["stream_strategy"] = "连续帧抽样 + 多车牌多帧聚合"
    final_result["best_frame_index"] = best["frame_index"]
    final_result["best_frame_plate_count"] = len(best.get("result", {}).get("plates", []) or [])
    final_result["sampled_frames"] = len(frame_results)
    final_result["frame_results"] = [compact_frame_result("plate", item) for item in frame_results]

    return {
        "best": best,
        "final_result": final_result,
    }


def aggregate_gesture_frame_results(task_type: str, frame_results: list[dict]) -> dict:
    valid_items = [
        item for item in frame_results
        if item.get("result", {}).get("gesture") not in (None, "", "unknown")
    ]

    if valid_items:
        counter = Counter(item["result"].get("gesture") for item in valid_items)

        def gesture_score(gesture: str):
            same_gesture_items = [
                item for item in valid_items if item["result"].get("gesture") == gesture
            ]
            avg_confidence = sum(item.get("confidence", 0) for item in same_gesture_items) / len(same_gesture_items)
            return counter[gesture], avg_confidence

        selected_gesture = max(counter.keys(), key=gesture_score)
        candidates = [
            item for item in valid_items if item["result"].get("gesture") == selected_gesture
        ]
        best = max(candidates, key=lambda item: float(item.get("confidence", 0) or 0))
        strategy = "连续帧抽样 + 手势多数投票"
    else:
        best = max(frame_results, key=lambda item: float(item.get("confidence", 0) or 0))
        counter = Counter()
        strategy = "连续帧抽样 + 最高置信度兜底"

    final_result = dict(best["result"])
    final_result["stream_strategy"] = strategy
    final_result["best_frame_index"] = best["frame_index"]
    final_result["sampled_frames"] = len(frame_results)
    final_result["vote_counts"] = dict(counter)
    final_result["frame_results"] = [compact_frame_result(task_type, item) for item in frame_results]

    return {
        "best": best,
        "final_result": final_result,
    }


def recognize_stream_for_task(task_type: str, sampled_frames: list[dict]) -> dict:
    frame_results = []

    for frame_info in sampled_frames:
        try:
            frame_results.append(
                recognize_single_sampled_frame(
                    frame_info=frame_info,
                    task_type=task_type,
                )
            )
        except Exception as error:
            frame_results.append(
                {
                    "frame_index": frame_info["frame_index"],
                    "saved_filename": frame_info["saved_filename"],
                    "image_url": frame_info["image_url"],
                    "output_filename": "",
                    "output_image_url": "",
                    "confidence": 0.0,
                    "result": {
                        "model": task_type,
                        "error": str(error),
                    },
                }
            )

    if not frame_results:
        raise HTTPException(status_code=500, detail="连续帧识别没有产生任何结果")

    if task_type == "plate":
        return aggregate_plate_frame_results(frame_results)

    return aggregate_gesture_frame_results(task_type, frame_results)


def maybe_create_stream_alert(record_id: int, task_type: str, result: dict) -> int | None:
    if task_type == "plate":
        return maybe_create_low_confidence_alert(record_id=record_id, result=result)

    if task_type in {"owner_gesture", "traffic_gesture"}:
        confidence = float(result.get("confidence", 0) or 0)
        gesture = result.get("gesture", "unknown")

        if gesture == "unknown" or confidence < 0.6:
            event_type = f"{task_type}_stream_low_confidence"
            summary = "连续帧手势识别置信度偏低"
            reason = f"本次连续帧识别结果为 {result.get('gesture_name', '未知')}，置信度 {confidence:.2f}。"
            suggestion = "建议调整摄像头角度、光照和目标距离，并使用包含完整手势动作的视频流。"

            return insert_alert_event(
                level="warning",
                event_type=event_type,
                summary=summary,
                reason=reason,
                suggestion=suggestion,
                related_record_id=record_id,
            )

    if task_type == "all":
        tasks = result.get("tasks", {})
        failed_tasks = []

        plate_result = tasks.get("plate", {}).get("result", {})
        if not plate_result.get("plates"):
            failed_tasks.append("车牌")

        for key, label in [("traffic_gesture", "交警手势"), ("owner_gesture", "车主手势")]:
            gesture_result = tasks.get(key, {}).get("result", {})
            if gesture_result.get("gesture") in (None, "", "unknown"):
                failed_tasks.append(label)

        if failed_tasks:
            return insert_alert_event(
                level="warning",
                event_type="stream_multi_task_partial_failed",
                summary="综合连续帧识别存在未识别目标",
                reason=f"以下任务未得到稳定识别结果：{'、'.join(failed_tasks)}。",
                suggestion="建议切换更合适的视频源，或分别使用车牌、交警、车主手势专用摄像头进行识别。",
                related_record_id=record_id,
            )

    return None

def maybe_create_low_confidence_alert(record_id: int, result: dict) -> int | None:
    plates = result.get("plates", [])

    if not plates:
        return insert_alert_event(
            level="warning",
            event_type="plate_recognition_failed",
            summary="车牌识别未检测到有效车牌",
            reason="当前图片未返回任何车牌检测结果，可能由图片模糊、角度过大或车辆距离过远导致。",
            suggestion="建议更换更清晰的道路图片，或检查摄像头角度与光照条件。",
            related_record_id=record_id,
        )

    min_confidence = min(float(plate.get("confidence", 0)) for plate in plates)

    if min_confidence < 0.6:
        return insert_alert_event(
            level="warning",
            event_type="low_confidence",
            summary="车牌识别置信度偏低",
            reason=f"本次识别最低置信度为 {min_confidence:.2f}，低于系统阈值 0.60。",
            suggestion="建议检查图片清晰度、车牌遮挡情况、光照条件，必要时切换更适合的识别模型。",
            related_record_id=record_id,
        )

    return None


def row_to_record(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "task_type": row["task_type"],
        "input_type": row["input_type"],
        "original_filename": row["original_filename"],
        "saved_filename": row["saved_filename"],
        "image_url": row["image_url"],
        "output_image_url": row["output_image_url"],
        "result": json.loads(row["result_json"]),
        "created_at": row["created_at"],
    }


def row_to_alert(row: sqlite3.Row) -> dict:
    return {
        "id": row["id"],
        "level": row["level"],
        "event_type": row["event_type"],
        "summary": row["summary"],
        "reason": row["reason"],
        "suggestion": row["suggestion"],
        "status": row["status"],
        "related_record_id": row["related_record_id"],
        "created_at": row["created_at"],
    }


def row_to_log(row: sqlite3.Row) -> dict:
    detail_text = row["detail"] or ""

    try:
        detail = json.loads(detail_text) if detail_text else {}
    except json.JSONDecodeError:
        detail = detail_text

    return {
        "id": row["id"],
        "action": row["action"],
        "detail": detail,
        "created_at": row["created_at"],
    }


def get_vehicle_state() -> dict:
    conn = get_db_connection()
    cursor = conn.cursor()

    row = cursor.execute(
        """
        SELECT
            id,
            system_awake,
            current_function,
            volume,
            temperature,
            phone_status,
            updated_at
        FROM vehicle_state
        WHERE id = 1
        """
    ).fetchone()

    conn.close()

    if row is None:
        raise HTTPException(status_code=500, detail="车辆状态初始化失败")

    return {
        "system_awake": bool(row["system_awake"]),
        "current_function": row["current_function"],
        "volume": row["volume"],
        "temperature": row["temperature"],
        "phone_status": row["phone_status"],
        "updated_at": row["updated_at"],
    }


def update_vehicle_state(state: dict):
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE vehicle_state
        SET
            system_awake = ?,
            current_function = ?,
            volume = ?,
            temperature = ?,
            phone_status = ?,
            updated_at = ?
        WHERE id = 1
        """,
        (
            1 if state["system_awake"] else 0,
            state["current_function"],
            state["volume"],
            state["temperature"],
            state["phone_status"],
            now_text(),
        ),
    )

    conn.commit()
    conn.close()


def apply_owner_gesture(gesture: str) -> dict:
    gesture_map = {
        "open_palm": "手掌张开",
        "fist": "握拳",
        "swipe_left": "左滑",
        "swipe_right": "右滑",
        "thumb_up": "拇指向上",
        "thumb_down": "拇指向下",
        "wave": "挥手",
        "circle": "单指画圈",
    }

    if gesture not in gesture_map:
        raise HTTPException(status_code=400, detail="不支持的车主手势")

    state = get_vehicle_state()
    functions = ["home", "music", "air_conditioner", "phone", "navigation"]

    action = ""
    description = ""

    if gesture == "open_palm":
        state["system_awake"] = True
        action = "wake_system"
        description = "系统已唤醒"

    elif gesture == "fist":
        action = "confirm"
        description = f"确认当前功能：{state['current_function']}"

    elif gesture == "swipe_left":
        index = functions.index(state["current_function"])
        state["current_function"] = functions[(index - 1) % len(functions)]
        action = "previous_function"
        description = f"切换到上一个功能：{state['current_function']}"

    elif gesture == "swipe_right":
        index = functions.index(state["current_function"])
        state["current_function"] = functions[(index + 1) % len(functions)]
        action = "next_function"
        description = f"切换到下一个功能：{state['current_function']}"

    elif gesture == "thumb_up":
        if state["current_function"] == "phone":
            state["phone_status"] = "已接听"
            action = "answer_phone"
            description = "电话已接听"
        else:
            state["volume"] = min(100, state["volume"] + 5)
            action = "volume_up"
            description = f"音量增加至 {state['volume']}"

    elif gesture == "thumb_down":
        if state["current_function"] == "phone":
            state["phone_status"] = "已挂断"
            action = "hang_up_phone"
            description = "电话已挂断"
        else:
            state["volume"] = max(0, state["volume"] - 5)
            action = "volume_down"
            description = f"音量降低至 {state['volume']}"

    elif gesture == "wave":
        state["current_function"] = "home"
        action = "back_home"
        description = "已返回主页"

    elif gesture == "circle":
        if state["current_function"] == "air_conditioner":
            state["temperature"] = min(30, state["temperature"] + 1)
            action = "temperature_up"
            description = f"空调温度调高至 {state['temperature']}℃"
        else:
            state["volume"] = min(100, state["volume"] + 10)
            action = "volume_adjust"
            description = f"音量快速调节至 {state['volume']}"

    update_vehicle_state(state)

    return {
        "gesture": gesture,
        "gesture_name": gesture_map[gesture],
        "action": action,
        "description": description,
        "confidence": 0.91,
        "vehicle_state": get_vehicle_state(),
    }


def recognize_traffic_gesture(gesture: str) -> dict:
    traffic_map = {
        "stop": {
            "gesture_name": "停止信号",
            "command": "车辆停止通行",
            "confidence": 0.93,
        },
        "straight": {
            "gesture_name": "直行信号",
            "command": "车辆允许直行",
            "confidence": 0.90,
        },
        "left_turn": {
            "gesture_name": "左转弯信号",
            "command": "车辆允许左转",
            "confidence": 0.89,
        },
        "left_turn_wait": {
            "gesture_name": "左转弯待转信号",
            "command": "车辆进入待转区",
            "confidence": 0.86,
        },
        "right_turn": {
            "gesture_name": "右转弯信号",
            "command": "车辆允许右转",
            "confidence": 0.88,
        },
        "lane_change": {
            "gesture_name": "变道信号",
            "command": "车辆按指令变道",
            "confidence": 0.84,
        },
        "slow_down": {
            "gesture_name": "减速慢行信号",
            "command": "车辆减速慢行",
            "confidence": 0.87,
        },
        "pull_over": {
            "gesture_name": "靠边停车信号",
            "command": "车辆靠边停车",
            "confidence": 0.85,
        },
    }

    if gesture not in traffic_map:
        raise HTTPException(status_code=400, detail="不支持的交警手势")

    result = traffic_map[gesture]

    return {
        "gesture": gesture,
        "gesture_name": result["gesture_name"],
        "traffic_command": result["command"],
        "confidence": result["confidence"],
        "keypoints": [
            {"name": "left_shoulder", "x": 0.35, "y": 0.42},
            {"name": "right_shoulder", "x": 0.65, "y": 0.42},
            {"name": "left_wrist", "x": 0.25, "y": 0.35},
            {"name": "right_wrist", "x": 0.75, "y": 0.36},
        ],
    }


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def root():
    return {
        "message": "智能车载视觉感知与告警系统后端已启动"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "ok",
        "message": "backend is running"
    }


@app.get("/api/self-check")
def self_check():
    checks = {
        "backend": True,
        "database": DB_PATH.exists(),
        "upload_dir": UPLOAD_DIR.exists(),
        "output_dir": OUTPUT_DIR.exists(),
        "rtsp_sources": len(RTSP_SOURCES),
    }

    return {
        "status": "success",
        "checks": checks,
        "message": "系统基础环境检查完成",
    }


@app.get("/api/dashboard/summary")
def get_dashboard_summary():
    conn = get_db_connection()
    cursor = conn.cursor()

    today = datetime.now().strftime("%Y-%m-%d")

    total_records = cursor.execute(
        "SELECT COUNT(*) AS count FROM recognition_records"
    ).fetchone()["count"]

    today_records = cursor.execute(
        """
        SELECT COUNT(*) AS count
        FROM recognition_records
        WHERE created_at LIKE ?
        """,
        (f"{today}%",),
    ).fetchone()["count"]

    total_alerts = cursor.execute(
        "SELECT COUNT(*) AS count FROM alert_events"
    ).fetchone()["count"]

    unresolved_alerts = cursor.execute(
        """
        SELECT COUNT(*) AS count
        FROM alert_events
        WHERE status = '未处理'
        """
    ).fetchone()["count"]

    conn.close()

    return {
        "status": "success",
        "summary": {
            "total_records": total_records,
            "today_records": today_records,
            "total_alerts": total_alerts,
            "unresolved_alerts": unresolved_alerts,
        }
    }


@app.get("/api/rtsp/sources")
def list_rtsp_sources():
    return {
        "status": "success",
        "sources": RTSP_SOURCES,
    }


@app.post("/api/rtsp/recognize")
def recognize_rtsp_frame(request: RtspRecognizeRequest):
    source = next((item for item in RTSP_SOURCES if item["id"] == request.source_id), None)

    if source is None:
        raise HTTPException(status_code=404, detail="未找到指定 RTSP 视频源")

    saved_filename = f"rtsp_{source['id']}_{uuid4().hex}.jpg"
    saved_path = UPLOAD_DIR / saved_filename

    if request.use_mock_frame:
        create_mock_rtsp_frame(source["name"], saved_path)
    else:
        capture_rtsp_frame(source, saved_path)

    image_url = f"/uploads/{saved_filename}"

    output_filename = f"annotated_{saved_filename}"
    output_path = OUTPUT_DIR / output_filename
    output_image_url = f"/outputs/{output_filename}"

    recognition_result = create_annotated_plate_image(
        input_path=saved_path,
        output_path=output_path,
        confidence=0.89,
    )

    recognition_result["source"] = {
        "id": source["id"],
        "name": source["name"],
        "url": source["url"],
        "mode": "mock_frame" if request.use_mock_frame else "real_rtsp",
    }

    record_id = insert_recognition_record(
        task_type="plate",
        input_type="rtsp",
        original_filename=source["name"],
        saved_filename=saved_filename,
        image_url=image_url,
        output_image_url=output_image_url,
        result=recognition_result,
    )

    insert_operation_log(
        action="rtsp_plate_recognition",
        detail={
            "record_id": record_id,
            "source_id": source["id"],
            "source_name": source["name"],
            "use_mock_frame": request.use_mock_frame,
        },
    )

    return {
        "status": "success",
        "record_id": record_id,
        "task_type": "plate",
        "input_type": "rtsp",
        "source": source,
        "image_url": image_url,
        "output_image_url": output_image_url,
        "result": recognition_result,
    }



def process_stream_recognition_job(
    source: dict,
    task_type: str,
    frame_count: int,
    sample_interval: int,
    use_mock_frame: bool,
    log_action: str = "stream_frame_recognition",
) -> dict:
    supported_tasks = {"plate", "traffic_gesture", "owner_gesture", "all"}

    if task_type not in supported_tasks:
        raise HTTPException(
            status_code=400,
            detail="task_type 只支持 plate、traffic_gesture、owner_gesture、all",
        )

    frame_count = clamp_int(frame_count, 5, 300)
    sample_interval = clamp_int(sample_interval, 1, 60)

    sampled_frames, frames_read = capture_stream_sampled_frames(
        source=source,
        frame_count=frame_count,
        sample_interval=sample_interval,
        use_mock_frame=use_mock_frame,
    )

    tasks_to_run = (
        ["plate", "traffic_gesture", "owner_gesture"]
        if task_type == "all"
        else [task_type]
    )

    task_outputs: dict[str, dict] = {}

    for task in tasks_to_run:
        task_outputs[task] = recognize_stream_for_task(
            task_type=task,
            sampled_frames=sampled_frames,
        )

    if task_type == "all":
        first_best = next(
            (output["best"] for output in task_outputs.values() if output.get("best")),
            None,
        )

        if first_best is None:
            raise HTTPException(status_code=500, detail="综合连续帧识别未得到有效结果")

        final_result = {
            "model": "RTSP Frame Sampling + Multi-task Recognition",
            "stream_strategy": "连续帧抽样 + 多任务分别融合",
            "source": {
                "id": source["id"],
                "name": source["name"],
                "url": source["url"],
                "mode": "mock_frame" if use_mock_frame else "real_rtsp",
            },
            "frame_count_requested": frame_count,
            "frames_read": frames_read,
            "sample_interval": sample_interval,
            "sampled_frames": len(sampled_frames),
            "tasks": {
                task: {
                    "best_frame_index": output["best"].get("frame_index"),
                    "image_url": output["best"].get("image_url"),
                    "output_image_url": output["best"].get("output_image_url"),
                    "confidence": output["best"].get("confidence", 0),
                    "result": output["final_result"],
                }
                for task, output in task_outputs.items()
            },
        }

        best_image_url = first_best.get("image_url", "")
        best_output_image_url = first_best.get("output_image_url", "")
        best_saved_filename = first_best.get("saved_filename", "")
    else:
        output = task_outputs[task_type]
        best = output["best"]
        final_result = output["final_result"]
        final_result["source"] = {
            "id": source["id"],
            "name": source["name"],
            "url": source["url"],
            "mode": "mock_frame" if use_mock_frame else "real_rtsp",
        }
        final_result["frame_count_requested"] = frame_count
        final_result["frames_read"] = frames_read
        final_result["sample_interval"] = sample_interval

        best_image_url = best.get("image_url", "")
        best_output_image_url = best.get("output_image_url", "")
        best_saved_filename = best.get("saved_filename", "")

    record_id = insert_recognition_record(
        task_type=task_type,
        input_type="rtsp_stream" if not use_mock_frame else "mock_stream",
        original_filename=source["name"],
        saved_filename=best_saved_filename,
        image_url=best_image_url,
        output_image_url=best_output_image_url,
        result=final_result,
    )

    alert_id = maybe_create_stream_alert(
        record_id=record_id,
        task_type=task_type,
        result=final_result,
    )

    insert_operation_log(
        action=log_action,
        detail={
            "record_id": record_id,
            "alert_id": alert_id,
            "source_id": source["id"],
            "source_name": source["name"],
            "task_type": task_type,
            "frame_count": frame_count,
            "frames_read": frames_read,
            "sample_interval": sample_interval,
            "sampled_frames": len(sampled_frames),
            "use_mock_frame": use_mock_frame,
        },
    )

    return {
        "status": "success",
        "record_id": record_id,
        "alert_id": alert_id,
        "task_type": task_type,
        "input_type": "rtsp_stream" if not use_mock_frame else "mock_stream",
        "source": source,
        "frame_count": frame_count,
        "frames_read": frames_read,
        "sample_interval": sample_interval,
        "sampled_frames": len(sampled_frames),
        "image_url": best_image_url,
        "output_image_url": best_output_image_url,
        "result": final_result,
    }


def get_monitor_source_list(source_ids: list[str] | None) -> list[dict]:
    if not source_ids or "all" in source_ids:
        return [dict(source) for source in RTSP_SOURCES]

    sources = []
    missing = []

    for source_id in source_ids:
        source = next((item for item in RTSP_SOURCES if item["id"] == source_id), None)
        if source is None:
            missing.append(source_id)
        else:
            sources.append(dict(source))

    if missing:
        raise HTTPException(status_code=404, detail=f"未找到视频源：{', '.join(missing)}")

    if not sources:
        raise HTTPException(status_code=400, detail="至少需要选择一个视频源")

    return sources


def summarize_monitor_result(result: dict) -> dict:
    task_type = result.get("task_type")
    final_result = result.get("result", {}) or {}
    summary = {
        "task_type": task_type,
        "record_id": result.get("record_id"),
        "alert_id": result.get("alert_id"),
        "input_type": result.get("input_type"),
        "frames_read": result.get("frames_read"),
        "sampled_frames": result.get("sampled_frames"),
        "image_url": result.get("image_url"),
        "output_image_url": result.get("output_image_url"),
    }

    if task_type == "plate":
        plates = final_result.get("plates", []) or []
        summary["plate_count"] = len(plates)
        summary["plates"] = plates
    elif task_type in {"traffic_gesture", "owner_gesture"}:
        summary["gesture"] = final_result.get("gesture")
        summary["gesture_name"] = final_result.get("gesture_name")
        summary["confidence"] = final_result.get("confidence", 0)
    elif task_type == "all":
        tasks = final_result.get("tasks", {}) or {}
        plate_result = tasks.get("plate", {}).get("result", {}) or {}
        summary["plate_count"] = len(plate_result.get("plates", []) or [])
        summary["plates"] = plate_result.get("plates", []) or []
        summary["traffic_gesture"] = tasks.get("traffic_gesture", {}).get("result", {}).get("gesture_name")
        summary["owner_gesture"] = tasks.get("owner_gesture", {}).get("result", {}).get("gesture_name")

    return summary


def monitor_add_event(event: dict):
    with MONITOR_LOCK:
        events = MONITOR_STATE.setdefault("recent_events", [])
        events.insert(0, event)
        del events[30:]


def monitor_worker(config: dict):
    sources = config["sources"]

    while not MONITOR_STOP_EVENT.is_set():
        round_started_at = now_text()

        with MONITOR_LOCK:
            MONITOR_STATE["last_round_at"] = round_started_at

        for source in sources:
            if MONITOR_STOP_EVENT.is_set():
                break

            source_id = source["id"]

            with MONITOR_LOCK:
                MONITOR_STATE.setdefault("source_status", {}).setdefault(source_id, {})
                MONITOR_STATE["source_status"][source_id].update(
                    {
                        "source_id": source_id,
                        "source_name": source["name"],
                        "url": source["url"],
                        "status": "running",
                        "last_started_at": now_text(),
                        "last_error": "",
                    }
                )

            try:
                result = process_stream_recognition_job(
                    source=source,
                    task_type=config["task_type"],
                    frame_count=config["frame_count"],
                    sample_interval=config["sample_interval"],
                    use_mock_frame=config["use_mock_frame"],
                    log_action="auto_monitor_stream_recognition",
                )
                summary = summarize_monitor_result(result)

                with MONITOR_LOCK:
                    MONITOR_STATE["source_status"][source_id].update(
                        {
                            "status": "success",
                            "last_finished_at": now_text(),
                            "last_error": "",
                            "last_result": summary,
                            "last_record_id": result.get("record_id"),
                            "last_alert_id": result.get("alert_id"),
                            "last_plate_count": summary.get("plate_count", 0),
                        }
                    )
                    MONITOR_STATE["total_records_created"] += 1
                    if result.get("alert_id"):
                        MONITOR_STATE["total_alerts_created"] += 1

                monitor_add_event(
                    {
                        "time": now_text(),
                        "source_id": source_id,
                        "source_name": source["name"],
                        "status": "success",
                        "record_id": result.get("record_id"),
                        "alert_id": result.get("alert_id"),
                        "summary": summary,
                    }
                )

            except Exception as error:
                detail = getattr(error, "detail", None) or str(error)
                alert_id = insert_alert_event(
                    level="error",
                    event_type="auto_monitor_source_failed",
                    summary="自动监控视频源识别失败",
                    reason=f"视频源 {source['name']}（{source_id}）本轮自动识别失败：{detail}",
                    suggestion="建议检查 RTSP 地址、沙盘推流状态、网络连通性和视频源是否包含目标对象。",
                    related_record_id=None,
                )

                with MONITOR_LOCK:
                    MONITOR_STATE["source_status"][source_id].update(
                        {
                            "status": "error",
                            "last_finished_at": now_text(),
                            "last_error": str(detail),
                            "last_alert_id": alert_id,
                        }
                    )
                    MONITOR_STATE["total_alerts_created"] += 1

                monitor_add_event(
                    {
                        "time": now_text(),
                        "source_id": source_id,
                        "source_name": source["name"],
                        "status": "error",
                        "alert_id": alert_id,
                        "error": str(detail),
                    }
                )

        with MONITOR_LOCK:
            MONITOR_STATE["rounds_completed"] += 1
            MONITOR_STATE["next_round_after"] = f"{config['interval_seconds']} 秒后"

        if MONITOR_STOP_EVENT.wait(config["interval_seconds"]):
            break

    with MONITOR_LOCK:
        MONITOR_STATE["running"] = False
        MONITOR_STATE["stopped_at"] = now_text()
        MONITOR_STATE["next_round_after"] = "已停止"


def get_monitor_state_snapshot() -> dict:
    with MONITOR_LOCK:
        return json.loads(json.dumps(MONITOR_STATE, ensure_ascii=False))


@app.post("/api/stream/recognize")
def recognize_stream_frames(request: StreamRecognizeRequest):
    """
    沙盘视频连续帧识别接口：
    1. 连接 RTSP 沙盘视频源，或使用模拟视频帧兜底
    2. 连续读取 frame_count 帧
    3. 每隔 sample_interval 帧抽样保存
    4. 对抽样帧执行车牌 / 交警手势 / 车主手势识别
    5. 对多帧结果做最高置信度或多数投票融合
    6. 写入识别记录、告警记录、操作日志
    """
    supported_tasks = {"plate", "traffic_gesture", "owner_gesture", "all"}

    if request.task_type not in supported_tasks:
        raise HTTPException(
            status_code=400,
            detail="task_type 只支持 plate、traffic_gesture、owner_gesture、all",
        )

    frame_count = clamp_int(request.frame_count, 5, 300)
    sample_interval = clamp_int(request.sample_interval, 1, 60)
    source = get_stream_source(request.source_id, request.custom_rtsp_url)

    sampled_frames, frames_read = capture_stream_sampled_frames(
        source=source,
        frame_count=frame_count,
        sample_interval=sample_interval,
        use_mock_frame=request.use_mock_frame,
    )

    tasks_to_run = (
        ["plate", "traffic_gesture", "owner_gesture"]
        if request.task_type == "all"
        else [request.task_type]
    )

    task_outputs: dict[str, dict] = {}

    for task in tasks_to_run:
        task_outputs[task] = recognize_stream_for_task(
            task_type=task,
            sampled_frames=sampled_frames,
        )

    if request.task_type == "all":
        first_best = next(
            (output["best"] for output in task_outputs.values() if output.get("best")),
            None,
        )

        if first_best is None:
            raise HTTPException(status_code=500, detail="综合连续帧识别未得到有效结果")

        final_result = {
            "model": "RTSP Frame Sampling + Multi-task Recognition",
            "stream_strategy": "连续帧抽样 + 多任务分别融合",
            "source": {
                "id": source["id"],
                "name": source["name"],
                "url": source["url"],
                "mode": "mock_frame" if request.use_mock_frame else "real_rtsp",
            },
            "frame_count_requested": frame_count,
            "frames_read": frames_read,
            "sample_interval": sample_interval,
            "sampled_frames": len(sampled_frames),
            "tasks": {
                task: {
                    "best_frame_index": output["best"].get("frame_index"),
                    "image_url": output["best"].get("image_url"),
                    "output_image_url": output["best"].get("output_image_url"),
                    "confidence": output["best"].get("confidence", 0),
                    "result": output["final_result"],
                }
                for task, output in task_outputs.items()
            },
        }

        best_image_url = first_best.get("image_url", "")
        best_output_image_url = first_best.get("output_image_url", "")
        best_saved_filename = first_best.get("saved_filename", "")
    else:
        output = task_outputs[request.task_type]
        best = output["best"]
        final_result = output["final_result"]
        final_result["source"] = {
            "id": source["id"],
            "name": source["name"],
            "url": source["url"],
            "mode": "mock_frame" if request.use_mock_frame else "real_rtsp",
        }
        final_result["frame_count_requested"] = frame_count
        final_result["frames_read"] = frames_read
        final_result["sample_interval"] = sample_interval

        best_image_url = best.get("image_url", "")
        best_output_image_url = best.get("output_image_url", "")
        best_saved_filename = best.get("saved_filename", "")

    record_id = insert_recognition_record(
        task_type=request.task_type,
        input_type="rtsp_stream" if not request.use_mock_frame else "mock_stream",
        original_filename=source["name"],
        saved_filename=best_saved_filename,
        image_url=best_image_url,
        output_image_url=best_output_image_url,
        result=final_result,
    )

    alert_id = maybe_create_stream_alert(
        record_id=record_id,
        task_type=request.task_type,
        result=final_result,
    )

    insert_operation_log(
        action="stream_frame_recognition",
        detail={
            "record_id": record_id,
            "alert_id": alert_id,
            "source_id": source["id"],
            "source_name": source["name"],
            "task_type": request.task_type,
            "frame_count": frame_count,
            "frames_read": frames_read,
            "sample_interval": sample_interval,
            "sampled_frames": len(sampled_frames),
            "use_mock_frame": request.use_mock_frame,
        },
    )

    return {
        "status": "success",
        "record_id": record_id,
        "alert_id": alert_id,
        "task_type": request.task_type,
        "input_type": "rtsp_stream" if not request.use_mock_frame else "mock_stream",
        "source": source,
        "frame_count": frame_count,
        "frames_read": frames_read,
        "sample_interval": sample_interval,
        "sampled_frames": len(sampled_frames),
        "image_url": best_image_url,
        "output_image_url": best_output_image_url,
        "result": final_result,
    }


@app.post("/api/monitor/start")
def start_global_monitor(request: MonitorStartRequest):
    supported_tasks = {"plate", "traffic_gesture", "owner_gesture", "all"}

    if request.task_type not in supported_tasks:
        raise HTTPException(
            status_code=400,
            detail="task_type 只支持 plate、traffic_gesture、owner_gesture、all",
        )

    sources = get_monitor_source_list(request.source_ids)
    interval_seconds = clamp_int(request.interval_seconds, 5, 3600)
    frame_count = clamp_int(request.frame_count, 5, 300)
    sample_interval = clamp_int(request.sample_interval, 1, 60)

    global MONITOR_THREAD

    with MONITOR_LOCK:
        if MONITOR_THREAD is not None and MONITOR_THREAD.is_alive():
            return {
                "status": "already_running",
                "message": "全局自动识别监控已经在运行",
                "monitor": get_monitor_state_snapshot(),
            }

        MONITOR_STOP_EVENT.clear()
        MONITOR_STATE.update(
            {
                "running": True,
                "task_type": request.task_type,
                "interval_seconds": interval_seconds,
                "frame_count": frame_count,
                "sample_interval": sample_interval,
                "use_mock_frame": request.use_mock_frame,
                "source_ids": [source["id"] for source in sources],
                "started_at": now_text(),
                "stopped_at": "",
                "last_round_at": "",
                "next_round_after": "启动后立即执行",
                "rounds_completed": 0,
                "total_records_created": 0,
                "total_alerts_created": 0,
                "source_status": {
                    source["id"]: {
                        "source_id": source["id"],
                        "source_name": source["name"],
                        "url": source["url"],
                        "status": "waiting",
                        "last_started_at": "",
                        "last_finished_at": "",
                        "last_error": "",
                        "last_result": None,
                        "last_record_id": None,
                        "last_alert_id": None,
                        "last_plate_count": 0,
                    }
                    for source in sources
                },
                "recent_events": [],
            }
        )

    config = {
        "sources": sources,
        "task_type": request.task_type,
        "interval_seconds": interval_seconds,
        "frame_count": frame_count,
        "sample_interval": sample_interval,
        "use_mock_frame": request.use_mock_frame,
    }

    MONITOR_THREAD = threading.Thread(
        target=monitor_worker,
        args=(config,),
        daemon=True,
        name="global_rtsp_monitor_worker",
    )
    MONITOR_THREAD.start()

    insert_operation_log(
        action="global_monitor_started",
        detail={
            "task_type": request.task_type,
            "source_ids": [source["id"] for source in sources],
            "interval_seconds": interval_seconds,
            "frame_count": frame_count,
            "sample_interval": sample_interval,
            "use_mock_frame": request.use_mock_frame,
        },
    )

    return {
        "status": "success",
        "message": "全局自动识别监控已启动",
        "monitor": get_monitor_state_snapshot(),
    }


@app.post("/api/monitor/stop")
def stop_global_monitor():
    global MONITOR_THREAD

    with MONITOR_LOCK:
        was_running = MONITOR_THREAD is not None and MONITOR_THREAD.is_alive()

    if was_running:
        MONITOR_STOP_EVENT.set()
        MONITOR_THREAD.join(timeout=3)

    with MONITOR_LOCK:
        MONITOR_STATE["running"] = False
        MONITOR_STATE["stopped_at"] = now_text()
        MONITOR_STATE["next_round_after"] = "已停止"

    insert_operation_log(
        action="global_monitor_stopped",
        detail={"was_running": was_running},
    )

    return {
        "status": "success",
        "message": "全局自动识别监控已停止" if was_running else "全局自动识别监控当前未运行",
        "monitor": get_monitor_state_snapshot(),
    }


@app.get("/api/monitor/status")
def get_global_monitor_status():
    return {
        "status": "success",
        "monitor": get_monitor_state_snapshot(),
    }


@app.post("/api/plate/image")
def recognize_plate_image(
    file: UploadFile = File(...),
    simulate_low_confidence: bool = Query(False, description="是否模拟低置信度识别结果"),
):
    allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    original_name = file.filename or ""
    suffix = Path(original_name).suffix.lower()

    if suffix not in allowed_suffixes:
        insert_operation_log(
            action="upload_image_failed",
            detail={
                "filename": original_name,
                "reason": "unsupported_file_type",
            },
        )
        raise HTTPException(
            status_code=400,
            detail="只支持 jpg、jpeg、png、bmp、webp 格式图片"
        )

    saved_filename = f"{uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / saved_filename

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_url = f"/uploads/{saved_filename}"

    output_filename = f"annotated_{saved_filename}"
    output_path = OUTPUT_DIR / output_filename
    output_image_url = f"/outputs/{output_filename}"

    confidence = 0.42 if simulate_low_confidence else 0.92

    recognition_result = create_annotated_plate_image(
        input_path=saved_path,
        output_path=output_path,
        confidence=confidence,
    )

    record_id = insert_recognition_record(
        task_type="plate",
        input_type="image",
        original_filename=original_name,
        saved_filename=saved_filename,
        image_url=image_url,
        output_image_url=output_image_url,
        result=recognition_result,
    )

    insert_operation_log(
        action="plate_image_recognition",
        detail={
            "record_id": record_id,
            "original_filename": original_name,
            "saved_filename": saved_filename,
            "simulate_low_confidence": simulate_low_confidence,
        },
    )

    alert_id = maybe_create_low_confidence_alert(
        record_id=record_id,
        result=recognition_result,
    )

    return {
        "status": "success",
        "record_id": record_id,
        "alert_id": alert_id,
        "task_type": "plate",
        "input_type": "image",
        "original_filename": original_name,
        "saved_filename": saved_filename,
        "image_url": image_url,
        "output_image_url": output_image_url,
        "result": recognition_result,
    }


@app.post("/api/gesture/owner/simulate")
def simulate_owner_gesture(request: GestureRequest):
    """
    车主手势模拟接口：
    保留按钮模拟功能，用于前端快速演示和对照测试。
    """
    result = apply_owner_gesture(request.gesture)

    record_id = insert_recognition_record(
        task_type="owner_gesture",
        input_type="simulate",
        original_filename="",
        saved_filename="",
        image_url="",
        output_image_url="",
        result=result,
    )

    insert_operation_log(
        action="owner_gesture_control",
        detail={
            "record_id": record_id,
            "gesture": result["gesture"],
            "gesture_name": result["gesture_name"],
            "action": result["action"],
            "description": result["description"],
        },
    )

    return {
        "status": "success",
        "record_id": record_id,
        "result": result,
    }


@app.post("/api/gesture/owner/image")
def recognize_owner_gesture_from_image(
    file: UploadFile = File(...),
    apply_control: bool = Query(True, description="是否将识别到的手势映射为车辆控制动作"),
):
    """
    车主手势图片 AI 识别接口：
    1. 上传手势图片
    2. 使用 MediaPipe Hands 检测 21 个手部关键点
    3. 根据关键点分类手势
    4. 生成手部骨架标注图
    5. 可选：将识别到的手势映射为车辆控制动作
    6. 写入历史记录、操作日志和低置信度告警
    """
    allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    original_name = file.filename or ""
    suffix = Path(original_name).suffix.lower()

    if suffix not in allowed_suffixes:
        insert_operation_log(
            action="owner_gesture_image_failed",
            detail={
                "filename": original_name,
                "reason": "unsupported_file_type",
            },
        )
        raise HTTPException(
            status_code=400,
            detail="只支持 jpg、jpeg、png、bmp、webp 格式图片",
        )

    saved_filename = f"owner_gesture_{uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / saved_filename

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_url = f"/uploads/{saved_filename}"

    output_filename = f"annotated_{saved_filename}"
    output_path = OUTPUT_DIR / output_filename
    output_image_url = f"/outputs/{output_filename}"

    try:
        ai_result = recognize_owner_gesture_image(
            input_path=saved_path,
            output_path=output_path,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"车主手势 AI 识别失败：{error}",
        )

    gesture = ai_result.get("gesture", "unknown")

    if apply_control and gesture != "unknown":
        control_result = apply_owner_gesture(gesture)

        ai_result["action"] = control_result["action"]
        ai_result["description"] = control_result["description"]
        ai_result["vehicle_state"] = control_result["vehicle_state"]
    else:
        ai_result["action"] = "none"
        ai_result["description"] = "未执行车辆控制动作"
        ai_result["vehicle_state"] = get_vehicle_state()

    record_id = insert_recognition_record(
        task_type="owner_gesture",
        input_type="image",
        original_filename=original_name,
        saved_filename=saved_filename,
        image_url=image_url,
        output_image_url=output_image_url,
        result=ai_result,
    )

    alert_id = None

    if ai_result.get("confidence", 0) < 0.6:
        alert_id = insert_alert_event(
            level="warning",
            event_type="owner_gesture_low_confidence",
            summary="车主手势识别置信度偏低",
            reason=f"本次手势识别置信度为 {ai_result.get('confidence', 0):.2f}，低于系统阈值 0.60。",
            suggestion="建议调整手部姿态、摄像头角度和光照条件，或使用更清晰的手势图片。",
            related_record_id=record_id,
        )

    insert_operation_log(
        action="owner_gesture_image_recognition",
        detail={
            "record_id": record_id,
            "alert_id": alert_id,
            "original_filename": original_name,
            "gesture": ai_result.get("gesture"),
            "gesture_name": ai_result.get("gesture_name"),
            "confidence": ai_result.get("confidence"),
            "apply_control": apply_control,
        },
    )

    return {
        "status": "success",
        "record_id": record_id,
        "alert_id": alert_id,
        "task_type": "owner_gesture",
        "input_type": "image",
        "original_filename": original_name,
        "saved_filename": saved_filename,
        "image_url": image_url,
        "output_image_url": output_image_url,
        "result": ai_result,
    }


@app.get("/api/vehicle/state")
def get_current_vehicle_state():
    return {
        "status": "success",
        "state": get_vehicle_state(),
    }


@app.post("/api/gesture/traffic/simulate")
def simulate_traffic_gesture(request: GestureRequest):
    """
    交警手势模拟接口：
    保留按钮模拟功能，用于前端快速演示和对照测试。
    """
    result = recognize_traffic_gesture(request.gesture)

    record_id = insert_recognition_record(
        task_type="traffic_gesture",
        input_type="simulate",
        original_filename="",
        saved_filename="",
        image_url="",
        output_image_url="",
        result=result,
    )

    insert_operation_log(
        action="traffic_gesture_recognition",
        detail={
            "record_id": record_id,
            "gesture": result["gesture"],
            "gesture_name": result["gesture_name"],
            "traffic_command": result["traffic_command"],
        },
    )

    return {
        "status": "success",
        "record_id": record_id,
        "result": result,
    }


@app.post("/api/gesture/traffic/image")
def recognize_traffic_gesture_from_image(
    file: UploadFile = File(...),
):
    """
    交警手势图片 AI 识别接口：
    1. 上传交警手势图片
    2. 使用 MediaPipe Pose 检测人体姿态关键点
    3. 根据人体关键点规则分类 8 类交警手势
    4. 生成姿态骨架标注图
    5. 写入历史记录、操作日志和低置信度告警
    """
    allowed_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    original_name = file.filename or ""
    suffix = Path(original_name).suffix.lower()

    if suffix not in allowed_suffixes:
        insert_operation_log(
            action="traffic_gesture_image_failed",
            detail={
                "filename": original_name,
                "reason": "unsupported_file_type",
            },
        )
        raise HTTPException(
            status_code=400,
            detail="只支持 jpg、jpeg、png、bmp、webp 格式图片",
        )

    saved_filename = f"traffic_gesture_{uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / saved_filename

    with saved_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_url = f"/uploads/{saved_filename}"

    output_filename = f"annotated_{saved_filename}"
    output_path = OUTPUT_DIR / output_filename
    output_image_url = f"/outputs/{output_filename}"

    try:
        ai_result = recognize_traffic_gesture_image(
            input_path=saved_path,
            output_path=output_path,
        )
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"交警手势 AI 识别失败：{error}",
        )

    record_id = insert_recognition_record(
        task_type="traffic_gesture",
        input_type="image",
        original_filename=original_name,
        saved_filename=saved_filename,
        image_url=image_url,
        output_image_url=output_image_url,
        result=ai_result,
    )

    alert_id = None

    if ai_result.get("confidence", 0) < 0.6:
        alert_id = insert_alert_event(
            level="warning",
            event_type="traffic_gesture_low_confidence",
            summary="交警手势识别置信度偏低",
            reason=f"本次交警手势识别置信度为 {ai_result.get('confidence', 0):.2f}，低于系统阈值 0.60。",
            suggestion="建议调整人体姿态、拍摄距离、摄像头角度和光照条件，或使用连续视频帧进行识别。",
            related_record_id=record_id,
        )

    insert_operation_log(
        action="traffic_gesture_image_recognition",
        detail={
            "record_id": record_id,
            "alert_id": alert_id,
            "original_filename": original_name,
            "gesture": ai_result.get("gesture"),
            "gesture_name": ai_result.get("gesture_name"),
            "traffic_command": ai_result.get("traffic_command"),
            "confidence": ai_result.get("confidence"),
        },
    )

    return {
        "status": "success",
        "record_id": record_id,
        "alert_id": alert_id,
        "task_type": "traffic_gesture",
        "input_type": "image",
        "original_filename": original_name,
        "saved_filename": saved_filename,
        "image_url": image_url,
        "output_image_url": output_image_url,
        "result": ai_result,
    }



def normalize_alert_type(event_type: str, summary: str | None = None) -> str:
    """把底层告警类型归并成前端和报告里更容易理解的风险类型。"""
    text = f"{event_type or ''} {summary or ''}"

    if "plate" in text or "车牌" in text:
        return "车牌识别异常"
    if "traffic_gesture" in text or "交警" in text or "人体" in text or "姿态" in text:
        return "交警手势识别异常"
    if "owner_gesture" in text or "车主" in text or "手部" in text:
        return "车主手势识别异常"
    if "stream" in text or "RTSP" in text or "视频" in text:
        return "视频流识别异常"
    if "low_confidence" in text or "置信度" in text:
        return "低置信度告警"

    return "其他系统告警"


def build_alert_analysis(alerts: list[dict]) -> dict:
    total_count = len(alerts)
    unresolved_alerts = [item for item in alerts if item.get("status") == "未处理"]
    unresolved_count = len(unresolved_alerts)

    warning_count = sum(1 for item in alerts if item.get("level") == "warning")
    critical_count = sum(1 for item in alerts if item.get("level") == "critical")

    type_counter = Counter(
        normalize_alert_type(item.get("event_type", ""), item.get("summary", ""))
        for item in alerts
    )

    event_counter = Counter(item.get("event_type", "unknown") for item in alerts)

    if critical_count > 0 or unresolved_count >= 8:
        risk_level = "high"
        risk_level_name = "高风险"
    elif unresolved_count >= 3 or warning_count >= 5:
        risk_level = "medium"
        risk_level_name = "中风险"
    elif unresolved_count > 0:
        risk_level = "low"
        risk_level_name = "低风险"
    else:
        risk_level = "normal"
        risk_level_name = "正常"

    main_risk_types = [name for name, _ in type_counter.most_common(3)]
    main_event_types = [
        {"event_type": name, "count": count}
        for name, count in event_counter.most_common(5)
    ]
    risk_type_stats = [
        {"risk_type": name, "count": count}
        for name, count in type_counter.most_common()
    ]

    if total_count == 0:
        analysis = "当前系统暂无告警记录，识别链路未发现异常事件。"
    elif unresolved_count == 0:
        analysis = f"当前共有 {total_count} 条告警记录，均已处理，系统处于可控状态。"
    else:
        risk_text = "、".join(main_risk_types) if main_risk_types else "未分类告警"
        analysis = (
            f"当前共有 {total_count} 条告警记录，其中 {unresolved_count} 条未处理。"
            f"主要风险集中在：{risk_text}。"
        )

    suggestions = []

    if total_count == 0:
        suggestions.append("继续执行车牌、交警手势、车主手势和 RTSP 连续帧识别测试，积累运行数据。")
    else:
        if any("车牌" in item for item in main_risk_types):
            suggestions.append("车牌识别异常较多时，优先检查沙盘车牌清晰度、摄像头角度、车辆距离和光照条件。")
            suggestions.append("若真实车牌可识别但沙盘车牌识别率低，后续应采集沙盘车牌数据并训练适配沙盘场景的检测模型。")
        if any("交警" in item for item in main_risk_types):
            suggestions.append("交警手势识别需要画面中包含清晰的人体上半身或全身姿态，建议切换到交警指挥区域摄像头或使用专门演示视频源。")
        if any("车主" in item for item in main_risk_types):
            suggestions.append("车主手势识别需要手部近景画面，建议使用车内摄像头、电脑摄像头或手势演示视频源。")
        if any("视频流" in item for item in main_risk_types):
            suggestions.append("视频流异常时，检查 RTSP 地址、沙盘推流状态、网络连通性和帧读取参数。")
        if unresolved_count > 0:
            suggestions.append("建议优先处理未处理告警，并在处理后点击“标记处理”关闭告警事件。")

    if not suggestions:
        suggestions.append("当前告警风险较低，建议继续保持日志记录和周期性检查。")

    return {
        "total_count": total_count,
        "unresolved_count": unresolved_count,
        "warning_count": warning_count,
        "critical_count": critical_count,
        "risk_level": risk_level,
        "risk_level_name": risk_level_name,
        "main_risk_types": main_risk_types,
        "risk_type_stats": risk_type_stats,
        "main_event_types": main_event_types,
        "analysis": analysis,
        "suggestions": suggestions,
        "generated_at": now_text(),
    }

@app.get("/api/records")
def list_recognition_records(limit: int = 20):
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit 必须大于 0")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            task_type,
            input_type,
            original_filename,
            saved_filename,
            image_url,
            output_image_url,
            result_json,
            created_at
        FROM recognition_records
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()
    conn.close()

    records = [row_to_record(row) for row in rows]

    return {
        "status": "success",
        "total": len(records),
        "records": records,
    }


@app.get("/api/alerts")
def list_alert_events(limit: int = 20):
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit 必须大于 0")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            level,
            event_type,
            summary,
            reason,
            suggestion,
            status,
            related_record_id,
            created_at
        FROM alert_events
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()
    conn.close()

    alerts = [row_to_alert(row) for row in rows]

    return {
        "status": "success",
        "total": len(alerts),
        "alerts": alerts,
    }


@app.get("/api/alerts/analysis")
def analyze_alert_events(limit: int = 100):
    """
    告警智能分析接口：
    读取最近告警记录，归并风险类型，计算风险等级，并生成处理建议。
    当前版本采用规则推理，后续可替换为大模型或更复杂的告警智能体。
    """
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit 必须大于 0")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            level,
            event_type,
            summary,
            reason,
            suggestion,
            status,
            related_record_id,
            created_at
        FROM alert_events
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()
    conn.close()

    alerts = [row_to_alert(row) for row in rows]
    analysis = build_alert_analysis(alerts)

    insert_operation_log(
        action="analyze_alert_events",
        detail={
            "limit": limit,
            "total_count": analysis["total_count"],
            "unresolved_count": analysis["unresolved_count"],
            "risk_level": analysis["risk_level"],
        },
    )

    return {
        "status": "success",
        "analysis": analysis,
    }


@app.post("/api/alerts/test")
def create_test_alert():
    alert_id = insert_alert_event(
        level="warning",
        event_type="manual_test",
        summary="测试告警：识别服务运行状态检查",
        reason="用户手动触发测试告警，用于验证告警中心、告警记录和前端展示功能。",
        suggestion="确认前端告警中心是否能正常刷新并显示该事件。",
        related_record_id=None,
    )

    insert_operation_log(
        action="create_test_alert",
        detail={
            "alert_id": alert_id,
        },
    )

    return {
        "status": "success",
        "alert_id": alert_id,
        "message": "测试告警已生成",
    }


@app.post("/api/alerts/{alert_id}/resolve")
def resolve_alert(alert_id: int):
    conn = get_db_connection()
    cursor = conn.cursor()

    existing = cursor.execute(
        """
        SELECT id
        FROM alert_events
        WHERE id = ?
        """,
        (alert_id,),
    ).fetchone()

    if existing is None:
        conn.close()
        raise HTTPException(status_code=404, detail="告警不存在")

    cursor.execute(
        """
        UPDATE alert_events
        SET status = '已处理'
        WHERE id = ?
        """,
        (alert_id,),
    )

    conn.commit()
    conn.close()

    insert_operation_log(
        action="resolve_alert",
        detail={
            "alert_id": alert_id,
        },
    )

    return {
        "status": "success",
        "alert_id": alert_id,
        "message": "告警已标记为已处理",
    }


@app.get("/api/logs")
def list_operation_logs(limit: int = 30):
    if limit <= 0:
        raise HTTPException(status_code=400, detail="limit 必须大于 0")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            action,
            detail,
            created_at
        FROM operation_logs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    )

    rows = cursor.fetchall()
    conn.close()

    logs = [row_to_log(row) for row in rows]

    return {
        "status": "success",
        "total": len(logs),
        "logs": logs,
    }

# OWNER_GESTURE_CONTROL_PATCH_V2
# 车主手势控车增强：车辆状态映射 + 上传视频识别接口
from datetime import datetime as _owner_datetime
from pathlib import Path as _OwnerPath
from uuid import uuid4 as _owner_uuid4
import shutil as _owner_shutil

from fastapi import UploadFile as _OwnerUploadFile
from fastapi import File as _OwnerFile
from fastapi import Query as _OwnerQuery
from fastapi import HTTPException as _OwnerHTTPException


_OWNER_FUNCTIONS = ["home", "music", "air_conditioner", "phone", "navigation"]


_OWNER_GESTURE_NAMES = {
    "open_palm": "手掌张开",
    "fist": "握拳",
    "one": "单指",
    "two": "双指",
    "thumb_up": "拇指向上",
    "thumb_down": "拇指向下",
    "ok": "OK手势",
    "swipe_left": "左滑",
    "swipe_right": "右滑",
    "wave": "挥手",
    "circle": "单指画圈",
    "unknown": "未知手势",
}


_OWNER_DEFAULT_VEHICLE_STATE = {
    "system_awake": False,
    "current_function": "home",
    "volume": 50,
    "temperature": 24,
    "phone_status": "空闲",
    "updated_at": "",
}


def _owner_now_text() -> str:
    return _owner_datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _get_owner_vehicle_state() -> dict:
    """
    尽量复用 main.py 原有车辆状态变量。
    如果旧代码里有 vehicle_state 或 VEHICLE_STATE，就直接更新它。
    如果没有，就使用增强模块自己的默认状态。
    """
    global _OWNER_DEFAULT_VEHICLE_STATE

    old_state = globals().get("vehicle_state")
    if isinstance(old_state, dict):
        return old_state

    old_state_upper = globals().get("VEHICLE_STATE")
    if isinstance(old_state_upper, dict):
        return old_state_upper

    return _OWNER_DEFAULT_VEHICLE_STATE


def _next_function(current: str, step: int) -> str:
    if current not in _OWNER_FUNCTIONS:
        current = "home"
    index = _OWNER_FUNCTIONS.index(current)
    return _OWNER_FUNCTIONS[(index + step) % len(_OWNER_FUNCTIONS)]


def apply_owner_gesture(gesture: str) -> dict:
    """
    增强版车主手势到车辆控制操作映射。
    会覆盖旧版同名函数，旧的图片接口也会自动使用这套新映射。
    """
    state = _get_owner_vehicle_state()

    state.setdefault("system_awake", False)
    state.setdefault("current_function", "home")
    state.setdefault("volume", 50)
    state.setdefault("temperature", 24)
    state.setdefault("phone_status", "空闲")

    gesture_name = _OWNER_GESTURE_NAMES.get(gesture, "未知手势")
    action = "no_action"
    description = "未触发车辆控制操作"

    if gesture == "open_palm":
        state["system_awake"] = True
        action = "wake_system"
        description = "系统已唤醒"

    elif gesture in {"fist", "ok"}:
        action = "confirm_action"
        description = f"已确认当前功能：{state.get('current_function', 'home')}"

    elif gesture == "one":
        state["volume"] = min(100, int(state.get("volume", 50)) + 5)
        action = "volume_up"
        description = f"音量增加至 {state['volume']}"

    elif gesture == "two":
        state["volume"] = max(0, int(state.get("volume", 50)) - 5)
        action = "volume_down"
        description = f"音量降低至 {state['volume']}"

    elif gesture == "circle":
        state["volume"] = min(100, int(state.get("volume", 50)) + 10)
        action = "adjust_volume"
        description = f"单指画圈调节音量，当前音量 {state['volume']}"

    elif gesture == "thumb_up":
        state["phone_status"] = "通话中"
        state["current_function"] = "phone"
        action = "answer_call"
        description = "已接听电话"

    elif gesture == "thumb_down":
        state["phone_status"] = "已挂断"
        state["current_function"] = "phone"
        action = "hang_up_call"
        description = "已挂断电话"

    elif gesture == "swipe_left":
        state["current_function"] = _next_function(str(state.get("current_function", "home")), -1)
        action = "previous_function"
        description = f"已切换到上一个功能：{state['current_function']}"

    elif gesture == "swipe_right":
        state["current_function"] = _next_function(str(state.get("current_function", "home")), 1)
        action = "next_function"
        description = f"已切换到下一个功能：{state['current_function']}"

    elif gesture == "wave":
        state["current_function"] = "home"
        action = "back_home"
        description = "已返回主页"

    state["updated_at"] = _owner_now_text()

    return {
        "gesture": gesture,
        "gesture_name": gesture_name,
        "action": action,
        "description": description,
        "vehicle_state": dict(state),
    }


@app.post("/api/gesture/owner/video")
def recognize_owner_gesture_from_video(
    file: _OwnerUploadFile = _OwnerFile(...),
    apply_control: bool = _OwnerQuery(True, description="是否将识别到的手势映射为车辆控制动作"),
    frame_sample_interval: int = _OwnerQuery(3, ge=1, le=30, description="视频抽帧间隔"),
    stable_threshold: int = _OwnerQuery(3, ge=1, le=20, description="误触发抑制阈值：至少多少帧确认后触发"),
):
    """
    车主手势视频识别接口：
    1. 上传 mp4 / avi / mov / mkv / webm
    2. 后端抽帧识别手势
    3. 静态手势做多帧投票
    4. 动态手势做轨迹判断
    5. 达到阈值后才触发车辆控制操作
    """
    from algorithm.owner_gesture_recognizer import recognize_owner_gesture_video

    allowed_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    original_name = file.filename or ""
    suffix = _OwnerPath(original_name).suffix.lower()

    if suffix not in allowed_suffixes:
        insert_operation_log(
            action="owner_gesture_video_failed",
            detail={
                "filename": original_name,
                "reason": "unsupported_file_type",
            },
        )
        raise _OwnerHTTPException(
            status_code=400,
            detail="只支持 mp4、avi、mov、mkv、webm 格式视频",
        )

    saved_filename = f"owner_gesture_video_{_owner_uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / saved_filename

    with saved_path.open("wb") as buffer:
        _owner_shutil.copyfileobj(file.file, buffer)

    image_url = f"/uploads/{saved_filename}"

    output_filename = f"annotated_owner_gesture_video_{_OwnerPath(saved_filename).stem}.jpg"
    output_path = OUTPUT_DIR / output_filename
    output_image_url = f"/outputs/{output_filename}"

    try:
        recognition_result = recognize_owner_gesture_video(
            input_path=saved_path,
            output_path=output_path,
            frame_sample_interval=frame_sample_interval,
            stable_threshold=stable_threshold,
        )
    except Exception as exc:
        insert_operation_log(
            action="owner_gesture_video_failed",
            detail={
                "filename": original_name,
                "saved_filename": saved_filename,
                "reason": str(exc),
            },
        )
        raise _OwnerHTTPException(
            status_code=500,
            detail=f"车主手势视频识别失败：{exc}",
        )

    control_result = {}
    if apply_control and recognition_result.get("triggered") and recognition_result.get("gesture") != "unknown":
        control_result = apply_owner_gesture(recognition_result["gesture"])
        recognition_result.update({
            "action": control_result.get("action"),
            "description": control_result.get("description"),
            "vehicle_state": control_result.get("vehicle_state"),
        })
    else:
        recognition_result.update({
            "action": "no_action",
            "description": recognition_result.get("trigger_reason", "未触发车辆控制操作"),
            "vehicle_state": dict(_get_owner_vehicle_state()),
        })

    record_id = insert_recognition_record(
        task_type="owner_gesture",
        input_type="video",
        original_filename=original_name,
        saved_filename=saved_filename,
        image_url=image_url,
        output_image_url=output_image_url,
        result=recognition_result,
    )

    insert_operation_log(
        action="owner_gesture_video_recognition",
        detail={
            "record_id": record_id,
            "original_filename": original_name,
            "saved_filename": saved_filename,
            "gesture": recognition_result.get("gesture"),
            "gesture_name": recognition_result.get("gesture_name"),
            "triggered": recognition_result.get("triggered"),
            "action": recognition_result.get("action"),
            "frame_sample_interval": frame_sample_interval,
            "stable_threshold": stable_threshold,
        },
    )

    return {
        "status": "success",
        "record_id": record_id,
        "alert_id": None,
        "task_type": "owner_gesture",
        "input_type": "video",
        "original_filename": original_name,
        "saved_filename": saved_filename,
        "image_url": image_url,
        "output_image_url": output_image_url,
        "result": recognition_result,
    }


# FUSION_DECISION_AGENT_PATCH_V1
# 融合决策智能体接口：跨模块综合车牌、交警手势、车主手势识别结果
from pathlib import Path as _FusionPath
from datetime import datetime as _FusionDateTime
import sqlite3 as _fusion_sqlite3
import json as _fusion_json

from fastapi import Body as _FusionBody
from fastapi import Query as _FusionQuery
from fastapi import HTTPException as _FusionHTTPException


def _fusion_db_path() -> _FusionPath:
    db_path = globals().get("DB_PATH")
    if db_path:
        return _FusionPath(db_path)
    return _FusionPath(__file__).resolve().parent / "data" / "app.db"


def _fusion_connect():
    db_path = _fusion_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _fusion_sqlite3.connect(str(db_path))
    conn.row_factory = _fusion_sqlite3.Row
    return conn


def _fusion_now_text() -> str:
    return _FusionDateTime.now().strftime("%Y-%m-%d %H:%M:%S")


def _fusion_safe_json_loads(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value

    try:
        return _fusion_json.loads(value)
    except Exception:
        return value


def _fusion_table_exists(conn, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row is not None


def _fusion_get_columns(conn, table_name: str) -> list[str]:
    if not _fusion_table_exists(conn, table_name):
        return []
    return [row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]


def _fusion_init_table():
    with _fusion_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fusion_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decision_id TEXT UNIQUE NOT NULL,
                scenario TEXT,
                risk_level TEXT,
                risk_score INTEGER,
                suggestion TEXT,
                reason TEXT,
                decision_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def _fusion_row_to_dict(row) -> dict:
    if row is None:
        return {}
    return {key: row[key] for key in row.keys()}


def _fusion_parse_record(row_dict: dict) -> dict:
    if not row_dict:
        return {}

    result_value = None
    for key in ["result", "result_json", "recognition_result"]:
        if key in row_dict:
            result_value = _fusion_safe_json_loads(row_dict.get(key))
            break

    if result_value is None:
        result_value = {}

    if isinstance(result_value, str):
        result_value = {"raw": result_value}

    created_at = (
        row_dict.get("created_at")
        or row_dict.get("created_time")
        or row_dict.get("timestamp")
        or ""
    )

    return {
        "record_id": row_dict.get("id") or row_dict.get("record_id"),
        "id": row_dict.get("id") or row_dict.get("record_id"),
        "task_type": row_dict.get("task_type", ""),
        "input_type": row_dict.get("input_type", ""),
        "original_filename": row_dict.get("original_filename", ""),
        "saved_filename": row_dict.get("saved_filename", ""),
        "image_url": row_dict.get("image_url", ""),
        "output_image_url": row_dict.get("output_image_url", ""),
        "created_at": created_at,
        "result": result_value,
    }


def _fusion_fetch_latest_record(task_type: str) -> dict:
    with _fusion_connect() as conn:
        if not _fusion_table_exists(conn, "recognition_records"):
            return {}

        columns = _fusion_get_columns(conn, "recognition_records")
        if "task_type" not in columns:
            return {}

        order_column = "id" if "id" in columns else None
        if order_column:
            sql = "SELECT * FROM recognition_records WHERE task_type=? ORDER BY id DESC LIMIT 1"
        else:
            sql = "SELECT * FROM recognition_records WHERE task_type=? LIMIT 1"

        row = conn.execute(sql, (task_type,)).fetchone()
        return _fusion_parse_record(_fusion_row_to_dict(row))


def _fusion_fetch_recent_alerts(limit: int = 5) -> list[dict]:
    with _fusion_connect() as conn:
        if not _fusion_table_exists(conn, "alert_events"):
            return []

        columns = _fusion_get_columns(conn, "alert_events")
        order_sql = "ORDER BY id DESC" if "id" in columns else ""
        rows = conn.execute(f"SELECT * FROM alert_events {order_sql} LIMIT ?", (limit,)).fetchall()

        result = []
        for row in rows:
            item = _fusion_row_to_dict(row)
            for key in ["detail", "result", "payload"]:
                if key in item:
                    item[key] = _fusion_safe_json_loads(item[key])
            result.append(item)

        return result


def _fusion_build_latest_payload() -> dict:
    return {
        "plate": _fusion_fetch_latest_record("plate"),
        "traffic_gesture": _fusion_fetch_latest_record("traffic_gesture"),
        "owner_gesture": _fusion_fetch_latest_record("owner_gesture"),
        "alerts": _fusion_fetch_recent_alerts(limit=5),
        "performance": {},
    }


def _fusion_save_decision(decision: dict) -> int:
    _fusion_init_table()

    with _fusion_connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO fusion_decisions (
                decision_id,
                scenario,
                risk_level,
                risk_score,
                suggestion,
                reason,
                decision_json,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                decision.get("decision_id"),
                decision.get("scenario"),
                decision.get("risk_level"),
                int(decision.get("risk_score") or 0),
                decision.get("suggestion"),
                decision.get("reason"),
                _fusion_json.dumps(decision, ensure_ascii=False),
                decision.get("created_at") or _fusion_now_text(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def _fusion_write_operation_log(action: str, detail: dict):
    log_func = globals().get("insert_operation_log")
    if callable(log_func):
        try:
            log_func(action=action, detail=detail)
        except Exception:
            pass


@app.get("/api/fusion/latest")
def get_fusion_latest():
    """
    获取最近一次车牌识别、交警手势识别、车主手势识别结果。
    用于融合决策前的证据汇总。
    """
    latest = _fusion_build_latest_payload()
    return {
        "status": "success",
        "latest": latest,
    }


@app.post("/api/fusion/decision")
def create_fusion_decision(
    payload: dict | None = _FusionBody(default=None),
    save: bool = _FusionQuery(True, description="是否保存融合决策记录"),
):
    """
    融合决策接口。

    用法：
    1. 不传 payload 或传空对象：自动读取数据库中最近的三类识别结果。
    2. 传 payload：使用调用方提供的车牌、交警手势、车主手势结果进行融合推理。
    """
    from algorithm.fusion_decision_agent import FusionDecisionAgent

    try:
        if not payload:
            payload = _fusion_build_latest_payload()
        elif "latest" in payload and isinstance(payload["latest"], dict):
            payload = payload["latest"]

        agent = FusionDecisionAgent()
        decision = agent.make_decision(payload)

        saved_id = None
        if save:
            saved_id = _fusion_save_decision(decision)

        _fusion_write_operation_log(
            action="fusion_decision",
            detail={
                "saved_id": saved_id,
                "decision_id": decision.get("decision_id"),
                "scenario": decision.get("scenario"),
                "risk_level": decision.get("risk_level"),
                "risk_score": decision.get("risk_score"),
            },
        )

        return {
            "status": "success",
            "saved_id": saved_id,
            "decision": decision,
        }

    except Exception as exc:
        _fusion_write_operation_log(
            action="fusion_decision_failed",
            detail={"reason": str(exc)},
        )
        raise _FusionHTTPException(status_code=500, detail=f"融合决策失败：{exc}")


@app.get("/api/fusion/history")
def get_fusion_history(
    limit: int = _FusionQuery(20, ge=1, le=100),
):
    """
    获取融合决策历史记录。
    """
    _fusion_init_table()

    with _fusion_connect() as conn:
        rows = conn.execute(
            """
            SELECT id, decision_id, scenario, risk_level, risk_score,
                   suggestion, reason, decision_json, created_at
            FROM fusion_decisions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    items = []
    for row in rows:
        item = _fusion_row_to_dict(row)
        item["decision"] = _fusion_safe_json_loads(item.get("decision_json"))
        item.pop("decision_json", None)
        items.append(item)

    return {
        "status": "success",
        "total": len(items),
        "items": items,
    }


# PERFORMANCE_MONITOR_PATCH_V1
# 端到端实时性 / 延迟测试接口
from pathlib import Path as _PerfPath
from datetime import datetime as _PerfDateTime
import sqlite3 as _perf_sqlite3
import json as _perf_json
import time as _perf_time
import statistics as _perf_statistics

from fastapi import Body as _PerfBody
from fastapi import Query as _PerfQuery
from fastapi import HTTPException as _PerfHTTPException


def _perf_db_path() -> _PerfPath:
    db_path = globals().get("DB_PATH")
    if db_path:
        return _PerfPath(db_path)
    return _PerfPath(__file__).resolve().parent / "data" / "app.db"


def _perf_connect():
    db_path = _perf_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _perf_sqlite3.connect(str(db_path))
    conn.row_factory = _perf_sqlite3.Row
    return conn


def _perf_now_text() -> str:
    return _PerfDateTime.now().strftime("%Y-%m-%d %H:%M:%S")


def _perf_init_table():
    with _perf_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS performance_latency_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                target TEXT NOT NULL,
                task_type TEXT,
                input_type TEXT,
                endpoint TEXT NOT NULL,
                method TEXT NOT NULL,
                status_code INTEGER,
                success INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                threshold_ms INTEGER NOT NULL,
                is_realtime INTEGER NOT NULL,
                request_meta TEXT,
                response_meta TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_performance_latency_target
            ON performance_latency_records(target)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_performance_latency_created_at
            ON performance_latency_records(created_at)
            """
        )
        conn.commit()


def _perf_row_to_dict(row) -> dict:
    if row is None:
        return {}
    return {key: row[key] for key in row.keys()}


def _perf_json_dumps(value) -> str:
    return _perf_json.dumps(value, ensure_ascii=False, default=str)


def _perf_json_loads(value):
    if not value:
        return {}
    try:
        return _perf_json.loads(value)
    except Exception:
        return {"raw": value}


def _perf_write_operation_log(action: str, detail: dict):
    log_func = globals().get("insert_operation_log")
    if callable(log_func):
        try:
            log_func(action=action, detail=detail)
        except Exception:
            pass


def _perf_save_record(record: dict) -> int:
    _perf_init_table()

    with _perf_connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO performance_latency_records (
                target,
                task_type,
                input_type,
                endpoint,
                method,
                status_code,
                success,
                latency_ms,
                threshold_ms,
                is_realtime,
                request_meta,
                response_meta,
                error_message,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("target"),
                record.get("task_type"),
                record.get("input_type"),
                record.get("endpoint"),
                record.get("method"),
                record.get("status_code"),
                1 if record.get("success") else 0,
                float(record.get("latency_ms") or 0),
                int(record.get("threshold_ms") or 1000),
                1 if record.get("is_realtime") else 0,
                _perf_json_dumps(record.get("request_meta") or {}),
                _perf_json_dumps(record.get("response_meta") or {}),
                record.get("error_message") or "",
                record.get("created_at") or _perf_now_text(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def _perf_project_root() -> _PerfPath:
    return _PerfPath(__file__).resolve().parent.parent


def _perf_demo_path(filename: str) -> _PerfPath:
    return _perf_project_root() / "demo" / filename


def _perf_content_type(path: _PerfPath) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix == ".mp4":
        return "video/mp4"
    if suffix == ".avi":
        return "video/x-msvideo"
    if suffix == ".mov":
        return "video/quicktime"
    return "application/octet-stream"


def _perf_build_target_specs(payload: dict) -> dict:
    """
    测试目标配置。
    默认使用 demo 目录里的测试文件：
    - demo/test.png
    - demo/hand.jpg
    - demo/traffic.png
    - demo/owner_gesture.mp4 如果存在则测试视频手势
    """
    source_id = payload.get("source_id", "live12")
    frame_count = int(payload.get("frame_count", 20))
    sample_interval = int(payload.get("sample_interval", 5))
    stream_task_type = payload.get("stream_task_type", "plate")

    owner_frame_interval = int(payload.get("owner_frame_sample_interval", 3))
    owner_stable_threshold = int(payload.get("owner_stable_threshold", 3))

    return {
        "health": {
            "target": "health",
            "task_type": "system",
            "input_type": "api",
            "method": "GET",
            "endpoint": "/api/health",
        },
        "plate_image": {
            "target": "plate_image",
            "task_type": "plate",
            "input_type": "image",
            "method": "POST",
            "endpoint": "/api/plate/image",
            "file_path": _perf_demo_path("test.png"),
            "file_field": "file",
        },
        "owner_image": {
            "target": "owner_image",
            "task_type": "owner_gesture",
            "input_type": "image",
            "method": "POST",
            "endpoint": "/api/gesture/owner/image",
            "file_path": _perf_demo_path("hand.jpg"),
            "file_field": "file",
        },
        "owner_video": {
            "target": "owner_video",
            "task_type": "owner_gesture",
            "input_type": "video",
            "method": "POST",
            "endpoint": (
                "/api/gesture/owner/video"
                f"?frame_sample_interval={owner_frame_interval}"
                f"&stable_threshold={owner_stable_threshold}"
            ),
            "file_path": _perf_demo_path("owner_gesture.mp4"),
            "file_field": "file",
        },
        "traffic_image": {
            "target": "traffic_image",
            "task_type": "traffic_gesture",
            "input_type": "image",
            "method": "POST",
            "endpoint": "/api/gesture/traffic/image",
            "file_path": _perf_demo_path("traffic.png"),
            "file_field": "file",
        },
        "stream_mock": {
            "target": "stream_mock",
            "task_type": stream_task_type,
            "input_type": "mock_stream",
            "method": "POST",
            "endpoint": "/api/stream/recognize",
            "json": {
                "source_id": source_id,
                "task_type": stream_task_type,
                "frame_count": frame_count,
                "sample_interval": sample_interval,
                "use_mock_frame": True,
            },
        },
        "stream_rtsp": {
            "target": "stream_rtsp",
            "task_type": stream_task_type,
            "input_type": "rtsp_stream",
            "method": "POST",
            "endpoint": "/api/stream/recognize",
            "json": {
                "source_id": source_id,
                "task_type": stream_task_type,
                "frame_count": frame_count,
                "sample_interval": sample_interval,
                "use_mock_frame": False,
            },
        },
        "fusion_decision": {
            "target": "fusion_decision",
            "task_type": "fusion",
            "input_type": "latest_records",
            "method": "POST",
            "endpoint": "/api/fusion/decision?save=false",
            "json": {},
        },
    }


def _perf_compact_response_meta(response) -> dict:
    meta = {
        "status_code": response.status_code,
        "response_size_bytes": len(response.content or b""),
    }

    try:
        data = response.json()
    except Exception:
        data = None

    if isinstance(data, dict):
        meta["status"] = data.get("status")
        meta["record_id"] = data.get("record_id")
        meta["saved_id"] = data.get("saved_id")

        if isinstance(data.get("result"), dict):
            result = data["result"]
            meta["gesture"] = result.get("gesture")
            meta["gesture_name"] = result.get("gesture_name")
            meta["plate_count"] = result.get("plate_count")
            meta["triggered"] = result.get("triggered")

        if isinstance(data.get("decision"), dict):
            decision = data["decision"]
            meta["scenario"] = decision.get("scenario")
            meta["risk_level"] = decision.get("risk_level")
            meta["risk_score"] = decision.get("risk_score")

    return meta


def _perf_run_one_target(client, spec: dict, threshold_ms: int) -> dict:
    target = spec["target"]
    method = spec["method"].upper()
    endpoint = spec["endpoint"]

    request_meta = {
        "target": target,
        "endpoint": endpoint,
        "method": method,
    }

    file_path = spec.get("file_path")

    if file_path is not None:
        file_path = _PerfPath(file_path)
        request_meta["file_path"] = str(file_path)

        if not file_path.exists():
            record = {
                "target": target,
                "task_type": spec.get("task_type"),
                "input_type": spec.get("input_type"),
                "endpoint": endpoint,
                "method": method,
                "status_code": 0,
                "success": False,
                "latency_ms": 0,
                "threshold_ms": threshold_ms,
                "is_realtime": False,
                "request_meta": request_meta,
                "response_meta": {"skipped": True},
                "error_message": f"测试文件不存在：{file_path}",
                "created_at": _perf_now_text(),
            }
            record["id"] = _perf_save_record(record)
            record["skipped"] = True
            return record

    start = _perf_time.perf_counter()
    status_code = 0
    response_meta = {}
    error_message = ""
    success = False

    try:
        if method == "GET":
            response = client.get(endpoint)
        elif method == "POST" and file_path is not None:
            field = spec.get("file_field", "file")
            with file_path.open("rb") as file_obj:
                files = {
                    field: (
                        file_path.name,
                        file_obj,
                        _perf_content_type(file_path),
                    )
                }
                response = client.post(endpoint, files=files)
        elif method == "POST":
            response = client.post(endpoint, json=spec.get("json") or {})
            request_meta["json"] = spec.get("json") or {}
        else:
            raise ValueError(f"暂不支持 method={method}")

        status_code = response.status_code
        response_meta = _perf_compact_response_meta(response)
        success = response.status_code < 400

        if not success:
            try:
                error_message = str(response.json())
            except Exception:
                error_message = response.text

    except Exception as exc:
        error_message = str(exc)
        success = False

    end = _perf_time.perf_counter()
    latency_ms = round((end - start) * 1000, 2)
    is_realtime = success and latency_ms <= threshold_ms

    record = {
        "target": target,
        "task_type": spec.get("task_type"),
        "input_type": spec.get("input_type"),
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "success": success,
        "latency_ms": latency_ms,
        "threshold_ms": threshold_ms,
        "is_realtime": is_realtime,
        "request_meta": request_meta,
        "response_meta": response_meta,
        "error_message": error_message,
        "created_at": _perf_now_text(),
    }

    record["id"] = _perf_save_record(record)
    return record


def _perf_summarize_records(records: list[dict], threshold_ms: int) -> dict:
    latencies = [
        float(item.get("latency_ms") or 0)
        for item in records
        if not item.get("skipped")
    ]

    if not latencies:
        return {
            "count": 0,
            "avg_latency_ms": 0,
            "min_latency_ms": 0,
            "max_latency_ms": 0,
            "p95_latency_ms": 0,
            "pass_count": 0,
            "fail_count": 0,
            "pass_rate": 0,
            "threshold_ms": threshold_ms,
            "is_realtime": False,
        }

    sorted_values = sorted(latencies)

    def percentile(values, p):
        if len(values) == 1:
            return round(values[0], 2)
        rank = (p / 100) * (len(values) - 1)
        lower = int(rank)
        upper = min(lower + 1, len(values) - 1)
        weight = rank - lower
        return round(values[lower] * (1 - weight) + values[upper] * weight, 2)

    pass_count = sum(1 for item in records if item.get("success") and item.get("latency_ms", 0) <= threshold_ms)
    fail_count = len([item for item in records if not item.get("skipped")]) - pass_count

    return {
        "count": len(latencies),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "p95_latency_ms": percentile(sorted_values, 95),
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_rate": round(pass_count / len(latencies), 4),
        "threshold_ms": threshold_ms,
        "is_realtime": fail_count == 0,
    }


@app.post("/api/performance/test")
def run_performance_test(
    payload: dict | None = _PerfBody(default=None),
):
    """
    端到端实时性测试接口。

    默认测试：
    - /api/health
    - /api/plate/image
    - /api/gesture/owner/image
    - /api/gesture/traffic/image
    - /api/stream/recognize 模拟流
    - /api/fusion/decision

    如果 demo/owner_gesture.mp4 存在，也可加入 owner_video。
    如果需要真实 RTSP，把 targets 里加入 stream_rtsp。
    """
    payload = payload or {}

    try:
        from fastapi.testclient import TestClient
    except Exception as exc:
        raise _PerfHTTPException(
            status_code=500,
            detail=f"性能测试需要 fastapi.testclient/httpx 支持，请执行 pip install httpx。原始错误：{exc}",
        )

    repeat = int(payload.get("repeat", 1))
    repeat = max(1, min(repeat, 20))

    threshold_ms = int(payload.get("threshold_ms", 1000))
    threshold_ms = max(1, threshold_ms)

    default_targets = [
        "health",
        "plate_image",
        "owner_image",
        "traffic_image",
        "stream_mock",
        "fusion_decision",
    ]

    targets = payload.get("targets") or default_targets
    if isinstance(targets, str):
        targets = [targets]

    target_specs = _perf_build_target_specs(payload)

    unknown_targets = [target for target in targets if target not in target_specs]
    if unknown_targets:
        raise _PerfHTTPException(
            status_code=400,
            detail=f"未知性能测试目标：{unknown_targets}。可用目标：{list(target_specs.keys())}",
        )

    client = TestClient(app)
    records = []

    for round_index in range(repeat):
        for target in targets:
            spec = dict(target_specs[target])
            spec["round_index"] = round_index + 1
            record = _perf_run_one_target(client, spec, threshold_ms)
            record["round_index"] = round_index + 1
            records.append(record)

    summary = _perf_summarize_records(records, threshold_ms)

    by_target = {}
    for target in targets:
        target_records = [item for item in records if item.get("target") == target]
        by_target[target] = _perf_summarize_records(target_records, threshold_ms)

    _perf_write_operation_log(
        action="performance_test",
        detail={
            "targets": targets,
            "repeat": repeat,
            "threshold_ms": threshold_ms,
            "summary": summary,
        },
    )

    return {
        "status": "success",
        "threshold_ms": threshold_ms,
        "constraint": "端到端识别延迟建议 <= 1000ms",
        "repeat": repeat,
        "targets": targets,
        "summary": summary,
        "by_target": by_target,
        "records": records,
    }


@app.get("/api/performance/latency-records")
def get_performance_latency_records(
    limit: int = _PerfQuery(50, ge=1, le=500),
    target: str | None = _PerfQuery(None),
):
    """
    获取最近延迟测试记录。
    """
    _perf_init_table()

    with _perf_connect() as conn:
        if target:
            rows = conn.execute(
                """
                SELECT *
                FROM performance_latency_records
                WHERE target=?
                ORDER BY id DESC
                LIMIT ?
                """,
                (target, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT *
                FROM performance_latency_records
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    items = []
    for row in rows:
        item = _perf_row_to_dict(row)
        item["success"] = bool(item.get("success"))
        item["is_realtime"] = bool(item.get("is_realtime"))
        item["request_meta"] = _perf_json_loads(item.get("request_meta"))
        item["response_meta"] = _perf_json_loads(item.get("response_meta"))
        items.append(item)

    return {
        "status": "success",
        "total": len(items),
        "items": items,
    }


@app.get("/api/performance/summary")
def get_performance_summary(
    limit: int = _PerfQuery(200, ge=1, le=2000),
    threshold_ms: int = _PerfQuery(1000, ge=1),
):
    """
    获取性能测试汇总：
    - 总体平均延迟
    - P95 延迟
    - 最大延迟
    - 达标率
    - 各目标接口延迟统计
    """
    _perf_init_table()

    with _perf_connect() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM performance_latency_records
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    records = []
    for row in rows:
        item = _perf_row_to_dict(row)
        item["success"] = bool(item.get("success"))
        item["is_realtime"] = bool(item.get("is_realtime"))
        records.append(item)

    overall = _perf_summarize_records(records, threshold_ms)

    by_target = {}
    targets = sorted({item.get("target") for item in records if item.get("target")})
    for target_name in targets:
        target_records = [item for item in records if item.get("target") == target_name]
        by_target[target_name] = _perf_summarize_records(target_records, threshold_ms)

    realtime_status = "pass" if overall.get("is_realtime") else "warning"

    return {
        "status": "success",
        "constraint": "端到端识别延迟建议 <= 1000ms",
        "threshold_ms": threshold_ms,
        "record_count": len(records),
        "realtime_status": realtime_status,
        "overall": overall,
        "by_target": by_target,
    }


# MULTISTREAM_CONCURRENCY_PATCH_V1
# 多路视频流并发处理接口：支持车牌、交警手势、车主手势并行消费不同视频源
from pathlib import Path as _MSPath
from datetime import datetime as _MSDateTime
import threading as _ms_threading
import time as _ms_time
import sqlite3 as _ms_sqlite3
import json as _ms_json
import uuid as _ms_uuid

from fastapi import Body as _MSBody
from fastapi import Query as _MSQuery
from fastapi import HTTPException as _MSHTTPException


_MS_LOCK = _ms_threading.RLock()
_MS_STOP_EVENT = _ms_threading.Event()
_MS_THREADS = {}
_MS_FUSION_THREAD = None

_MS_STATE = {
    "running": False,
    "started_at": None,
    "stopped_at": None,
    "enable_fusion": False,
    "fusion_interval_seconds": 5,
    "workers": {},
    "latest_results": [],
    "latest_fusion": None,
}


def _ms_now_text() -> str:
    return _MSDateTime.now().strftime("%Y-%m-%d %H:%M:%S")


def _ms_db_path() -> _MSPath:
    db_path = globals().get("DB_PATH")
    if db_path:
        return _MSPath(db_path)
    return _MSPath(__file__).resolve().parent / "data" / "app.db"


def _ms_connect():
    db_path = _ms_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = _ms_sqlite3.connect(str(db_path))
    conn.row_factory = _ms_sqlite3.Row
    return conn


def _ms_init_table():
    with _ms_connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS multistream_worker_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                worker_id TEXT NOT NULL,
                source_id TEXT,
                source_url TEXT,
                task_type TEXT NOT NULL,
                input_type TEXT,
                endpoint TEXT,
                cycle_index INTEGER,
                success INTEGER NOT NULL,
                latency_ms REAL NOT NULL,
                result_summary TEXT,
                error_message TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_multistream_worker_id
            ON multistream_worker_records(worker_id)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_multistream_created_at
            ON multistream_worker_records(created_at)
            """
        )
        conn.commit()


def _ms_json_dumps(value) -> str:
    return _ms_json.dumps(value, ensure_ascii=False, default=str)


def _ms_json_loads(value):
    if not value:
        return {}
    try:
        return _ms_json.loads(value)
    except Exception:
        return {"raw": value}


def _ms_row_to_dict(row) -> dict:
    if row is None:
        return {}
    return {key: row[key] for key in row.keys()}


def _ms_save_record(record: dict) -> int:
    _ms_init_table()

    with _ms_connect() as conn:
        cursor = conn.execute(
            """
            INSERT INTO multistream_worker_records (
                worker_id,
                source_id,
                source_url,
                task_type,
                input_type,
                endpoint,
                cycle_index,
                success,
                latency_ms,
                result_summary,
                error_message,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("worker_id"),
                record.get("source_id"),
                record.get("source_url"),
                record.get("task_type"),
                record.get("input_type"),
                record.get("endpoint"),
                int(record.get("cycle_index") or 0),
                1 if record.get("success") else 0,
                float(record.get("latency_ms") or 0),
                _ms_json_dumps(record.get("result_summary") or {}),
                record.get("error_message") or "",
                record.get("created_at") or _ms_now_text(),
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def _ms_project_root() -> _MSPath:
    return _MSPath(__file__).resolve().parent.parent


def _ms_demo_path(filename: str) -> _MSPath:
    return _ms_project_root() / "demo" / filename


def _ms_upload_dir() -> _MSPath:
    path = _MSPath(__file__).resolve().parent / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ms_content_type(path: _MSPath) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    return "application/octet-stream"


def _ms_compact_response_meta(response) -> dict:
    meta = {
        "status_code": response.status_code,
        "response_size_bytes": len(response.content or b""),
    }

    try:
        data = response.json()
    except Exception:
        data = None

    if not isinstance(data, dict):
        return meta

    meta["status"] = data.get("status")
    meta["record_id"] = data.get("record_id")
    meta["saved_id"] = data.get("saved_id")

    if isinstance(data.get("result"), dict):
        result = data["result"]
        meta["model"] = result.get("model")
        meta["gesture"] = result.get("gesture")
        meta["gesture_name"] = result.get("gesture_name")
        meta["traffic_command"] = result.get("traffic_command")
        meta["action"] = result.get("action")
        meta["description"] = result.get("description")
        meta["confidence"] = result.get("confidence")
        meta["plate_count"] = result.get("plate_count")
        meta["plates"] = result.get("plates")
        meta["frames_read"] = result.get("frames_read")
        meta["sampled_frames"] = result.get("sampled_frames")

    if isinstance(data.get("decision"), dict):
        decision = data["decision"]
        meta["scenario"] = decision.get("scenario")
        meta["risk_level"] = decision.get("risk_level")
        meta["risk_score"] = decision.get("risk_score")
        meta["suggestion"] = decision.get("suggestion")

    return meta


def _ms_update_worker(worker_id: str, **kwargs):
    with _MS_LOCK:
        worker = _MS_STATE["workers"].setdefault(worker_id, {})
        worker.update(kwargs)
        worker["updated_at"] = _ms_now_text()


def _ms_append_latest_result(record: dict):
    with _MS_LOCK:
        _MS_STATE["latest_results"].insert(0, record)
        _MS_STATE["latest_results"] = _MS_STATE["latest_results"][:50]


def _ms_public_state() -> dict:
    with _MS_LOCK:
        data = _ms_json.loads(_ms_json_dumps(_MS_STATE))

    for worker_id, thread in list(_MS_THREADS.items()):
        if worker_id in data.get("workers", {}):
            data["workers"][worker_id]["thread_alive"] = thread.is_alive()

    if _MS_FUSION_THREAD is not None:
        data["fusion_thread_alive"] = _MS_FUSION_THREAD.is_alive()
    else:
        data["fusion_thread_alive"] = False

    return data


def _ms_default_workers() -> list[dict]:
    """
    默认使用 mock/demo 模式，保证没有真实 RTSP 时也能演示三路并发。
    后续接 MediaMTX 时，将 use_mock_frame 改为 false，并填 source_url 即可。
    """
    return [
        {
            "worker_id": "plate_stream_worker",
            "source_id": "live12",
            "task_type": "plate",
            "use_mock_frame": True,
            "interval_seconds": 5,
            "frame_count": 20,
            "sample_interval": 5,
        },
        {
            "worker_id": "traffic_stream_worker",
            "source_id": "traffic_demo",
            "task_type": "traffic_gesture",
            "use_mock_frame": True,
            "demo_file": "traffic.png",
            "interval_seconds": 5,
        },
        {
            "worker_id": "owner_stream_worker",
            "source_id": "owner_demo",
            "task_type": "owner_gesture",
            "use_mock_frame": True,
            "demo_file": "hand.jpg",
            "interval_seconds": 5,
        },
    ]


def _ms_normalize_workers(payload: dict) -> list[dict]:
    workers = payload.get("workers") or payload.get("sources") or _ms_default_workers()

    if not isinstance(workers, list) or not workers:
        raise _MSHTTPException(status_code=400, detail="workers 必须是非空数组。")

    result = []
    allowed_tasks = {"plate", "traffic_gesture", "owner_gesture"}

    for index, item in enumerate(workers):
        if not isinstance(item, dict):
            raise _MSHTTPException(status_code=400, detail=f"第 {index + 1} 个 worker 配置不是对象。")

        task_type = str(item.get("task_type") or "").strip()
        if task_type not in allowed_tasks:
            raise _MSHTTPException(
                status_code=400,
                detail=f"不支持的 task_type={task_type}，只支持 {sorted(allowed_tasks)}。",
            )

        source_id = str(item.get("source_id") or f"source_{index + 1}").strip()
        worker_id = str(item.get("worker_id") or f"{task_type}_{source_id}_{index + 1}").strip()

        config = dict(item)
        config["worker_id"] = worker_id
        config["source_id"] = source_id
        config["task_type"] = task_type
        config["interval_seconds"] = max(1, int(item.get("interval_seconds", 5)))
        config["frame_count"] = max(1, int(item.get("frame_count", 20)))
        config["sample_interval"] = max(1, int(item.get("sample_interval", 5)))
        config["warmup_frames"] = max(0, int(item.get("warmup_frames", 3)))
        config["use_mock_frame"] = bool(item.get("use_mock_frame", False))
        config["fallback_demo"] = bool(item.get("fallback_demo", True))

        result.append(config)

    worker_ids = [item["worker_id"] for item in result]
    if len(worker_ids) != len(set(worker_ids)):
        raise _MSHTTPException(status_code=400, detail="worker_id 不能重复。")

    return result


def _ms_resolve_source_url(client, config: dict) -> str:
    if config.get("source_url"):
        return str(config["source_url"])

    source_id = config.get("source_id")
    if not source_id:
        return ""

    try:
        response = client.get("/api/rtsp/sources")
        data = response.json()
    except Exception:
        return ""

    sources = data.get("sources") or data.get("items") or data.get("data") or []
    if isinstance(sources, dict):
        sources = list(sources.values())

    if not isinstance(sources, list):
        return ""

    for source in sources:
        if not isinstance(source, dict):
            continue
        if str(source.get("id")) == str(source_id):
            return str(source.get("url") or "")

    return ""


def _ms_capture_frame_from_rtsp(source_url: str, worker_id: str, warmup_frames: int = 3) -> _MSPath:
    if not source_url:
        raise RuntimeError("缺少 source_url，无法从真实视频流抽帧。")

    try:
        import cv2 as _ms_cv2
    except Exception as exc:
        raise RuntimeError(f"OpenCV 不可用，无法读取 RTSP：{exc}")

    cap = _ms_cv2.VideoCapture(source_url)

    try:
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频流：{source_url}")

        frame = None
        ok = False

        for _ in range(max(0, warmup_frames)):
            cap.read()

        for _ in range(5):
            ok, frame = cap.read()
            if ok and frame is not None:
                break

        if not ok or frame is None:
            raise RuntimeError(f"视频流读取失败：{source_url}")

        output_path = _ms_upload_dir() / f"multistream_{worker_id}_{_MSDateTime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg"
        success = _ms_cv2.imwrite(str(output_path), frame)

        if not success:
            raise RuntimeError(f"抽帧图片保存失败：{output_path}")

        return output_path

    finally:
        cap.release()


def _ms_prepare_image_input(client, config: dict) -> tuple[_MSPath, dict]:
    task_type = config.get("task_type")
    use_mock = bool(config.get("use_mock_frame", False))

    if task_type == "traffic_gesture":
        demo_file = config.get("demo_file") or "traffic.png"
    elif task_type == "owner_gesture":
        demo_file = config.get("demo_file") or "hand.jpg"
    else:
        demo_file = config.get("demo_file") or "test.png"

    meta = {
        "use_mock_frame": use_mock,
        "demo_file": demo_file,
    }

    if use_mock:
        path = _ms_demo_path(demo_file)
        if not path.exists():
            raise FileNotFoundError(f"demo 测试文件不存在：{path}")
        meta["input_path"] = str(path)
        return path, meta

    source_url = _ms_resolve_source_url(client, config)
    meta["source_url"] = source_url

    try:
        frame_path = _ms_capture_frame_from_rtsp(
            source_url=source_url,
            worker_id=config.get("worker_id", "worker"),
            warmup_frames=int(config.get("warmup_frames", 3)),
        )
        meta["input_path"] = str(frame_path)
        return frame_path, meta
    except Exception:
        if not bool(config.get("fallback_demo", True)):
            raise

        fallback_path = _ms_demo_path(demo_file)
        if not fallback_path.exists():
            raise

        meta["fallback_to_demo"] = True
        meta["input_path"] = str(fallback_path)
        return fallback_path, meta


def _ms_run_plate_worker_once(client, config: dict, cycle_index: int, threshold_ms: int) -> dict:
    endpoint = "/api/stream/recognize"

    body = {
        "source_id": config.get("source_id", "live12"),
        "task_type": "plate",
        "frame_count": int(config.get("frame_count", 20)),
        "sample_interval": int(config.get("sample_interval", 5)),
        "use_mock_frame": bool(config.get("use_mock_frame", True)),
    }

    start = _ms_time.perf_counter()
    response = client.post(endpoint, json=body)
    latency_ms = round((_ms_time.perf_counter() - start) * 1000, 2)

    success = response.status_code < 400
    response_meta = _ms_compact_response_meta(response)

    error_message = ""
    if not success:
        try:
            error_message = str(response.json())
        except Exception:
            error_message = response.text

    return {
        "worker_id": config.get("worker_id"),
        "source_id": config.get("source_id"),
        "source_url": config.get("source_url", ""),
        "task_type": "plate",
        "input_type": "mock_stream" if body["use_mock_frame"] else "rtsp_stream",
        "endpoint": endpoint,
        "cycle_index": cycle_index,
        "success": success,
        "latency_ms": latency_ms,
        "threshold_ms": threshold_ms,
        "is_realtime": success and latency_ms <= threshold_ms,
        "result_summary": response_meta,
        "error_message": error_message,
        "created_at": _ms_now_text(),
    }


def _ms_run_image_stream_worker_once(client, config: dict, cycle_index: int, threshold_ms: int) -> dict:
    task_type = config.get("task_type")

    if task_type == "traffic_gesture":
        endpoint = "/api/gesture/traffic/image"
    elif task_type == "owner_gesture":
        endpoint = "/api/gesture/owner/image"
    else:
        raise RuntimeError(f"不支持的图片流 worker task_type={task_type}")

    start = _ms_time.perf_counter()
    success = False
    response_meta = {}
    error_message = ""
    status_code = 0
    input_meta = {}

    try:
        image_path, input_meta = _ms_prepare_image_input(client, config)

        with image_path.open("rb") as file_obj:
            files = {
                "file": (
                    image_path.name,
                    file_obj,
                    _ms_content_type(image_path),
                )
            }
            response = client.post(endpoint, files=files)

        status_code = response.status_code
        success = response.status_code < 400
        response_meta = _ms_compact_response_meta(response)
        response_meta["input_meta"] = input_meta

        if not success:
            try:
                error_message = str(response.json())
            except Exception:
                error_message = response.text

    except Exception as exc:
        error_message = str(exc)

    latency_ms = round((_ms_time.perf_counter() - start) * 1000, 2)

    return {
        "worker_id": config.get("worker_id"),
        "source_id": config.get("source_id"),
        "source_url": config.get("source_url", "") or input_meta.get("source_url", ""),
        "task_type": task_type,
        "input_type": "mock_frame" if bool(config.get("use_mock_frame", False)) else "rtsp_frame",
        "endpoint": endpoint,
        "cycle_index": cycle_index,
        "success": success,
        "latency_ms": latency_ms,
        "threshold_ms": threshold_ms,
        "is_realtime": success and latency_ms <= threshold_ms,
        "result_summary": response_meta,
        "error_message": error_message,
        "created_at": _ms_now_text(),
        "status_code": status_code,
    }


def _ms_run_worker_once(client, config: dict, cycle_index: int, threshold_ms: int) -> dict:
    task_type = config.get("task_type")

    if task_type == "plate":
        return _ms_run_plate_worker_once(client, config, cycle_index, threshold_ms)

    if task_type in {"traffic_gesture", "owner_gesture"}:
        return _ms_run_image_stream_worker_once(client, config, cycle_index, threshold_ms)

    raise RuntimeError(f"不支持的 task_type={task_type}")


def _ms_worker_loop(config: dict, threshold_ms: int):
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:
        _ms_update_worker(
            config.get("worker_id", "unknown"),
            status="failed",
            error_message=f"缺少 TestClient/httpx2 支持：{exc}",
        )
        return

    worker_id = config["worker_id"]
    client = TestClient(app)
    cycle_index = 0

    _ms_update_worker(
        worker_id,
        worker_id=worker_id,
        source_id=config.get("source_id"),
        source_url=config.get("source_url", ""),
        task_type=config.get("task_type"),
        status="running",
        started_at=_ms_now_text(),
        cycle_count=0,
        success_count=0,
        fail_count=0,
        last_latency_ms=None,
        is_realtime=None,
        config=config,
    )

    while not _MS_STOP_EVENT.is_set():
        cycle_index += 1

        _ms_update_worker(
            worker_id,
            status="running",
            current_cycle=cycle_index,
            last_started_at=_ms_now_text(),
        )

        try:
            record = _ms_run_worker_once(client, config, cycle_index, threshold_ms)
        except Exception as exc:
            record = {
                "worker_id": worker_id,
                "source_id": config.get("source_id"),
                "source_url": config.get("source_url", ""),
                "task_type": config.get("task_type"),
                "input_type": "",
                "endpoint": "",
                "cycle_index": cycle_index,
                "success": False,
                "latency_ms": 0,
                "threshold_ms": threshold_ms,
                "is_realtime": False,
                "result_summary": {},
                "error_message": str(exc),
                "created_at": _ms_now_text(),
            }

        try:
            record["id"] = _ms_save_record(record)
        except Exception as exc:
            record["db_error"] = str(exc)

        _ms_append_latest_result(record)

        with _MS_LOCK:
            worker = _MS_STATE["workers"].setdefault(worker_id, {})
            worker["cycle_count"] = int(worker.get("cycle_count") or 0) + 1

            if record.get("success"):
                worker["success_count"] = int(worker.get("success_count") or 0) + 1
            else:
                worker["fail_count"] = int(worker.get("fail_count") or 0) + 1

            worker["status"] = "running"
            worker["last_latency_ms"] = record.get("latency_ms")
            worker["is_realtime"] = record.get("is_realtime")
            worker["last_result_summary"] = record.get("result_summary")
            worker["last_error_message"] = record.get("error_message")
            worker["last_record_id"] = record.get("id")
            worker["updated_at"] = _ms_now_text()

        interval = max(1, int(config.get("interval_seconds", 5)))
        _MS_STOP_EVENT.wait(interval)

    _ms_update_worker(
        worker_id,
        status="stopped",
        stopped_at=_ms_now_text(),
    )


def _ms_fusion_loop(interval_seconds: int):
    try:
        from fastapi.testclient import TestClient
    except Exception as exc:
        with _MS_LOCK:
            _MS_STATE["latest_fusion"] = {
                "status": "failed",
                "error_message": f"缺少 TestClient/httpx2 支持：{exc}",
                "created_at": _ms_now_text(),
            }
        return

    client = TestClient(app)

    while not _MS_STOP_EVENT.is_set():
        start = _ms_time.perf_counter()

        try:
            response = client.post("/api/fusion/decision?save=true", json={})
            latency_ms = round((_ms_time.perf_counter() - start) * 1000, 2)
            success = response.status_code < 400
            summary = _ms_compact_response_meta(response)

            with _MS_LOCK:
                _MS_STATE["latest_fusion"] = {
                    "status": "success" if success else "failed",
                    "latency_ms": latency_ms,
                    "summary": summary,
                    "created_at": _ms_now_text(),
                }

        except Exception as exc:
            with _MS_LOCK:
                _MS_STATE["latest_fusion"] = {
                    "status": "failed",
                    "error_message": str(exc),
                    "created_at": _ms_now_text(),
                }

        _MS_STOP_EVENT.wait(max(1, int(interval_seconds)))


def _ms_stop_all(join_timeout: float = 5.0) -> dict:
    global _MS_FUSION_THREAD

    _MS_STOP_EVENT.set()

    for thread in list(_MS_THREADS.values()):
        if thread.is_alive():
            thread.join(timeout=join_timeout)

    if _MS_FUSION_THREAD is not None and _MS_FUSION_THREAD.is_alive():
        _MS_FUSION_THREAD.join(timeout=join_timeout)

    with _MS_LOCK:
        _MS_STATE["running"] = False
        _MS_STATE["stopped_at"] = _ms_now_text()

    _MS_THREADS.clear()
    _MS_FUSION_THREAD = None

    return _ms_public_state()




# MULTISTREAM_WORKER_SAFE_FIX_V2
# 修复多路并发 worker 线程异常不可见的问题，并在启动前预注册 worker 状态
def _ms_worker_loop_safe(config: dict, threshold_ms: int):
    worker_id = str(config.get("worker_id") or "unknown_worker")

    try:
        _ms_update_worker(
            worker_id,
            source_id=config.get("source_id"),
            source_url=config.get("source_url", ""),
            task_type=config.get("task_type"),
            status="starting",
            started_at=_ms_now_text(),
            cycle_count=0,
            success_count=0,
            fail_count=0,
            last_latency_ms=None,
            is_realtime=None,
            last_result_summary=None,
            last_error_message="",
            config=config,
        )

        _ms_worker_loop(config, threshold_ms)

    except BaseException as exc:
        _ms_update_worker(
            worker_id,
            source_id=config.get("source_id"),
            source_url=config.get("source_url", ""),
            task_type=config.get("task_type"),
            status="failed",
            last_error_message=repr(exc),
            stopped_at=_ms_now_text(),
            config=config,
        )

        try:
            record = {
                "worker_id": worker_id,
                "source_id": config.get("source_id"),
                "source_url": config.get("source_url", ""),
                "task_type": config.get("task_type", ""),
                "input_type": "",
                "endpoint": "",
                "cycle_index": 0,
                "success": False,
                "latency_ms": 0,
                "threshold_ms": threshold_ms,
                "is_realtime": False,
                "result_summary": {"thread_error": repr(exc)},
                "error_message": repr(exc),
                "created_at": _ms_now_text(),
            }
            record["id"] = _ms_save_record(record)
            _ms_append_latest_result(record)
        except Exception:
            pass


@app.post("/api/multistream/start")
def start_multistream_processing(payload: dict | None = _MSBody(default=None)):
    """
    启动多路视频流并发处理。

    worker.task_type 支持：
    - plate：调用 /api/stream/recognize
    - traffic_gesture：从视频源抽帧后调用 /api/gesture/traffic/image
    - owner_gesture：从视频源抽帧后调用 /api/gesture/owner/image
    """
    global _MS_FUSION_THREAD

    payload = payload or {}

    try:
        from fastapi.testclient import TestClient  # noqa: F401
    except Exception as exc:
        raise _MSHTTPException(
            status_code=500,
            detail=f"多路并发处理需要 fastapi.testclient/httpx2 支持，请确认 httpx2 已安装在后端虚拟环境。原始错误：{exc}",
        )

    with _MS_LOCK:
        if _MS_STATE.get("running"):
            raise _MSHTTPException(status_code=400, detail="多路视频流并发处理已经在运行，请先调用 /api/multistream/stop。")

    workers = _ms_normalize_workers(payload)
    threshold_ms = int(payload.get("threshold_ms", 1000))
    enable_fusion = bool(payload.get("enable_fusion", True))
    fusion_interval_seconds = max(1, int(payload.get("fusion_interval_seconds", 5)))

    _MS_STOP_EVENT.clear()
    _MS_THREADS.clear()

    with _MS_LOCK:
        _MS_STATE["running"] = True
        _MS_STATE["started_at"] = _ms_now_text()
        _MS_STATE["stopped_at"] = None
        _MS_STATE["enable_fusion"] = enable_fusion
        _MS_STATE["fusion_interval_seconds"] = fusion_interval_seconds
        _MS_STATE["workers"] = {}
        _MS_STATE["latest_results"] = []
        _MS_STATE["latest_fusion"] = None

    for config in workers:
        worker_id = config["worker_id"]

        _ms_update_worker(
            worker_id,
            source_id=config.get("source_id"),
            source_url=config.get("source_url", ""),
            task_type=config.get("task_type"),
            status="starting",
            started_at=_ms_now_text(),
            cycle_count=0,
            success_count=0,
            fail_count=0,
            last_latency_ms=None,
            is_realtime=None,
            last_result_summary=None,
            last_error_message="",
            config=config,
        )

        thread = _ms_threading.Thread(
            target=_ms_worker_loop_safe,
            args=(config, threshold_ms),
            daemon=True,
            name=f"multistream-{worker_id}",
        )

        _MS_THREADS[worker_id] = thread
        thread.start()

    if enable_fusion:
        _MS_FUSION_THREAD = _ms_threading.Thread(
            target=_ms_fusion_loop,
            args=(fusion_interval_seconds,),
            daemon=True,
            name="multistream-fusion",
        )
        _MS_FUSION_THREAD.start()
    else:
        _MS_FUSION_THREAD = None

    return {
        "status": "success",
        "message": "多路视频流并发处理已启动",
        "worker_count": len(workers),
        "enable_fusion": enable_fusion,
        "threshold_ms": threshold_ms,
        "state": _ms_public_state(),
    }


@app.post("/api/multistream/stop")
def stop_multistream_processing():
    """
    停止多路视频流并发处理。
    """
    state = _ms_stop_all()
    return {
        "status": "success",
        "message": "多路视频流并发处理已停止",
        "state": state,
    }


@app.get("/api/multistream/status")
def get_multistream_status():
    """
    获取多路视频流并发处理运行状态。
    """
    return {
        "status": "success",
        "state": _ms_public_state(),
    }


@app.get("/api/multistream/latest")
def get_multistream_latest_records(
    limit: int = _MSQuery(50, ge=1, le=500),
    worker_id: str | None = _MSQuery(None),
):
    """
    获取多路并发 worker 最近运行记录。
    """
    _ms_init_table()

    with _ms_connect() as conn:
        if worker_id:
            rows = conn.execute(
                """
                SELECT *
                FROM multistream_worker_records
                WHERE worker_id=?
                ORDER BY id DESC
                LIMIT ?
                """,
                (worker_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT *
                FROM multistream_worker_records
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

    items = []
    for row in rows:
        item = _ms_row_to_dict(row)
        item["success"] = bool(item.get("success"))
        item["result_summary"] = _ms_json_loads(item.get("result_summary"))
        items.append(item)

    return {
        "status": "success",
        "total": len(items),
        "items": items,
    }


# MULTISTREAM_UPDATE_WORKER_OVERRIDE_V1
# 兼容修复：允许 _ms_update_worker(worker_id, worker_id=worker_id, ...) 这种重复传参写法
def _ms_update_worker(*args, **kwargs):
    """
    多路并发 worker 状态更新函数。

    修复点：
    - 旧代码里部分调用同时传了位置参数 worker_id 和关键字参数 worker_id。
    - 如果函数签名写成 def _ms_update_worker(worker_id, **kwargs)，会触发：
      TypeError: got multiple values for argument 'worker_id'
    - 这里改成 *args, **kwargs，手动解析 worker_id，避免线程启动后直接失败。
    """
    positional_worker_id = None

    if args:
        positional_worker_id = args[0]

    keyword_worker_id = kwargs.pop("worker_id", None)

    worker_id = positional_worker_id or keyword_worker_id

    if worker_id is None:
        worker_id = kwargs.get("source_id") or kwargs.get("task_type") or "unknown_worker"

    worker_id = str(worker_id)

    with _MS_LOCK:
        worker = _MS_STATE["workers"].setdefault(worker_id, {})
        worker["worker_id"] = worker_id

        if keyword_worker_id is not None:
            worker["configured_worker_id"] = str(keyword_worker_id)

        worker.update(kwargs)
        worker["updated_at"] = _ms_now_text()

    return worker


# OWNER_CAMERA_REALTIME_ENDPOINT_V1
# 车主手势：电脑摄像头实时帧专用识别接口
from fastapi import UploadFile as _OCUploadFile, File as _OCFile, HTTPException as _OCHTTPException
from pathlib import Path as _OCPath
from datetime import datetime as _OCDatetime
import time as _oc_time
import uuid as _oc_uuid
import math as _oc_math
import cv2 as _oc_cv2
import numpy as _oc_np


_OC_MP = None
_OC_HANDS = None
_OC_HISTORY = []
_OC_HISTORY_MAX_SECONDS = 2.5

_OC_VEHICLE_STATE = {
    "system_awake": False,
    "current_function": "home",
    "volume": 50,
    "temperature": 24,
    "phone_status": "空闲",
    "updated_at": "",
}


def _oc_now_text() -> str:
    return _OCDatetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _oc_backend_dir() -> _OCPath:
    return _OCPath(__file__).resolve().parent


def _oc_upload_dir() -> _OCPath:
    path = _oc_backend_dir() / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _oc_output_dir() -> _OCPath:
    path = _oc_backend_dir() / "outputs"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _oc_get_hands():
    global _OC_MP, _OC_HANDS

    if _OC_HANDS is None:
        import mediapipe as _mp
        _OC_MP = _mp
        _OC_HANDS = _mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.50,
        )

    return _OC_HANDS


def _oc_dist(a, b) -> float:
    return _oc_math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _oc_to_landmark_dicts(hand_landmarks, width: int, height: int) -> list[dict]:
    items = []

    for idx, lm in enumerate(hand_landmarks.landmark):
        items.append({
            "index": idx,
            "x": round(float(lm.x), 4),
            "y": round(float(lm.y), 4),
            "z": round(float(lm.z), 4),
            "pixel_x": int(lm.x * width),
            "pixel_y": int(lm.y * height),
        })

    return items


def _oc_finger_extended(lms, tip: int, pip: int, mcp: int, wrist: int = 0) -> bool:
    """
    摄像头场景下的手指伸展判定。
    兼顾 y 方向和相对掌心距离，避免轻微弯曲造成误判。
    """
    tip_lm = lms[tip]
    pip_lm = lms[pip]
    mcp_lm = lms[mcp]
    wrist_lm = lms[wrist]

    y_extended = tip_lm.y < pip_lm.y - 0.025
    distance_extended = _oc_dist(tip_lm, wrist_lm) > _oc_dist(pip_lm, wrist_lm) * 1.05

    # 如果手指横向伸出，y 条件可能不明显，这里用 tip 到 mcp 距离兜底。
    long_enough = _oc_dist(tip_lm, mcp_lm) > 0.11

    return bool((y_extended and distance_extended) or (distance_extended and long_enough and tip_lm.y < mcp_lm.y + 0.02))


def _oc_detect_wave(hand_center: dict, static_gesture: str) -> bool:
    """
    通过最近 2.5 秒手掌中心点的水平往返运动检测挥手。
    挥手通常静态形态像 open_palm，所以必须用连续帧轨迹判断。
    """
    now = _oc_time.time()

    _OC_HISTORY.append({
        "t": now,
        "x": float(hand_center["x"]),
        "y": float(hand_center["y"]),
        "gesture": static_gesture,
    })

    while _OC_HISTORY and now - _OC_HISTORY[0]["t"] > _OC_HISTORY_MAX_SECONDS:
        _OC_HISTORY.pop(0)

    if len(_OC_HISTORY) < 4:
        return False

    xs = [item["x"] for item in _OC_HISTORY]
    amplitude = max(xs) - min(xs)

    if amplitude < 0.16:
        return False

    signs = []
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        if abs(dx) < 0.025:
            continue
        signs.append(1 if dx > 0 else -1)

    changes = 0
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            changes += 1

    open_like_count = sum(1 for item in _OC_HISTORY if item.get("gesture") in {"open_palm", "wave"})

    return changes >= 1 and open_like_count >= 2


def _oc_classify_static_hand(hand_landmarks) -> tuple[str, str, float, dict]:
    lms = hand_landmarks.landmark

    wrist = lms[0]
    thumb_tip = lms[4]
    thumb_ip = lms[3]
    thumb_mcp = lms[2]
    index_mcp = lms[5]
    index_tip = lms[8]

    index_extended = _oc_finger_extended(lms, 8, 6, 5)
    middle_extended = _oc_finger_extended(lms, 12, 10, 9)
    ring_extended = _oc_finger_extended(lms, 16, 14, 13)
    pinky_extended = _oc_finger_extended(lms, 20, 18, 17)

    extended_map = {
        "index": index_extended,
        "middle": middle_extended,
        "ring": ring_extended,
        "pinky": pinky_extended,
    }

    extended_count = sum(1 for value in extended_map.values() if value)

    thumb_index_distance = _oc_dist(thumb_tip, index_tip)
    thumb_to_wrist = _oc_dist(thumb_tip, wrist)
    palm_size = max(_oc_dist(wrist, lms[9]), 1e-6)

    # 收紧 thumb_up / thumb_down：
    # 只有拇指明显竖直向上/向下，且四指没有伸出，才判定为拇指手势。
    thumb_far_enough = thumb_to_wrist > palm_size * 0.72
    thumb_up = (
        extended_count == 0
        and thumb_far_enough
        and thumb_tip.y < wrist.y - 0.13
        and thumb_tip.y < index_mcp.y - 0.06
        and thumb_tip.y < thumb_ip.y - 0.025
    )
    thumb_down = (
        extended_count == 0
        and thumb_far_enough
        and thumb_tip.y > wrist.y + 0.13
        and thumb_tip.y > index_mcp.y + 0.06
        and thumb_tip.y > thumb_ip.y + 0.025
    )

    ok_gesture = (
        thumb_index_distance < 0.075
        and middle_extended
        and ring_extended
        and pinky_extended
    )

    features = {
        "index_extended": index_extended,
        "middle_extended": middle_extended,
        "ring_extended": ring_extended,
        "pinky_extended": pinky_extended,
        "non_thumb_extended_count": extended_count,
        "thumb_index_distance": round(thumb_index_distance, 4),
        "thumb_to_wrist": round(thumb_to_wrist, 4),
        "palm_size": round(palm_size, 4),
        "thumb_up_condition": bool(thumb_up),
        "thumb_down_condition": bool(thumb_down),
    }

    if ok_gesture:
        return "ok", "OK 手势", 0.86, features

    if thumb_up:
        return "thumb_up", "拇指向上", 0.84, features

    if thumb_down:
        return "thumb_down", "拇指向下", 0.84, features

    if extended_count >= 4:
        return "open_palm", "手掌张开", 0.88, features

    # 握拳优先于 one/two 之外的模糊状态，避免误判为 thumb_up。
    if extended_count == 0:
        return "fist", "握拳", 0.82, features

    if index_extended and not middle_extended and not ring_extended and not pinky_extended:
        return "one", "单指", 0.80, features

    if index_extended and middle_extended and not ring_extended and not pinky_extended:
        return "two", "双指", 0.80, features

    if extended_count >= 3:
        return "open_palm", "手掌张开", 0.76, features

    return "unknown", "未识别手势", 0.45, features


def _oc_apply_vehicle_action(gesture: str) -> tuple[str, str, dict]:
    state = _OC_VEHICLE_STATE
    now = _oc_now_text()

    action = "none"
    description = "未触发车辆控制"

    if gesture == "open_palm":
        state["system_awake"] = True
        state["current_function"] = "home"
        action = "wake_system"
        description = "系统已唤醒"

    elif gesture in {"fist", "ok"}:
        state["system_awake"] = True
        action = "confirm"
        description = "已确认当前操作"

    elif gesture == "one":
        state["system_awake"] = True
        state["volume"] = min(100, int(state.get("volume", 50)) + 5)
        state["current_function"] = "media"
        action = "volume_up"
        description = "音量已调高"

    elif gesture == "two":
        state["system_awake"] = True
        state["volume"] = max(0, int(state.get("volume", 50)) - 5)
        state["current_function"] = "media"
        action = "volume_down"
        description = "音量已调低"

    elif gesture == "thumb_up":
        state["system_awake"] = True
        state["current_function"] = "phone"
        state["phone_status"] = "通话中"
        action = "answer_call"
        description = "已接听电话"

    elif gesture == "thumb_down":
        state["system_awake"] = True
        state["current_function"] = "phone"
        state["phone_status"] = "已挂断"
        action = "hang_up_call"
        description = "已挂断电话"

    elif gesture == "wave":
        state["system_awake"] = True
        state["current_function"] = "home"
        action = "back_home"
        description = "已返回主页"

    state["updated_at"] = now

    return action, description, dict(state)


@app.post("/api/gesture/owner/camera-frame")
def recognize_owner_gesture_camera_frame(file: _OCUploadFile = _OCFile(...)):
    """
    电脑摄像头实时帧识别接口。

    与 /api/gesture/owner/image 的区别：
    - 专门用于前端摄像头循环抽帧
    - 保留连续帧轨迹，用于识别 wave 挥手
    - 收紧 thumb_up 规则，降低握拳误判
    """
    start_time = _oc_time.perf_counter()

    try:
        content = file.file.read()

        if not content:
            raise _OCHTTPException(status_code=400, detail="上传图片为空。")

        image_array = _oc_np.frombuffer(content, dtype=_oc_np.uint8)
        image_bgr = _oc_cv2.imdecode(image_array, _oc_cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise _OCHTTPException(status_code=400, detail="无法解析摄像头帧图片。")

        height, width = image_bgr.shape[:2]

        saved_name = f"owner_camera_{_oc_uuid.uuid4().hex}.jpg"
        output_name = f"annotated_owner_camera_{_oc_uuid.uuid4().hex}.jpg"

        saved_path = _oc_upload_dir() / saved_name
        output_path = _oc_output_dir() / output_name

        _oc_cv2.imwrite(str(saved_path), image_bgr)

        image_rgb = _oc_cv2.cvtColor(image_bgr, _oc_cv2.COLOR_BGR2RGB)
        hands = _oc_get_hands()
        result = hands.process(image_rgb)

        annotated = image_bgr.copy()

        if not result.multi_hand_landmarks:
            _OC_HISTORY.clear()
            latency_ms = round((_oc_time.perf_counter() - start_time) * 1000, 2)
            _oc_cv2.imwrite(str(output_path), annotated)

            return {
                "status": "success",
                "record_id": None,
                "input_type": "camera_frame",
                "original_filename": file.filename,
                "saved_filename": saved_name,
                "image_url": f"/uploads/{saved_name}",
                "output_image_url": f"/outputs/{output_name}",
                "latency_ms": latency_ms,
                "result": {
                    "model": "MediaPipe Hands",
                    "gesture": "no_hand",
                    "gesture_name": "未检测到手部",
                    "confidence": 0,
                    "action": "none",
                    "description": "未检测到手部",
                    "vehicle_state": dict(_OC_VEHICLE_STATE),
                    "landmarks": [],
                    "hand_center": None,
                    "camera_mode": True,
                },
            }

        hand_landmarks = result.multi_hand_landmarks[0]

        static_gesture, static_name, confidence, features = _oc_classify_static_hand(hand_landmarks)

        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]
        hand_center = {
            "x": round(sum(xs) / len(xs), 4),
            "y": round(sum(ys) / len(ys), 4),
        }

        gesture = static_gesture
        gesture_name = static_name

        if _oc_detect_wave(hand_center, static_gesture):
            gesture = "wave"
            gesture_name = "挥手"
            confidence = 0.86

        action, description, vehicle_state = _oc_apply_vehicle_action(gesture)

        if _OC_MP is not None:
            _OC_MP.solutions.drawing_utils.draw_landmarks(
                annotated,
                hand_landmarks,
                _OC_MP.solutions.hands.HAND_CONNECTIONS,
            )

        _oc_cv2.imwrite(str(output_path), annotated)

        latency_ms = round((_oc_time.perf_counter() - start_time) * 1000, 2)

        return {
            "status": "success",
            "record_id": None,
            "input_type": "camera_frame",
            "original_filename": file.filename,
            "saved_filename": saved_name,
            "image_url": f"/uploads/{saved_name}",
            "output_image_url": f"/outputs/{output_name}",
            "latency_ms": latency_ms,
            "result": {
                "model": "MediaPipe Hands",
                "gesture": gesture,
                "gesture_name": gesture_name,
                "static_gesture": static_gesture,
                "static_gesture_name": static_name,
                "confidence": confidence,
                "handedness": (
                    result.multi_handedness[0].classification[0].label
                    if result.multi_handedness else ""
                ),
                "landmarks": _oc_to_landmark_dicts(hand_landmarks, width, height),
                "finger_features": features,
                "hand_center": hand_center,
                "action": action,
                "description": description,
                "vehicle_state": vehicle_state,
                "camera_mode": True,
                "dynamic_policy": "连续帧手掌中心轨迹判断 wave",
            },
        }

    except _OCHTTPException:
        raise
    except Exception as exc:
        raise _OCHTTPException(status_code=500, detail=f"摄像头手势识别失败：{exc}")


# OWNER_CAMERA_FAST_ENDPOINT_V1
# 车主手势：电脑摄像头快速实时帧接口
@app.post("/api/gesture/owner/camera-fast-frame")
def recognize_owner_gesture_camera_fast_frame(file: _OCUploadFile = _OCFile(...)):
    """
    电脑摄像头快速识别接口。

    与 /api/gesture/owner/camera-frame 的区别：
    - 不保存上传图片
    - 不绘制/保存骨架图
    - 只返回手势结果、车辆控制状态、关键点坐标和延迟
    - 适合前端高频实时识别
    """
    start_time = _oc_time.perf_counter()

    try:
        content = file.file.read()

        if not content:
            raise _OCHTTPException(status_code=400, detail="上传图片为空。")

        image_array = _oc_np.frombuffer(content, dtype=_oc_np.uint8)
        image_bgr = _oc_cv2.imdecode(image_array, _oc_cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise _OCHTTPException(status_code=400, detail="无法解析摄像头帧图片。")

        height, width = image_bgr.shape[:2]

        image_rgb = _oc_cv2.cvtColor(image_bgr, _oc_cv2.COLOR_BGR2RGB)
        hands = _oc_get_hands()
        result = hands.process(image_rgb)

        latency_ms = round((_oc_time.perf_counter() - start_time) * 1000, 2)

        if not result.multi_hand_landmarks:
            _OC_HISTORY.clear()

            return {
                "status": "success",
                "record_id": None,
                "input_type": "camera_fast_frame",
                "output_image_url": "",
                "latency_ms": latency_ms,
                "result": {
                    "model": "MediaPipe Hands",
                    "gesture": "no_hand",
                    "gesture_name": "未检测到手部",
                    "confidence": 0,
                    "action": "none",
                    "description": "未检测到手部",
                    "vehicle_state": dict(_OC_VEHICLE_STATE),
                    "landmarks": [],
                    "hand_center": None,
                    "camera_mode": True,
                    "fast_mode": True,
                },
            }

        hand_landmarks = result.multi_hand_landmarks[0]

        static_gesture, static_name, confidence, features = _oc_classify_static_hand(hand_landmarks)

        xs = [lm.x for lm in hand_landmarks.landmark]
        ys = [lm.y for lm in hand_landmarks.landmark]

        hand_center = {
            "x": round(sum(xs) / len(xs), 4),
            "y": round(sum(ys) / len(ys), 4),
        }

        gesture = static_gesture
        gesture_name = static_name

        if _oc_detect_wave(hand_center, static_gesture):
            gesture = "wave"
            gesture_name = "挥手"
            confidence = 0.86

        action, description, vehicle_state = _oc_apply_vehicle_action(gesture)

        return {
            "status": "success",
            "record_id": None,
            "input_type": "camera_fast_frame",
            "output_image_url": "",
            "latency_ms": latency_ms,
            "result": {
                "model": "MediaPipe Hands",
                "gesture": gesture,
                "gesture_name": gesture_name,
                "static_gesture": static_gesture,
                "static_gesture_name": static_name,
                "confidence": confidence,
                "handedness": (
                    result.multi_handedness[0].classification[0].label
                    if result.multi_handedness else ""
                ),
                "landmarks": _oc_to_landmark_dicts(hand_landmarks, width, height),
                "finger_features": features,
                "hand_center": hand_center,
                "action": action,
                "description": description,
                "vehicle_state": vehicle_state,
                "camera_mode": True,
                "fast_mode": True,
                "dynamic_policy": "快速模式：连续帧手掌中心轨迹判断 wave，不返回骨架图",
            },
        }

    except _OCHTTPException:
        raise
    except Exception as exc:
        raise _OCHTTPException(status_code=500, detail=f"摄像头快速手势识别失败：{exc}")
