
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import mediapipe as mp
except ImportError:
    mp = None


GESTURE_NAMES = {
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


def get_chinese_font(size: int = 28):
    font_candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]
    for font_path in font_candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def draw_chinese_label(image_bgr: np.ndarray, text: str, x: int, y: int):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    draw = ImageDraw.Draw(pil_image)
    font = get_chinese_font(28)

    x = max(0, int(x))
    y = max(0, int(y))

    text_bbox = draw.textbbox((x, y), text, font=font)
    bg_x1, bg_y1, bg_x2, bg_y2 = text_bbox
    draw.rectangle(
        [bg_x1 - 6, bg_y1 - 6, bg_x2 + 6, bg_y2 + 6],
        fill=(0, 120, 220),
    )
    draw.text((x, y), text, font=font, fill=(255, 255, 255))

    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def landmark_to_dict(landmark: Any, index: int, image_width: int, image_height: int):
    return {
        "index": index,
        "x": round(float(landmark.x), 4),
        "y": round(float(landmark.y), 4),
        "z": round(float(landmark.z), 4),
        "pixel_x": int(landmark.x * image_width),
        "pixel_y": int(landmark.y * image_height),
    }


def distance_2d(a: dict, b: dict) -> float:
    return float(((a["x"] - b["x"]) ** 2 + (a["y"] - b["y"]) ** 2) ** 0.5)


def is_finger_extended(points: list[dict], tip_id: int, pip_id: int) -> bool:
    """
    图像坐标中 y 越小越靠上。
    对食指/中指/无名指/小指，指尖明显高于 PIP 关节，可近似认为伸直。
    """
    return points[tip_id]["y"] < points[pip_id]["y"] - 0.025


def get_hand_center(points: list[dict]) -> dict:
    ids = [0, 5, 9, 13, 17]
    return {
        "x": round(sum(points[i]["x"] for i in ids) / len(ids), 4),
        "y": round(sum(points[i]["y"] for i in ids) / len(ids), 4),
    }


def classify_static_hand_gesture(points: list[dict], handedness_score: float = 0.9) -> dict:
    """
    基于 MediaPipe 21 个手部关键点做静态手势分类。
    支持：
    - open_palm：手掌张开
    - fist：握拳
    - one：单指
    - two：双指
    - thumb_up：拇指向上
    - thumb_down：拇指向下
    - ok：OK 手势
    - unknown：未知手势

    动态手势 swipe_left / swipe_right / wave / circle 在视频连续帧里判断。
    """
    wrist = points[0]
    thumb_tip = points[4]
    index_tip = points[8]

    index_extended = is_finger_extended(points, 8, 6)
    middle_extended = is_finger_extended(points, 12, 10)
    ring_extended = is_finger_extended(points, 16, 14)
    pinky_extended = is_finger_extended(points, 20, 18)

    non_thumb_extended_count = sum([
        index_extended,
        middle_extended,
        ring_extended,
        pinky_extended,
    ])

    thumb_index_distance = distance_2d(thumb_tip, index_tip)

    thumb_up_pose = (
        thumb_tip["y"] < wrist["y"] - 0.12
        and non_thumb_extended_count <= 1
    )
    thumb_down_pose = (
        thumb_tip["y"] > wrist["y"] + 0.12
        and non_thumb_extended_count <= 1
    )

    ok_pose = (
        thumb_index_distance < 0.07
        and middle_extended
        and ring_extended
        and pinky_extended
    )

    if ok_pose:
        gesture = "ok"
        gesture_name = "OK手势"
        confidence = 0.86
    elif thumb_up_pose:
        gesture = "thumb_up"
        gesture_name = "拇指向上"
        confidence = 0.86
    elif thumb_down_pose:
        gesture = "thumb_down"
        gesture_name = "拇指向下"
        confidence = 0.86
    elif index_extended and not middle_extended and not ring_extended and not pinky_extended:
        gesture = "one"
        gesture_name = "单指"
        confidence = 0.86
    elif index_extended and middle_extended and not ring_extended and not pinky_extended:
        gesture = "two"
        gesture_name = "双指"
        confidence = 0.86
    elif non_thumb_extended_count >= 3:
        gesture = "open_palm"
        gesture_name = "手掌张开"
        confidence = 0.88
    elif non_thumb_extended_count <= 1:
        gesture = "fist"
        gesture_name = "握拳"
        confidence = 0.84
    else:
        gesture = "unknown"
        gesture_name = "未知手势"
        confidence = 0.45

    confidence = round(min(confidence, float(handedness_score)), 4)

    return {
        "gesture": gesture,
        "gesture_name": gesture_name,
        "confidence": confidence,
        "finger_features": {
            "index_extended": index_extended,
            "middle_extended": middle_extended,
            "ring_extended": ring_extended,
            "pinky_extended": pinky_extended,
            "non_thumb_extended_count": non_thumb_extended_count,
            "thumb_index_distance": round(thumb_index_distance, 4),
        },
        "hand_center": get_hand_center(points),
    }


def detect_dynamic_gesture(frame_items: list[dict]) -> dict | None:
    """
    基于连续帧手部中心点和食指指尖轨迹判断动态手势。
    """
    valid_items = [
        item for item in frame_items
        if item.get("gesture") != "unknown" and item.get("hand_center")
    ]

    if len(valid_items) < 4:
        return None

    centers = [item["hand_center"] for item in valid_items]
    index_tips = [item["landmarks"][8] for item in valid_items if len(item.get("landmarks", [])) > 8]

    xs = [p["x"] for p in centers]
    ys = [p["y"] for p in centers]

    dx = xs[-1] - xs[0]
    dy = ys[-1] - ys[0]

    # 左右滑动：整体手掌中心明显横向位移
    if abs(dx) > 0.18 and abs(dx) > abs(dy) * 1.4:
        if dx > 0:
            return {
                "gesture": "swipe_right",
                "gesture_name": "右滑",
                "confidence": round(min(0.95, 0.70 + abs(dx)), 4),
                "dynamic_feature": {
                    "dx": round(dx, 4),
                    "dy": round(dy, 4),
                    "valid_frame_count": len(valid_items),
                },
            }
        return {
            "gesture": "swipe_left",
            "gesture_name": "左滑",
            "confidence": round(min(0.95, 0.70 + abs(dx)), 4),
            "dynamic_feature": {
                "dx": round(dx, 4),
                "dy": round(dy, 4),
                "valid_frame_count": len(valid_items),
            },
        }

    # 挥手：横向方向多次变化
    diffs = np.diff(np.array(xs))
    strong_diffs = [d for d in diffs if abs(float(d)) > 0.025]
    sign_changes = 0
    for i in range(1, len(strong_diffs)):
        if strong_diffs[i - 1] * strong_diffs[i] < 0:
            sign_changes += 1

    if sign_changes >= 2:
        return {
            "gesture": "wave",
            "gesture_name": "挥手",
            "confidence": 0.84,
            "dynamic_feature": {
                "sign_changes": sign_changes,
                "valid_frame_count": len(valid_items),
            },
        }

    # 单指画圈：食指指尖轨迹同时覆盖较大 x/y 范围，且起终点较近
    if len(index_tips) >= 6:
        ix = np.array([p["x"] for p in index_tips], dtype=float)
        iy = np.array([p["y"] for p in index_tips], dtype=float)

        x_range = float(ix.max() - ix.min())
        y_range = float(iy.max() - iy.min())
        start_end = float(((ix[-1] - ix[0]) ** 2 + (iy[-1] - iy[0]) ** 2) ** 0.5)
        path_len = float(np.sum(np.sqrt(np.diff(ix) ** 2 + np.diff(iy) ** 2)))

        if x_range > 0.10 and y_range > 0.10 and path_len > 0.45 and start_end < 0.16:
            return {
                "gesture": "circle",
                "gesture_name": "单指画圈",
                "confidence": 0.82,
                "dynamic_feature": {
                    "x_range": round(x_range, 4),
                    "y_range": round(y_range, 4),
                    "path_len": round(path_len, 4),
                    "start_end": round(start_end, 4),
                    "valid_frame_count": len(valid_items),
                },
            }

    return None


def recognize_owner_gesture_image(input_path: Path, output_path: Path) -> dict:
    """
    车主手势图片识别：
    1. OpenCV 读取图片
    2. MediaPipe Hands 检测手部 21 个关键点
    3. 基于关键点规则分类静态手势
    4. 绘制手部骨架和识别标签
    5. 返回结构化识别结果
    """
    if mp is None:
        raise RuntimeError("未安装 mediapipe。请先执行：pip install mediapipe")

    image_bgr = cv2.imread(str(input_path))
    if image_bgr is None:
        raise RuntimeError("图片读取失败，请检查图片格式是否正确")

    image_height, image_width = image_bgr.shape[:2]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        results = hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            output_image = draw_chinese_label(
                image_bgr.copy(),
                "未检测到手部",
                30,
                30,
            )
            cv2.imwrite(str(output_path), output_image)
            return {
                "model": "MediaPipe Hands",
                "gesture": "unknown",
                "gesture_name": "未检测到手部",
                "confidence": 0.0,
                "handedness": "",
                "landmarks": [],
                "finger_features": {},
                "hand_center": {},
            }

        hand_landmarks = results.multi_hand_landmarks[0]

        if results.multi_handedness:
            handedness_item = results.multi_handedness[0].classification[0]
            handedness_label = handedness_item.label
            handedness_score = handedness_item.score
        else:
            handedness_label = ""
            handedness_score = 0.8

        landmarks = [
            landmark_to_dict(landmark, index, image_width, image_height)
            for index, landmark in enumerate(hand_landmarks.landmark)
        ]

        classify_result = classify_static_hand_gesture(
            points=landmarks,
            handedness_score=handedness_score,
        )

        output_image = image_bgr.copy()
        mp_drawing.draw_landmarks(
            output_image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_styles.get_default_hand_landmarks_style(),
            mp_styles.get_default_hand_connections_style(),
        )

        label = f"{classify_result['gesture_name']} {classify_result['confidence']:.2f}"
        output_image = draw_chinese_label(output_image, label, 30, 30)

        success = cv2.imwrite(str(output_path), output_image)
        if not success:
            raise RuntimeError("手势识别标注图保存失败")

        return {
            "model": "MediaPipe Hands",
            "gesture": classify_result["gesture"],
            "gesture_name": classify_result["gesture_name"],
            "confidence": classify_result["confidence"],
            "handedness": handedness_label,
            "landmarks": landmarks,
            "finger_features": classify_result["finger_features"],
            "hand_center": classify_result["hand_center"],
        }


def recognize_owner_gesture_video(
    input_path: Path,
    output_path: Path,
    frame_sample_interval: int = 3,
    stable_threshold: int = 3,
    max_sampled_frames: int = 120,
) -> dict:
    """
    车主手势视频识别：
    1. OpenCV 打开上传视频
    2. 每隔 frame_sample_interval 帧抽样
    3. MediaPipe Hands 检测手部关键点
    4. 单帧做静态手势分类
    5. 多帧轨迹判断动态手势
    6. 使用 stable_threshold 做误触发抑制
    7. 输出一张最佳关键帧标注图
    """
    if mp is None:
        raise RuntimeError("未安装 mediapipe。请先执行：pip install mediapipe")

    frame_sample_interval = max(1, int(frame_sample_interval))
    stable_threshold = max(1, int(stable_threshold))
    max_sampled_frames = max(10, int(max_sampled_frames))

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("视频读取失败，请检查视频格式是否正确")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    frame_items: list[dict] = []
    best_frame = None
    best_hand_landmarks = None
    best_label = "未检测到手部"
    best_confidence = 0.0
    frames_read = 0
    sampled_frames = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        frame_index = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            frames_read += 1

            if frame_index % frame_sample_interval != 0:
                frame_index += 1
                continue

            if sampled_frames >= max_sampled_frames:
                break

            sampled_frames += 1
            image_height, image_width = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            item = {
                "frame_index": frame_index,
                "gesture": "unknown",
                "gesture_name": "未检测到手部",
                "confidence": 0.0,
                "landmarks": [],
                "finger_features": {},
                "hand_center": {},
            }

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]

                if results.multi_handedness:
                    handedness_item = results.multi_handedness[0].classification[0]
                    handedness_score = handedness_item.score
                else:
                    handedness_score = 0.8

                landmarks = [
                    landmark_to_dict(landmark, idx, image_width, image_height)
                    for idx, landmark in enumerate(hand_landmarks.landmark)
                ]

                classify_result = classify_static_hand_gesture(
                    points=landmarks,
                    handedness_score=handedness_score,
                )

                item.update({
                    "gesture": classify_result["gesture"],
                    "gesture_name": classify_result["gesture_name"],
                    "confidence": classify_result["confidence"],
                    "landmarks": landmarks,
                    "finger_features": classify_result["finger_features"],
                    "hand_center": classify_result["hand_center"],
                })

                if classify_result["confidence"] > best_confidence:
                    best_confidence = classify_result["confidence"]
                    best_frame = frame_bgr.copy()
                    best_hand_landmarks = hand_landmarks
                    best_label = f"{classify_result['gesture_name']} {classify_result['confidence']:.2f}"

            frame_items.append(item)
            frame_index += 1

    cap.release()

    dynamic_result = detect_dynamic_gesture(frame_items)

    valid_gestures = [
        item["gesture"]
        for item in frame_items
        if item.get("gesture") not in {"unknown", ""}
    ]

    gesture_counts = Counter(valid_gestures)
    static_winner = None
    static_count = 0

    if gesture_counts:
        static_winner, static_count = gesture_counts.most_common(1)[0]

    # 动态手势优先级高于静态手势，但仍要求有效帧足够。
    if dynamic_result is not None:
        final_gesture = dynamic_result["gesture"]
        final_gesture_name = dynamic_result["gesture_name"]
        final_confidence = dynamic_result["confidence"]
        triggered = True
        trigger_reason = "检测到连续帧动态轨迹"
        dynamic_feature = dynamic_result.get("dynamic_feature", {})
    elif static_winner is not None and static_count >= stable_threshold:
        final_gesture = static_winner
        final_gesture_name = GESTURE_NAMES.get(static_winner, "未知手势")
        final_confidence = round(static_count / max(1, len(frame_items)), 4)
        triggered = True
        trigger_reason = f"同一手势连续/累计达到 {static_count} 帧，超过阈值 {stable_threshold}"
        dynamic_feature = {}
    else:
        final_gesture = "unknown"
        final_gesture_name = "未触发有效手势"
        final_confidence = 0.0
        triggered = False
        trigger_reason = f"未达到误触发抑制阈值 stable_threshold={stable_threshold}"
        dynamic_feature = {}

    if best_frame is None:
        # 如果完全没有检测到手部，仍输出一张空白提示图，避免前端无法显示。
        width = 960
        height = 540
        best_frame = np.zeros((height, width, 3), dtype=np.uint8)
        best_label = "未检测到手部"
    else:
        if best_hand_landmarks is not None:
            mp_drawing.draw_landmarks(
                best_frame,
                best_hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style(),
            )

    label = f"{final_gesture_name} {final_confidence:.2f}"
    best_frame = draw_chinese_label(best_frame, label, 30, 30)

    success = cv2.imwrite(str(output_path), best_frame)
    if not success:
        raise RuntimeError("视频手势识别关键帧标注图保存失败")

    return {
        "model": "MediaPipe Hands",
        "gesture": final_gesture,
        "gesture_name": final_gesture_name,
        "confidence": final_confidence,
        "triggered": triggered,
        "trigger_policy": "连续帧确认 + 动态轨迹判断",
        "trigger_reason": trigger_reason,
        "stable_threshold": stable_threshold,
        "static_winner": static_winner or "",
        "static_winner_count": static_count,
        "dynamic_feature": dynamic_feature,
        "video_info": {
            "total_frames": total_frames,
            "fps": round(fps, 2),
            "frames_read": frames_read,
            "sampled_frames": sampled_frames,
            "frame_sample_interval": frame_sample_interval,
        },
        "frame_results": frame_items,
    }
