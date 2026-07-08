
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any
import uuid


@dataclass
class FusionRuleResult:
    scenario: str
    risk_level: str
    risk_score: int
    suggestion: str
    reason: str
    control_advice: str


class FusionDecisionAgent:
    """
    跨模块融合决策智能体。

    设计目标：
    1. 综合车牌识别、交警手势识别、车主手势识别结果。
    2. 生成结构化驾驶建议。
    3. 当前采用规则推理，后续可将 evidence + rule_result 交给 LLM 生成更自然的解释。
    """

    STOP_GESTURES = {
        "stop",
        "stop_signal",
        "stop_moving",
        "parking",
        "halt",
        "forbid",
        "unknown_stop",
    }

    LANE_CHANGE_GESTURES = {
        "lane_change",
        "change_lane",
        "turn_left",
        "turn_right",
        "left_turn",
        "right_turn",
    }

    GO_GESTURES = {
        "go_straight",
        "straight",
        "move_forward",
        "go",
        "pass",
    }

    OWNER_PHONE_GESTURES = {
        "thumb_up",
        "thumb_down",
    }

    OWNER_INTERACTION_GESTURES = {
        "open_palm",
        "fist",
        "ok",
        "one",
        "two",
        "circle",
        "swipe_left",
        "swipe_right",
        "wave",
    }

    def make_decision(self, payload: dict[str, Any] | None) -> dict[str, Any]:
        payload = payload or {}

        evidence = self._normalize_payload(payload)
        rule_result = self._run_rules(evidence)

        decision_id = f"fusion_{uuid.uuid4().hex}"
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        llm_prompt = self._build_llm_prompt(evidence, rule_result)

        return {
            "decision_id": decision_id,
            "created_at": now,
            "agent": {
                "name": "FusionDecisionAgent",
                "version": "1.0",
                "mode": "rule_based_with_llm_extension",
                "llm_enabled": False,
                "description": "当前使用规则推理生成综合驾驶建议，预留 LLM 解释生成接口。",
            },
            "scenario": rule_result.scenario,
            "risk_level": rule_result.risk_level,
            "risk_score": rule_result.risk_score,
            "suggestion": rule_result.suggestion,
            "reason": rule_result.reason,
            "control_advice": rule_result.control_advice,
            "evidence": evidence,
            "llm_prompt_preview": llm_prompt,
        }

    def _normalize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        plate = payload.get("plate") or payload.get("plate_result") or {}
        traffic = payload.get("traffic_gesture") or payload.get("traffic") or {}
        owner = payload.get("owner_gesture") or payload.get("owner") or {}
        performance = payload.get("performance") or {}
        alerts = payload.get("alerts") or []

        plate_result = plate.get("result") if isinstance(plate.get("result"), dict) else plate
        traffic_result = traffic.get("result") if isinstance(traffic.get("result"), dict) else traffic
        owner_result = owner.get("result") if isinstance(owner.get("result"), dict) else owner

        plates = self._extract_plates(plate_result)
        plate_count = self._extract_plate_count(plate_result, plates)

        traffic_gesture = self._safe_str(traffic_result.get("gesture"))
        traffic_gesture_name = self._safe_str(traffic_result.get("gesture_name"))

        owner_gesture = self._safe_str(owner_result.get("gesture"))
        owner_gesture_name = self._safe_str(owner_result.get("gesture_name"))
        owner_action = self._safe_str(owner_result.get("action"))
        vehicle_state = owner_result.get("vehicle_state") if isinstance(owner_result.get("vehicle_state"), dict) else {}

        latency_ms = self._extract_latency_ms(performance, plate, traffic, owner)

        return {
            "plate": {
                "available": bool(plate),
                "record_id": plate.get("record_id") or plate.get("id"),
                "input_type": plate.get("input_type", ""),
                "plate_count": plate_count,
                "plates": plates,
                "raw": plate_result,
            },
            "traffic_gesture": {
                "available": bool(traffic),
                "record_id": traffic.get("record_id") or traffic.get("id"),
                "input_type": traffic.get("input_type", ""),
                "gesture": traffic_gesture,
                "gesture_name": traffic_gesture_name,
                "traffic_command": traffic_result.get("traffic_command", ""),
                "confidence": traffic_result.get("confidence", 0),
                "raw": traffic_result,
            },
            "owner_gesture": {
                "available": bool(owner),
                "record_id": owner.get("record_id") or owner.get("id"),
                "input_type": owner.get("input_type", ""),
                "gesture": owner_gesture,
                "gesture_name": owner_gesture_name,
                "action": owner_action,
                "description": owner_result.get("description", ""),
                "vehicle_state": vehicle_state,
                "confidence": owner_result.get("confidence", 0),
                "raw": owner_result,
            },
            "performance": {
                "latency_ms": latency_ms,
                "is_realtime": latency_ms is None or latency_ms <= 1000,
                "raw": performance,
            },
            "alerts": alerts if isinstance(alerts, list) else [],
        }

    def _run_rules(self, evidence: dict[str, Any]) -> FusionRuleResult:
        plate_info = evidence["plate"]
        traffic_info = evidence["traffic_gesture"]
        owner_info = evidence["owner_gesture"]
        performance = evidence["performance"]

        plate_count = int(plate_info.get("plate_count") or 0)
        has_plate = plate_count > 0

        traffic_gesture = str(traffic_info.get("gesture") or "").lower()
        traffic_name = str(traffic_info.get("gesture_name") or "")

        owner_gesture = str(owner_info.get("gesture") or "").lower()
        owner_action = str(owner_info.get("action") or "")

        latency_ms = performance.get("latency_ms")

        # 规则 1：延迟超过 1 秒，优先提示系统实时性风险。
        if latency_ms is not None and latency_ms > 1000:
            return FusionRuleResult(
                scenario="端到端识别延迟超限",
                risk_level="high",
                risk_score=90,
                suggestion="当前全链路延迟超过 1 秒，建议降低抽帧频率或减少并发任务数。",
                reason=f"系统检测到 latency_ms={latency_ms}，超过实时性约束建议值 1000ms。",
                control_advice="降低视频帧处理频率，优先保障关键感知任务。",
            )

        # 规则 2：交警停止手势 + 前方车辆。
        if self._match_stop(traffic_gesture, traffic_name) and has_plate:
            return FusionRuleResult(
                scenario="交警停止指令与前方车辆同时存在",
                risk_level="high",
                risk_score=88,
                suggestion="建议立即减速并停车等待交警指挥。",
                reason="系统检测到交警停止类手势，同时车牌模块检测到前方车辆目标，继续通行存在冲突风险。",
                control_advice="减速、保持安全车距、停车等待指令。",
            )

        # 规则 3：交警变道手势 + 前方车辆。
        if self._match_lane_change(traffic_gesture, traffic_name) and has_plate:
            return FusionRuleResult(
                scenario="交警变道指令与车辆目标同时存在",
                risk_level="medium",
                risk_score=68,
                suggestion="建议按交警指令谨慎变道，并保持与前车的安全距离。",
                reason="交警手势模块检测到变道类指令，车牌模块检测到车辆目标，需避免变道过程中的碰撞风险。",
                control_advice="开启转向提示、观察侧后方、低速变道。",
            )

        # 规则 4：交警通行手势，无明显车辆冲突。
        if self._match_go(traffic_gesture, traffic_name) and not has_plate:
            return FusionRuleResult(
                scenario="交警通行指令且未检测到前方车辆冲突",
                risk_level="low",
                risk_score=25,
                suggestion="建议按交警指令低速通行，继续保持环境观察。",
                reason="交警手势模块检测到通行类指令，当前未检测到明确前方车辆目标。",
                control_advice="低速通行，持续监控车牌与交警手势变化。",
            )

        # 规则 5：车主电话类手势。
        if owner_gesture in self.OWNER_PHONE_GESTURES or owner_action in {"answer_call", "hang_up_call"}:
            return FusionRuleResult(
                scenario="车主电话交互手势触发",
                risk_level="medium",
                risk_score=55,
                suggestion="已执行电话相关车载交互，建议驾驶员保持注意力集中。",
                reason="车主手势模块检测到接听或挂断电话类手势，属于驾驶过程中的交互操作。",
                control_advice="完成电话操作后恢复道路观察，避免长时间分心。",
            )

        # 规则 6：车主普通控制手势。
        if owner_gesture in self.OWNER_INTERACTION_GESTURES:
            return FusionRuleResult(
                scenario="车主车载功能控制",
                risk_level="low",
                risk_score=30,
                suggestion="已根据车主手势执行车载功能控制，建议继续保持安全驾驶。",
                reason="系统检测到车主手势输入，并完成对应的模拟车辆功能控制。",
                control_advice="维持当前驾驶状态，必要时继续监控交警手势与前方车辆。",
            )

        # 规则 7：视频流正常但车牌长期为空，可提示模型适配问题。
        if plate_info.get("available") and plate_count == 0 and str(plate_info.get("input_type")) in {"rtsp_stream", "mock_stream", "stream", "video"}:
            return FusionRuleResult(
                scenario="视频流车牌识别为空",
                risk_level="medium",
                risk_score=50,
                suggestion="视频流读取正常但未识别到车牌，建议检查摄像头角度或后续训练沙盘车牌检测模型。",
                reason="车牌模块存在视频输入结果，但 plate_count=0，可能与沙盘车牌尺寸、清晰度或模型域适配有关。",
                control_advice="保留视频流监控，后续使用动态沙盘数据训练车牌检测模型。",
            )

        # 默认规则。
        return FusionRuleResult(
            scenario="常规多模态感知状态",
            risk_level="low",
            risk_score=20,
            suggestion="当前未发现明显冲突风险，建议继续监控车牌、交警手势与车主手势输入。",
            reason="融合智能体未检测到高风险组合条件。",
            control_advice="保持系统监控运行，等待新的感知结果。",
        )

    def _build_llm_prompt(self, evidence: dict[str, Any], rule_result: FusionRuleResult) -> str:
        return (
            "请根据以下多模态交通感知证据生成简洁驾驶建议：\n"
            f"车牌信息：plate_count={evidence['plate']['plate_count']}, plates={evidence['plate']['plates']}\n"
            f"交警手势：{evidence['traffic_gesture']['gesture_name']} ({evidence['traffic_gesture']['gesture']})\n"
            f"车主手势：{evidence['owner_gesture']['gesture_name']} ({evidence['owner_gesture']['gesture']})\n"
            f"规则推理场景：{rule_result.scenario}\n"
            f"风险等级：{rule_result.risk_level}\n"
            f"建议：{rule_result.suggestion}"
        )

    def _extract_plates(self, plate_result: dict[str, Any]) -> list[dict[str, Any]]:
        if not isinstance(plate_result, dict):
            return []

        plates = plate_result.get("plates")
        if isinstance(plates, list):
            return plates

        # 兼容单车牌字段。
        plate_number = plate_result.get("plate_number")
        if plate_number:
            return [{
                "plate_number": plate_number,
                "confidence": plate_result.get("confidence"),
                "plate_color": plate_result.get("plate_color", ""),
            }]

        return []

    def _extract_plate_count(self, plate_result: dict[str, Any], plates: list[dict[str, Any]]) -> int:
        if not isinstance(plate_result, dict):
            return 0

        raw_count = plate_result.get("plate_count")
        if isinstance(raw_count, int):
            return raw_count

        try:
            if raw_count is not None:
                return int(raw_count)
        except Exception:
            pass

        return len(plates)

    def _extract_latency_ms(
        self,
        performance: dict[str, Any],
        plate: dict[str, Any],
        traffic: dict[str, Any],
        owner: dict[str, Any],
    ) -> int | None:
        candidates = [
            performance.get("latency_ms") if isinstance(performance, dict) else None,
            performance.get("total_ms") if isinstance(performance, dict) else None,
            plate.get("latency_ms") if isinstance(plate, dict) else None,
            traffic.get("latency_ms") if isinstance(traffic, dict) else None,
            owner.get("latency_ms") if isinstance(owner, dict) else None,
        ]

        for value in candidates:
            if value is None:
                continue
            try:
                return int(float(value))
            except Exception:
                continue

        return None

    def _match_stop(self, gesture: str, gesture_name: str) -> bool:
        text = f"{gesture} {gesture_name}".lower()
        return gesture in self.STOP_GESTURES or "停止" in text or "停车" in text or "stop" in text

    def _match_lane_change(self, gesture: str, gesture_name: str) -> bool:
        text = f"{gesture} {gesture_name}".lower()
        return (
            gesture in self.LANE_CHANGE_GESTURES
            or "变道" in text
            or "转弯" in text
            or "lane" in text
            or "turn" in text
        )

    def _match_go(self, gesture: str, gesture_name: str) -> bool:
        text = f"{gesture} {gesture_name}".lower()
        return (
            gesture in self.GO_GESTURES
            or "直行" in text
            or "通行" in text
            or "go" in text
            or "straight" in text
        )

    def _safe_str(self, value: Any) -> str:
        if value is None:
            return ""
        return str(value)
