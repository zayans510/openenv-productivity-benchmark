from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Dict, List, Tuple


def _normalize_text(value: Any) -> str:
    return " ".join(str(value).strip().lower().split())


def _normalize_bool(value: Any) -> str:
    normalized = _normalize_text(value)
    if normalized in {"yes", "true", "1", "reply", "needed"}:
        return "yes"
    if normalized in {"no", "false", "0", "none", "not needed"}:
        return "no"
    return normalized


def _normalize_date(value: Any) -> str:
    return _normalize_text(value).replace("/", "-")


def _normalize_time(value: Any) -> str:
    text = _normalize_text(value)
    if len(text) == 4 and ":" not in text and text.isdigit():
        return f"{text[:2]}:{text[2:]}"
    return text


def _minutes_since_midnight(value: Any) -> int:
    text = _normalize_time(value)
    parts = text.split(":")
    if len(parts) != 2:
        return -1
    if not parts[0].isdigit() or not parts[1].isdigit():
        return -1
    hour = int(parts[0])
    minute = int(parts[1])
    if hour < 0 or hour > 23 or minute < 0 or minute > 59:
        return -1
    return hour * 60 + minute


def _normalize_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return sorted({_normalize_text(item) for item in values if str(item).strip()})


def _normalize_list_in_order(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    output: List[str] = []
    for item in values:
        normalized = _normalize_text(item)
        if normalized:
            output.append(normalized)
    return output


def _normalize_decimal(value: Any) -> str:
    try:
        decimal = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, TypeError, ValueError):
        return ""
    return format(decimal, ".2f")


def _exact_match(candidate: Any, expected: Any, normalizer) -> float:
    return 1.0 if normalizer(candidate) == normalizer(expected) else 0.0


def _score_list(candidate: Any, expected: List[str]) -> float:
    actual = set(_normalize_list(candidate))
    target = set(_normalize_list(expected))
    if not target:
        return 1.0 if not actual else 0.0
    return len(actual.intersection(target)) / len(target)


def _in_any_window(day: str, start_minute: int, end_minute: int, windows: List[Dict[str, str]]) -> bool:
    for window in windows:
        if _normalize_date(window.get("day", "")) != day:
            continue
        ws = _minutes_since_midnight(window.get("start", ""))
        we = _minutes_since_midnight(window.get("end", ""))
        if ws < 0 or we < 0 or ws >= we:
            continue
        if start_minute >= ws and end_minute <= we:
            return True
    return False


@dataclass(frozen=True)
class TaskSpec:
    name: str
    difficulty: str
    instruction: str
    payload: Dict[str, Any]
    schema: Dict[str, str]
    expected: Dict[str, Any]
    max_steps: int

    def grade_submission(self, candidate: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        if not isinstance(candidate, dict):
            return 0.0, {key: 0.0 for key in self.expected.keys()}

        if self.name == "easy":
            components = {
                "label": _exact_match(candidate.get("label"), self.expected["label"], _normalize_text),
                "priority": _exact_match(candidate.get("priority"), self.expected["priority"], _normalize_text),
                "needs_reply": _exact_match(
                    candidate.get("needs_reply"), self.expected["needs_reply"], _normalize_bool
                ),
            }
            weights = {"label": 0.6, "priority": 0.2, "needs_reply": 0.2}
        elif self.name == "medium":
            day = _normalize_date(candidate.get("day", ""))
            start_minute = _minutes_since_midnight(candidate.get("start", ""))
            end_minute = _minutes_since_midnight(candidate.get("end", ""))
            required_duration = int(self.payload.get("duration_minutes", 60))
            duration_ok = 1.0 if start_minute >= 0 and end_minute - start_minute == required_duration else 0.0
            no_blocked_overlap = 1.0
            if day and start_minute >= 0 and end_minute > start_minute:
                for block in self.payload.get("blocked_windows", []):
                    if _normalize_date(block.get("day", "")) != day:
                        continue
                    bs = _minutes_since_midnight(block.get("start", ""))
                    be = _minutes_since_midnight(block.get("end", ""))
                    if bs < 0 or be < 0:
                        continue
                    overlap = max(start_minute, bs) < min(end_minute, be)
                    if overlap:
                        no_blocked_overlap = 0.0
                        break
            else:
                no_blocked_overlap = 0.0

            room_ok = 0.0
            room_name = _normalize_text(candidate.get("room", ""))
            participants = _normalize_list(candidate.get("participants", []))
            for room in self.payload.get("rooms", []):
                if _normalize_text(room.get("name", "")) != room_name:
                    continue
                capacity_ok = int(room.get("capacity", 0)) >= len(participants)
                slot_ok = _in_any_window(day, start_minute, end_minute, room.get("available", []))
                room_ok = 1.0 if capacity_ok and slot_ok else 0.0
                break

            participant_availability = 1.0
            required_people = _normalize_list(self.payload.get("required_participants", []))
            if not set(required_people).issubset(set(participants)):
                participant_availability = 0.0
            else:
                for person in required_people:
                    availability = self.payload.get("availability", {}).get(person.title(), [])
                    if not _in_any_window(day, start_minute, end_minute, availability):
                        participant_availability = 0.0
                        break

            components = {
                "day": _exact_match(candidate.get("day"), self.expected["day"], _normalize_date),
                "start": _exact_match(candidate.get("start"), self.expected["start"], _normalize_time),
                "end": _exact_match(candidate.get("end"), self.expected["end"], _normalize_time),
                "participants": _score_list(candidate.get("participants"), self.expected["participants"]),
                "room": _exact_match(candidate.get("room"), self.expected["room"], _normalize_text),
                "duration_ok": duration_ok,
                "no_blocked_overlap": no_blocked_overlap,
                "participant_availability": participant_availability,
                "room_valid": room_ok,
            }
            weights = {
                "day": 0.1,
                "start": 0.1,
                "end": 0.1,
                "participants": 0.15,
                "room": 0.1,
                "duration_ok": 0.15,
                "no_blocked_overlap": 0.1,
                "participant_availability": 0.1,
                "room_valid": 0.1,
            }
        else:
            components = {
                "valid_rows": _exact_match(candidate.get("valid_rows"), self.expected["valid_rows"], _normalize_text),
                "duplicate_ids": _score_list(candidate.get("duplicate_ids"), self.expected["duplicate_ids"]),
                "invalid_emails": _score_list(candidate.get("invalid_emails"), self.expected["invalid_emails"]),
                "normalized_total": _exact_match(
                    candidate.get("normalized_total"), self.expected["normalized_total"], _normalize_decimal
                ),
                "retained_ids": _score_list(candidate.get("retained_ids"), self.expected["retained_ids"]),
                "retained_ids_order": _exact_match(
                    candidate.get("retained_ids"),
                    self.expected["retained_ids"],
                    lambda x: json.dumps(_normalize_list_in_order(x), separators=(",", ":")),
                ),
            }
            weights = {
                "valid_rows": 0.2,
                "duplicate_ids": 0.15,
                "invalid_emails": 0.15,
                "normalized_total": 0.2,
                "retained_ids": 0.2,
                "retained_ids_order": 0.1,
            }

        score = sum(components[key] * weights[key] for key in weights)
        score = float(Decimal(str(score)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))
        return score, components

    def public_payload(self) -> Dict[str, Any]:
        return copy.deepcopy(self.payload)

    def public_schema(self) -> Dict[str, str]:
        return copy.deepcopy(self.schema)


TASK_SPECS: Dict[str, TaskSpec] = {
    "easy": TaskSpec(
        name="easy",
        difficulty="easy",
        instruction=(
            "Classify the email into a label, priority, and whether it needs a reply. "
            "Output JSON with keys: label, priority, needs_reply. "
            "Allowed labels: work, personal, spam, finance. "
            "Allowed priorities: low, normal, high. "
            "needs_reply must be yes or no."
        ),
        payload={
            "email": {
                "from": "billing@northstarbank.example",
                "subject": "Invoice 88421 available in your portal",
                "body": (
                    "Hello, your March statement is now available. "
                    "No action is needed unless you notice an error."
                ),
            },
            "edge_cases": [
                "Promotional language should not override sender intent.",
                "If no response is requested, needs_reply should be no.",
                "Classification must be based on content, not guesswork.",
            ],
        },
        schema={"label": "string", "priority": "string", "needs_reply": "string"},
        expected={"label": "finance", "priority": "normal", "needs_reply": "no"},
        max_steps=5,
    ),
    "medium": TaskSpec(
        name="medium",
        difficulty="medium",
        instruction=(
            "Schedule a 60-minute project sync that includes every required participant. "
            "Avoid blocked windows and lunch hours. "
            "Output JSON with keys: day, start, end, participants, room."
        ),
        payload={
            "duration_minutes": 60,
            "timezone": "Asia/Kolkata",
            "required_participants": ["Alex", "Priya", "Sam"],
            "blocked_windows": [
                {"day": "2026-04-09", "start": "12:00", "end": "13:00", "reason": "lunch"},
                {"day": "2026-04-09", "start": "16:00", "end": "17:00", "reason": "company all-hands"},
            ],
            "availability": {
                "Alex": [
                    {"day": "2026-04-09", "start": "09:00", "end": "11:00"},
                    {"day": "2026-04-09", "start": "14:00", "end": "16:00"},
                ],
                "Priya": [
                    {"day": "2026-04-09", "start": "10:00", "end": "11:00"},
                    {"day": "2026-04-09", "start": "14:00", "end": "15:30"},
                ],
                "Sam": [
                    {"day": "2026-04-09", "start": "09:30", "end": "10:30"},
                    {"day": "2026-04-09", "start": "14:00", "end": "17:00"},
                ],
            },
            "rooms": [
                {"name": "Focus-2", "capacity": 2, "available": [{"day": "2026-04-09", "start": "14:00", "end": "15:00"}]},
                {"name": "Focus-3", "capacity": 3, "available": [{"day": "2026-04-09", "start": "14:00", "end": "15:00"}]},
                {"name": "Board-6", "capacity": 6, "available": [{"day": "2026-04-09", "start": "15:00", "end": "16:00"}]},
            ],
            "edge_cases": [
                "A room with insufficient capacity is invalid even if time matches.",
                "The interval must fit every attendee exactly, not just overlap partially.",
                "Lunch block must be avoided even if users appear available.",
            ],
        },
        schema={
            "day": "YYYY-MM-DD",
            "start": "HH:MM",
            "end": "HH:MM",
            "participants": "list[string]",
            "room": "string",
        },
        expected={
            "day": "2026-04-09",
            "start": "14:00",
            "end": "15:00",
            "participants": ["alex", "priya", "sam"],
            "room": "focus-3",
        },
        max_steps=5,
    ),
    "hard": TaskSpec(
        name="hard",
        difficulty="hard",
        instruction=(
            "Clean the dataset deterministically using the provided rules. "
            "Keep the first occurrence of a duplicate id, drop rows with invalid emails, "
            "normalize amount to two decimals, and report summary metrics. "
            "Output JSON with keys: valid_rows, duplicate_ids, invalid_emails, normalized_total, retained_ids."
        ),
        payload={
            "rules": [
                "Trim whitespace from every string field.",
                "Emails must contain one @ and at least one dot after @.",
                "Duplicate ids are counted once per repeated id; keep the first occurrence only.",
                "Rows with invalid emails are removed before summing amounts.",
                "Sum uses the retained rows only and must be rounded to two decimals.",
            ],
            "rows": [
                {"id": "a001", "email": "alice@example.com", "amount": "120"},
                {"id": "b002", "email": "bob@example.com ", "amount": "80.5"},
                {"id": "c003", "email": "bad-email", "amount": "10.00"},
                {"id": "c003", "email": "carol@example.com", "amount": "10.00"},
                {"id": "d004", "email": " dan@example.org", "amount": "200.40"},
                {"id": "e005", "email": "eve@example.org", "amount": "160.50"},
            ],
            "edge_cases": [
                "Whitespace around email fields should be removed before validation.",
                "The second c003 row is discarded because the id is duplicate even though the email is valid.",
                "Amounts may arrive as integers or decimal strings.",
            ],
        },
        schema={
            "valid_rows": "integer",
            "duplicate_ids": "list[string]",
            "invalid_emails": "list[string]",
            "normalized_total": "string or number with two decimals",
            "retained_ids": "list[string]",
        },
        expected={
            "valid_rows": 4,
            "duplicate_ids": ["c003"],
            "invalid_emails": ["bad-email"],
            "normalized_total": "561.40",
            "retained_ids": ["a001", "b002", "d004", "e005"],
        },
        max_steps=5,
    ),
}


def get_task(task_name: str) -> TaskSpec:
    normalized = _normalize_text(task_name)
    if normalized not in TASK_SPECS:
        valid = ", ".join(sorted(TASK_SPECS.keys()))
        raise ValueError(f"unknown task '{task_name}'. expected one of: {valid}")
    return TASK_SPECS[normalized]


def task_names() -> List[str]:
    return ["easy", "medium", "hard"]


def schema_json(task_name: str) -> str:
    return json.dumps(get_task(task_name).public_schema(), sort_keys=True)


def _strict_open_interval(score: float, eps: float = 1e-3) -> float:
    """Map score into (0, 1) for external validators requiring strict bounds."""
    if score <= 0.0:
        return eps
    if score >= 1.0:
        return 1.0 - eps
    return score


def easy_grader(candidate: Dict[str, Any]) -> float:
    score, _ = get_task("easy").grade_submission(candidate)
    return _strict_open_interval(score)


def medium_grader(candidate: Dict[str, Any]) -> float:
    score, _ = get_task("medium").grade_submission(candidate)
    return _strict_open_interval(score)


def hard_grader(candidate: Dict[str, Any]) -> float:
    score, _ = get_task("hard").grade_submission(candidate)
    return _strict_open_interval(score)


# Canonical checker-friendly task list with explicit grader functions.
TASKS = [
    {
        "name": "easy",
        "difficulty": "easy",
        "grader": easy_grader,
        "spec": TASK_SPECS["easy"],
    },
    {
        "name": "medium",
        "difficulty": "medium",
        "grader": medium_grader,
        "spec": TASK_SPECS["medium"],
    },
    {
        "name": "hard",
        "difficulty": "hard",
        "grader": hard_grader,
        "spec": TASK_SPECS["hard"],
    },
]

# Compatibility alias for validators that look for this exact name.
TASKS_WITH_GRADERS = TASKS

GRADERS = {
    "easy": easy_grader,
    "medium": medium_grader,
    "hard": hard_grader,
}
