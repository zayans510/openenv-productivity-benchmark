from __future__ import annotations

import json
import unittest

from env.environment import ProductivityEnvironment
from env.tasks import get_task


class EnvironmentDeterminismTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = ProductivityEnvironment(max_steps=5)

    def test_easy_perfect_answer_scores_one(self) -> None:
        task = get_task("easy")
        self.env.reset("easy")
        action = f"final:{json.dumps(task.expected, separators=(',', ':'), sort_keys=True)}"
        _, reward, done, info = self.env.step(action)
        self.assertTrue(done)
        self.assertEqual(info["best_score"], 1.0)
        self.assertAlmostEqual(reward.value, 0.98, places=2)

    def test_medium_invalid_room_capacity_penalized(self) -> None:
        self.env.reset("medium")
        bad = {
            "day": "2026-04-09",
            "start": "14:00",
            "end": "15:00",
            "participants": ["Alex", "Priya", "Sam"],
            "room": "Focus-2",
        }
        _, reward, done, info = self.env.step(f"final:{json.dumps(bad, separators=(',', ':'))}")
        self.assertTrue(done)
        self.assertLess(info["best_score"], 1.0)
        self.assertLess(reward.value, 0.98)

    def test_hard_deterministic(self) -> None:
        self.env.reset("hard")
        answer = {
            "valid_rows": 4,
            "duplicate_ids": ["c003"],
            "invalid_emails": ["bad-email"],
            "normalized_total": "561.40",
            "retained_ids": ["a001", "b002", "d004", "e005"],
        }
        _, _, _, info_a = self.env.step(f"final:{json.dumps(answer, separators=(',', ':'))}")
        self.env.reset("hard")
        _, _, _, info_b = self.env.step(f"final:{json.dumps(answer, separators=(',', ':'))}")
        self.assertEqual(info_a["best_score"], info_b["best_score"])

    def test_loop_penalty(self) -> None:
        self.env.reset("easy")
        self.env.step("inspect")
        _, reward, _, _ = self.env.step("inspect")
        self.assertLess(reward.value, -0.02)


if __name__ == "__main__":
    unittest.main()
