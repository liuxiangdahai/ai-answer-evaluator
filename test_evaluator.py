import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest

from evaluator import EvaluationError, evaluate_items, load_items, token_overlap_f1


PROJECT_DIR = Path(__file__).resolve().parent


class EvaluatorTests(unittest.TestCase):
    def test_exact_match_after_normalization_scores_one(self):
        output = evaluate_items(
            [
                {
                    "id": "item-1",
                    "reference_answer": "Hello World",
                    "model_answer": "  hello world  ",
                }
            ]
        )

        self.assertEqual(output["results"][0]["score"], 1.0)
        self.assertEqual(
            output["results"][0]["reason"], "Exact match after normalization."
        )
        self.assertEqual(output["summary"]["average_score"], 1.0)
        self.assertEqual(output["summary"]["exact_matches"], 1)
        self.assertEqual(output["summary"]["failed_items"], 0)

    def test_partial_score_uses_token_overlap_f1(self):
        score = token_overlap_f1("red white blue", "red and blue")

        self.assertAlmostEqual(score, 2 / 3)

    def test_punctuation_does_not_block_token_overlap(self):
        output = evaluate_items(
            [
                {
                    "id": "numeric-overlap",
                    "reference_answer": "4",
                    "model_answer": "The answer is 4.",
                }
            ]
        )

        self.assertEqual(output["results"][0]["score"], 0.4)
        self.assertEqual(
            output["results"][0]["reason"],
            "Partial token-overlap F1 after normalization.",
        )

    def test_missing_answer_scores_zero(self):
        output = evaluate_items([{"id": "missing", "reference_answer": "yes"}])

        self.assertEqual(output["results"][0]["score"], 0.0)
        self.assertEqual(output["summary"]["failed_items"], 1)

    def test_empty_input_list_returns_zeroed_summary(self):
        output = evaluate_items([])

        self.assertEqual(output["results"], [])
        self.assertEqual(output["summary"]["average_score"], 0.0)
        self.assertEqual(output["summary"]["number_of_items"], 0)
        self.assertEqual(output["summary"]["exact_matches"], 0)
        self.assertEqual(output["summary"]["failed_items"], 0)

    def test_empty_answer_scores_zero(self):
        output = evaluate_items(
            [
                {"id": "blank-reference", "reference_answer": " ", "model_answer": "yes"},
                {"id": "blank-model", "reference_answer": "yes", "model_answer": ""},
            ]
        )

        self.assertEqual(output["results"][0]["score"], 0.0)
        self.assertEqual(output["results"][1]["score"], 0.0)
        self.assertEqual(
            output["results"][0]["reason"], "Empty reference_answer or model_answer."
        )
        self.assertEqual(output["summary"]["exact_matches"], 0)
        self.assertEqual(output["summary"]["failed_items"], 2)

    def test_original_sample_summary_is_stable(self):
        output = evaluate_items(
            [
                {
                    "id": "1",
                    "prompt": "What is 2 + 2?",
                    "reference_answer": "4",
                    "model_answer": "The answer is 4.",
                },
                {
                    "id": "2",
                    "prompt": "Name one primary color.",
                    "reference_answer": "red",
                    "model_answer": "blue",
                },
                {
                    "id": "3",
                    "prompt": "What planet do humans live on?",
                    "reference_answer": "Earth",
                    "model_answer": "Humans live on Earth.",
                },
                {
                    "id": "4",
                    "prompt": "Reply with exactly: approved",
                    "reference_answer": "approved",
                    "model_answer": "Approved",
                },
            ]
        )

        self.assertEqual(output["summary"]["average_score"], 0.45)
        self.assertEqual(output["summary"]["number_of_items"], 4)
        self.assertEqual(output["summary"]["exact_matches"], 1)
        self.assertEqual(output["summary"]["failed_items"], 1)

    def test_load_items_rejects_non_list_input(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            input_file = Path(tmp_dir) / "bad.json"
            input_file.write_text(json.dumps({"id": "not-a-list"}), encoding="utf-8")

            with self.assertRaisesRegex(EvaluationError, "must be a list"):
                load_items(input_file)

    def test_invalid_item_format_raises_error(self):
        with self.assertRaisesRegex(EvaluationError, "must be an object"):
            evaluate_items(["not an object"])

    def test_non_string_answer_field_raises_error(self):
        with self.assertRaisesRegex(EvaluationError, "expected a string"):
            evaluate_items([{"reference_answer": 4, "model_answer": "4"}])

    def test_cli_success_outputs_json(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / "main.py"),
                str(PROJECT_DIR / "sample_input.json"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0)
        output = json.loads(completed.stdout)
        self.assertIn("results", output)
        self.assertIn("summary", output)
        self.assertEqual(output["summary"]["average_score"], 0.45)
        self.assertEqual(output["summary"]["number_of_items"], 4)
        self.assertEqual(output["summary"]["exact_matches"], 1)
        self.assertEqual(output["summary"]["failed_items"], 1)

    def test_cli_missing_file_outputs_json_error(self):
        completed = subprocess.run(
            [
                sys.executable,
                str(PROJECT_DIR / "main.py"),
                str(PROJECT_DIR / "missing-input.json"),
            ],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 1)
        error = json.loads(completed.stderr)
        self.assertIn("error", error)
        self.assertIn("Missing file", error["error"])


if __name__ == "__main__":
    unittest.main()
