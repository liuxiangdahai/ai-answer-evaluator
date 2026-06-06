# Minimal Python CLI Evaluator

This project scores AI assistant answers against reference answers using deterministic rules only. The evaluator is fully local and does not call any external LLM API.

Requires Python 3.10+.

## Project Structure

- `evaluator.py` - core loading, validation, normalization, and scoring logic
- `main.py` - command-line interface
- `sample_input.json` - example evaluation file
- `test_evaluator.py` - unit tests
- `README.md` - usage notes and design tradeoffs

## Input Format

The input file must be JSON with a top-level list of evaluation items. Each item is an object and may include:

- `id`
- `prompt`
- `reference_answer`
- `model_answer`

Example:

```json
[
  {
    "id": "1",
    "prompt": "What is 2 + 2?",
    "reference_answer": "4",
    "model_answer": "The answer is 4."
  },
  {
    "id": "2",
    "prompt": "Name one primary color.",
    "reference_answer": "red",
    "model_answer": "blue"
  },
  {
    "id": "3",
    "prompt": "What planet do humans live on?",
    "reference_answer": "Earth",
    "model_answer": "Humans live on Earth."
  },
  {
    "id": "4",
    "prompt": "Reply with exactly: approved",
    "reference_answer": "approved",
    "model_answer": "Approved"
  }
]
```

## How To Run

Run the evaluator:

```bash
python main.py sample_input.json
```

Run the tests:

```bash
python -m unittest -v
```

The CLI prints JSON containing per-item results and a summary:

```json
{
  "results": [
    {
      "id": "1",
      "score": 0.4,
      "reason": "Partial token-overlap F1 after normalization."
    },
    {
      "id": "2",
      "score": 0.0,
      "reason": "No token overlap after normalization."
    },
    {
      "id": "3",
      "score": 0.4,
      "reason": "Partial token-overlap F1 after normalization."
    },
    {
      "id": "4",
      "score": 1.0,
      "reason": "Exact match after normalization."
    }
  ],
  "summary": {
    "average_score": 0.45,
    "number_of_items": 4,
    "exact_matches": 1,
    "failed_items": 1
  }
}
```

## Scoring Rules

1. Normalize `reference_answer` and `model_answer` by trimming surrounding whitespace and lowercasing.
2. Score `0.0` if either answer field is missing, `null`, empty, or only whitespace.
3. Score `1.0` for exact matches after normalization.
4. Otherwise, tokenize each normalized answer with the regex `[a-z0-9]+`.
5. Compute token-overlap F1 using token counts.

`failed_items` counts items with a final score of `0.0`.
The `prompt` field is preserved in the input format but does not affect scoring.

## Error Handling

The CLI exits with status code `1` and prints a JSON error to stderr for:

- Missing input file
- Invalid JSON
- Top-level JSON that is not a list
- Items that are not JSON objects
- Present answer fields that are not strings

## AI Usage

AI tools were used to clarify the task, scaffold the implementation, review edge cases, and draft documentation. The final code was manually reviewed, tested, and adjusted before submission.

## Tradeoffs

- Exact matching intentionally uses only the required normalization: trim whitespace and lowercase.
- Partial scoring ignores common punctuation by extracting alphanumeric tokens.
- The tokenizer is simple and English-oriented; it does not perform stemming, synonym matching, semantic matching, or factual verification.
- Token-overlap F1 rewards shared words but does not understand word order or whether a longer answer is fully correct.
- Empty strings are treated as invalid answers and receive `0.0`.

## Future Improvements

- Add configurable tokenization rules.
- Add configurable stopword handling.
- Support CSV output for spreadsheet workflows.
- Include per-item precision and recall for debugging.
- Add schema validation for stricter input contracts.
