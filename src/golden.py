"""Build the golden set: 100 answerable tickets with reference answers.

Pulled from WixQA's expert-written split (200 real question/answer pairs grounded in
the KB) rather than hand-written, per the plan — hand-writing is reserved for the
adversarial set, where the judgement encoded matters more than the answer content.
"""
import json

from datasets import load_dataset

from src import config

GOLDEN_SIZE = 100


def build_golden_set() -> list[dict]:
    qa = load_dataset(config.KB_DATASET, "wixqa_expertwritten", split="train")
    records = []
    for i, row in enumerate(qa.select(range(GOLDEN_SIZE))):
        records.append(
            {
                "item_id": f"G{i + 1:03d}",
                "question": row["question"],
                "reference_answer": row["answer"],
                "article_ids": row["article_ids"],
                "expected_decision": "RESOLVE",
            }
        )
    return records


def write_golden_jsonl() -> None:
    config.DATA_DIR.joinpath("golden").mkdir(parents=True, exist_ok=True)
    path = config.DATA_DIR / "golden" / "golden_set.jsonl"
    records = build_golden_set()
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} golden items to {path}")


if __name__ == "__main__":
    write_golden_jsonl()
