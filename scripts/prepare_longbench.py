"""Fetch a pinned official LongBench archive; write a small reproducible JSONL."""

import argparse
import hashlib
import json
import random
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/longbench.jsonl"))
    parser.add_argument("--tasks", nargs="+", default=["qasper", "gov_report", "repobench-p"])
    parser.add_argument("--per-task", type=int, default=1)
    args = parser.parse_args()
    if args.per_task < 1:
        parser.error("per-task must be positive")
    archive = hf_hub_download(
        "zai-org/LongBench", "data.zip", repo_type="dataset", revision=REVISION
    )
    rows = []
    with zipfile.ZipFile(archive) as data:
        for task in args.tasks:
            matches = [
                name
                for name in data.namelist()
                if name.endswith(f"/{task}.jsonl") or name == f"{task}.jsonl"
            ]
            if len(matches) != 1:
                raise ValueError(f"expected exactly one file for {task}: {matches}")
            examples = [
                json.loads(line) for line in data.read(matches[0]).decode().splitlines() if line
            ]
            selected = random.Random(42).sample(
                range(len(examples)), min(args.per_task, len(examples))
            )
            for index in selected:
                example = examples[index]
                question = example.get("input") or (
                    "Summarize this document."
                    if task == "gov_report"
                    else "Continue the code below."
                )
                rows.append(
                    {
                        "id": f"{task}:{index}",
                        "task": task,
                        "context": example["context"],
                        "question": question,
                        "answers": example.get("answers", []),
                        "source": {
                            "dataset": "zai-org/LongBench",
                            "revision": REVISION,
                            "row": index,
                            "context_sha256": hashlib.sha256(
                                example["context"].encode()
                            ).hexdigest(),
                        },
                    }
                )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(json.dumps({"output": str(args.output), "samples": len(rows), "revision": REVISION}))


if __name__ == "__main__":
    main()
