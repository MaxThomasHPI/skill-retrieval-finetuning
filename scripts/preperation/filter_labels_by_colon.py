#!/usr/bin/env python3
"""Filter labels in a JSON dataset to only keep labels containing a double-colon ':'

When used after enrich_skill.py, this effectively removes all non ESCO skill lables.
These lables are often very short, could not be enriched with a description, duplicate or are in conflict with a more specific skill.

Usage:
    python filter_labels_by_colon.py --input data/ESCO/augmented/combinedESCOAugmentedExpanded2.json --output data/ESCO/augmented/combinedESCOAugmentedExpanded2.filtered.json

Behavior:
- For each record in the top-level JSON array, keep only strings in 'pos' and 'neg' lists that contain ':'
- If a record has no 'pos' labels left after filtering, the record will be dropped (configurable)
- Preserves all other fields
"""
import json
from pathlib import Path
import argparse
from typing import List, Dict, Any


def filter_labels(
    record: Dict[str, Any], keep_only_with_colon: bool = True
) -> Dict[str, Any]:
    new_record = dict(record)

    def filter_list(lst: List[str]) -> List[str]:
        if not isinstance(lst, list):
            return lst
        return (
            [x for x in lst if isinstance(x, str) and (":" in x)]
            if keep_only_with_colon
            else [x for x in lst if isinstance(x, str)]
        )

    if "pos" in record:
        new_record["pos"] = filter_list(record.get("pos", []))
    if "neg" in record:
        new_record["neg"] = filter_list(record.get("neg", []))

    return new_record


def process_file(
    input_path: Path, output_path: Path, drop_empty_pos: bool = True
) -> int:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a top-level array of records")

    out: List[Dict[str, Any]] = []
    for rec in data:
        new_rec = filter_labels(rec, keep_only_with_colon=True)
        # If drop_empty_pos is True, skip records with no pos labels
        if drop_empty_pos and (not new_rec.get("pos")):
            continue
        out.append(new_rec)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    return len(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter labels keeping only those with ':'"
    )
    parser.add_argument("--input", "-i", required=True, help="Input JSON file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument(
        "--keep-empty-pos",
        action="store_true",
        help="Keep records even if pos list becomes empty after filtering",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    count = process_file(
        input_path, output_path, drop_empty_pos=not args.keep_empty_pos
    )
    print(f"Wrote {count} records to {output_path}")
