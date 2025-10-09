#!/usr/bin/env python3
"""Extract skill names from suggested skills JSON into a simple JSON list.

Usage:
    python extract_skill_names.py --input output/skillExpansion2.suggested_skills.json --output output/skill_names.simple.json
"""
import json
from pathlib import Path
import argparse


def extract_skill_names(input_path: Path, output_path: Path, unique: bool = True):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Expecting a list of objects with a 'skill' field
    skills = []
    for item in data:
        name = item.get("skill") if isinstance(item, dict) else None
        if name:
            skills.append(name)

    if unique:
        # Preserve order while removing duplicates
        seen = set()
        unique_skills = []
        for s in skills:
            if s not in seen:
                seen.add(s)
                unique_skills.append(s)
        skills = unique_skills

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(skills, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(skills)} skills to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract skill names to a simple JSON list"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Input suggested skills JSON file",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=False,
        help="Output simple JSON file (default: same folder skill_names.simple.json)",
    )
    parser.add_argument(
        "--no-unique",
        action="store_true",
        help="Do not remove duplicates; keep original order and duplicates",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / "skill_names.simple.json"

    extract_skill_names(input_path, output_path, unique=not args.no_unique)
