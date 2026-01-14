#!/usr/bin/env python3
"""Render an editable entity review HTML for a markdown + grounded JSON pair."""

from __future__ import annotations

import argparse
from pathlib import Path

from information_extraction.review import visualize_entity_review


def _default_output_path(json_path: Path) -> Path:
    return json_path.with_suffix(".review.html")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a review HTML for grounded entity JSON outputs."
    )
    parser.add_argument("--input-md", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-html", type=Path)
    parser.add_argument("--no-file-picker", action="store_true")
    args = parser.parse_args()

    output_path = args.output_html or _default_output_path(args.input_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = visualize_entity_review(
        args.input_md,
        args.input_json,
        show_file_picker=not args.no_file_picker,
    )
    output_path.write_text(str(html), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
