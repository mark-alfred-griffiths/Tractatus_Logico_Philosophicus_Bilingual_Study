from __future__ import annotations

import argparse
from pathlib import Path

from tractatus_structure_latents.data.dataset import GUTENBERG_URL, build_dataset, download_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Build enriched Tractatus proposition dataset.")
    parser.add_argument("--source", type=Path, help="Local raw Tractatus text file. If omitted, downloads Project Gutenberg text.")
    parser.add_argument("--url", default=GUTENBERG_URL, help="Source URL used when --source is omitted.")
    parser.add_argument("--output", type=Path, default=Path("tractatus_structure_latents/data/tractatus.json"))
    parser.add_argument(
        "--languages",
        default="en",
        help="Comma-separated proposition languages to extract, in source order for plain text. Use en,de for bilingual data.",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Keep propositions even when one of the requested languages is missing.",
    )
    args = parser.parse_args()

    languages = tuple(language.strip() for language in args.languages.split(",") if language.strip())
    raw = args.source.read_text(encoding="utf-8") if args.source else download_text(args.url)
    rows = build_dataset(raw, args.output, languages=languages, allow_incomplete=args.allow_incomplete)
    language_list = ",".join(languages)
    print(f"wrote {len(rows)} propositions ({language_list}) to {args.output}")


if __name__ == "__main__":
    main()
