from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
from typing import Iterable, Sequence
from urllib.request import urlopen

GUTENBERG_URL = "https://www.gutenberg.org/files/5740/5740-t/5740-t.tex"
PROPOSITION_RE = re.compile(r"^(?P<id>[1-7](?:\.\d+)*)\s+(?P<text>\S.*)$")
TEX_PROPOSITION_MACROS = {
    "en": r"\PropositionE",
    "de": r"\PropositionG",
}


@dataclass(frozen=True)
class Proposition:
    id: str
    parent_id: str | None
    depth: int
    texts: dict[str, str]


def proposition_depth(prop_id: str) -> int:
    return 1 if "." not in prop_id else len(prop_id.split(".")[1]) + 1


def parent_id(prop_id: str, known_ids: set[str] | None = None) -> str | None:
    if "." not in prop_id:
        return None
    head, suffix = prop_id.split(".", 1)
    candidates = [head] + [f"{head}.{suffix[:i]}" for i in range(1, len(suffix))]
    if known_ids is None:
        return candidates[-1] if candidates else None
    for candidate in reversed(candidates):
        if candidate in known_ids:
            return candidate
    return head if head in known_ids else None


def strip_gutenberg_boilerplate(raw: str) -> str:
    start_markers = [
        "TRACTATUS LOGICO-PHILOSOPHICUS",
        "LOGISCH-PHILOSOPHISCHE ABHANDLUNG",
    ]
    starts = [raw.find(marker) for marker in start_markers if raw.find(marker) != -1]
    start = min(starts) if starts else 0
    end_match = re.search(r"\*\*\* END OF (?:THE )?PROJECT GUTENBERG", raw)
    end = end_match.start() if end_match else len(raw)
    return raw[start:end]


def _read_braced(raw: str, start: int) -> tuple[str, int]:
    if start >= len(raw) or raw[start] != "{":
        raise ValueError("expected opening brace")
    depth = 0
    chars: list[str] = []
    i = start
    while i < len(raw):
        char = raw[i]
        if char == "\\" and i + 1 < len(raw):
            chars.append(raw[i : i + 2])
            i += 2
            continue
        if char == "{":
            depth += 1
            if depth > 1:
                chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars), i + 1
            chars.append(char)
        else:
            chars.append(char)
        i += 1
    raise ValueError("unterminated braced block")


def clean_tex(text: str) -> str:
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\footnote\{(?:[^{}]|\{[^{}]*\})*\}", "", text, flags=re.DOTALL)
    text = text.replace("``", '"').replace("''", '"').replace("~", " ")
    text = text.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_")
    text = re.sub(r"\\(?:emph|textit|text|German|BookTitle|Emph)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\DPtypo\{([^{}]*)\}\{([^{}]*)\}", r"\2", text)
    text = re.sub(r"\\(?:PropERef|PropGRef|hyperref)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[a-zA-Z]+(?:\[[^\]]*\])?", " ", text)
    text = text.replace("{", "").replace("}", "").replace("$", "")
    return re.sub(r"\s+", " ", text).strip()


def _parse_tex_macro(raw: str, marker: str, language: str) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    i = 0
    while True:
        macro = raw.find(marker, i)
        if macro == -1:
            break
        # Skip the macro definition near the top of the TeX source.
        if macro > 0 and raw[macro - 10 : macro].find("newcommand") != -1:
            i = macro + len(marker)
            continue
        j = macro + len(marker)
        while j < len(raw) and raw[j].isspace():
            j += 1
        if j >= len(raw) or raw[j] != "{":
            i = j
            continue
        prop_id, j = _read_braced(raw, j)
        while j < len(raw) and raw[j].isspace():
            j += 1
        if j >= len(raw) or raw[j] != "{":
            i = j
            continue
        text, j = _read_braced(raw, j)
        if PROPOSITION_RE.match(f"{prop_id} x"):
            rows.append((language, prop_id, clean_tex(text)))
        i = j
    return rows


def parse_tex_propositions(raw: str, languages: Sequence[str] = ("en",)) -> list[tuple[str, str, str]]:
    rows: list[tuple[str, str, str]] = []
    for language in languages:
        marker = TEX_PROPOSITION_MACROS.get(language)
        if marker is None:
            raise ValueError(f"Unsupported TeX proposition language: {language}")
        rows.extend(_parse_tex_macro(raw, marker, language))
    return rows


def parse_plain_propositions(raw: str, languages: Sequence[str] = ("en",)) -> list[tuple[str, str, str]]:
    body = strip_gutenberg_boilerplate(raw)
    raw_rows: list[tuple[str, str]] = []
    current_id: str | None = None
    current_text: list[str] = []

    for line in body.splitlines():
        clean = line.strip()
        if not clean:
            continue
        match = PROPOSITION_RE.match(clean)
        if match:
            if current_id is not None:
                raw_rows.append((current_id, " ".join(current_text).strip()))
            current_id = match.group("id")
            current_text = [match.group("text")]
        elif current_id is not None:
            current_text.append(clean)

    if current_id is not None:
        raw_rows.append((current_id, " ".join(current_text).strip()))

    seen_counts: dict[str, int] = {}
    rows: list[tuple[str, str, str]] = []
    for prop_id, text in raw_rows:
        occurrence = seen_counts.get(prop_id, 0)
        seen_counts[prop_id] = occurrence + 1
        if occurrence >= len(languages):
            continue
        rows.append((languages[occurrence], prop_id, clean_tex(text)))
    return rows


def parse_propositions(
    raw: str,
    languages: Sequence[str] = ("en",),
    allow_incomplete: bool = False,
) -> list[Proposition]:
    languages = tuple(languages)
    if not languages:
        raise ValueError("At least one language is required")

    if any(marker in raw for marker in TEX_PROPOSITION_MACROS.values()):
        rows = parse_tex_propositions(raw, languages)
    else:
        rows = parse_plain_propositions(raw, languages)

    texts_by_id: dict[str, dict[str, str]] = {}
    ordered_ids: list[str] = []
    for language, prop_id, text in rows:
        if len(text) < 2:
            continue
        if prop_id not in texts_by_id:
            texts_by_id[prop_id] = {}
            ordered_ids.append(prop_id)
        # Keep the first text for a language/proposition pair. Repeated source
        # entries are treated as duplicates, not extra training variants.
        texts_by_id[prop_id].setdefault(language, text)

    required = set(languages)
    known: set[str] = set()
    propositions: list[Proposition] = []
    for prop_id in ordered_ids:
        texts = {language: texts_by_id[prop_id][language] for language in languages if language in texts_by_id[prop_id]}
        if not allow_incomplete and set(texts) != required:
            continue
        pid = parent_id(prop_id, known)
        propositions.append(
            Proposition(id=prop_id, parent_id=pid, depth=proposition_depth(prop_id), texts=texts)
        )
        known.add(prop_id)
    return propositions


def enrich_propositions(propositions: Iterable[Proposition]) -> list[dict]:
    props = list(propositions)
    ids = [p.id for p in props]
    by_parent: dict[str | None, list[str]] = {}
    for p in props:
        by_parent.setdefault(p.parent_id, []).append(p.id)

    enriched: list[dict] = []
    for i, p in enumerate(props):
        siblings = [sid for sid in by_parent.get(p.parent_id, []) if sid != p.id]
        ancestors: list[str] = []
        cursor = p.parent_id
        by_id = {q.id: q for q in props}
        while cursor is not None and cursor in by_id:
            ancestors.append(cursor)
            cursor = by_id[cursor].parent_id
        ancestors.reverse()

        row = asdict(p)
        row.update(
            {
                "next_id": ids[i + 1] if i + 1 < len(ids) else None,
                "children": by_parent.get(p.id, []),
                "siblings": siblings,
                "ancestor_chain": ancestors,
                "child_count": len(by_parent.get(p.id, [])),
            }
        )
        enriched.append(row)
    return enriched


def download_text(url: str = GUTENBERG_URL) -> str:
    with urlopen(url, timeout=30) as response:
        data = response.read()
        charset = response.headers.get_content_charset() or "latin-1"
        return data.decode(charset, errors="replace")


def build_dataset(
    raw_text: str,
    output_path: str | Path,
    languages: Sequence[str] = ("en",),
    allow_incomplete: bool = False,
) -> list[dict]:
    propositions = parse_propositions(raw_text, languages=languages, allow_incomplete=allow_incomplete)
    if not propositions:
        raise ValueError("No propositions parsed from source text")
    enriched = enrich_propositions(propositions)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
    return enriched
