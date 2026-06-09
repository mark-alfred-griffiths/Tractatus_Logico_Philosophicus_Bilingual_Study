from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import torch
from torch.utils.data import Dataset

TOKEN_RE = re.compile(r"[A-Za-z0-9]+|[^\w\s]", re.UNICODE)
PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


@dataclass
class Vocabulary:
    token_to_id: dict[str, int]

    @classmethod
    def build(cls, texts: list[str], min_freq: int = 1) -> "Vocabulary":
        counts: dict[str, int] = {}
        for text in texts:
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
        tokens = [PAD, BOS, EOS, UNK] + sorted(t for t, c in counts.items() if c >= min_freq)
        return cls({token: i for i, token in enumerate(tokens)})

    @property
    def pad_idx(self) -> int:
        return self.token_to_id[PAD]

    def encode(self, text: str) -> list[int]:
        unk = self.token_to_id[UNK]
        return [self.token_to_id[BOS], *[self.token_to_id.get(t, unk) for t in tokenize(text)], self.token_to_id[EOS]]

    def to_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.token_to_id, indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "Vocabulary":
        return cls(json.loads(path.read_text(encoding="utf-8")))


class TractatusDataset(Dataset):
    def __init__(
        self,
        data_path: str | Path,
        vocab: Vocabulary | None = None,
        max_len: int = 96,
        languages: list[str] | None = None,
        language_to_id: dict[str, int] | None = None,
    ):
        self.rows = json.loads(Path(data_path).read_text(encoding="utf-8"))
        self.max_len = max_len
        self.id_to_index = {row["id"]: i + 1 for i, row in enumerate(self.rows)}
        self.max_depth = max(row["depth"] for row in self.rows)
        self.languages = self._resolve_languages(languages)
        self.language_to_id = language_to_id or {language: i for i, language in enumerate(self.languages)}
        self.samples = self._build_samples()
        if not self.samples:
            raise ValueError(f"No text samples found for languages: {','.join(self.languages)}")
        self.vocab = vocab or Vocabulary.build([sample["text"] for sample in self.samples])

    def _row_texts(self, row: dict) -> dict[str, str]:
        if "texts" in row:
            return row["texts"]
        if "text" in row:
            return {"en": row["text"]}
        return {}

    def _resolve_languages(self, languages: list[str] | None) -> list[str]:
        if languages:
            return languages
        seen: list[str] = []
        for row in self.rows:
            for language in self._row_texts(row):
                if language not in seen:
                    seen.append(language)
        return seen or ["en"]

    def _build_samples(self) -> list[dict]:
        samples: list[dict] = []
        for row in self.rows:
            texts = self._row_texts(row)
            for language in self.languages:
                text = texts.get(language)
                if text:
                    samples.append({"row": row, "id": row["id"], "language": language, "text": text})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def proposition_count(self) -> int:
        return len(self.rows)

    @property
    def language_count(self) -> int:
        return len(self.language_to_id)

    def _index_or_zero(self, prop_id: str | None) -> int:
        return self.id_to_index.get(prop_id, 0) if prop_id else 0

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        row = sample["row"]
        ids = self.vocab.encode(sample["text"])[: self.max_len]
        if ids[-1] != self.vocab.token_to_id[EOS]:
            ids[-1] = self.vocab.token_to_id[EOS]
        return {
            "id": sample["id"],
            "language": sample["language"],
            "language_id": torch.tensor(self.language_to_id[sample["language"]], dtype=torch.long),
            "index": torch.tensor(self.id_to_index[row["id"]], dtype=torch.long),
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "parent": torch.tensor(self._index_or_zero(row["parent_id"]), dtype=torch.long),
            "depth": torch.tensor(row["depth"], dtype=torch.long),
            "next": torch.tensor(self._index_or_zero(row["next_id"]), dtype=torch.long),
            "child_count": torch.tensor(row["child_count"], dtype=torch.float),
        }


def collate_batch(batch: list[dict], pad_idx: int = 0) -> dict[str, torch.Tensor | list[str]]:
    lengths = torch.tensor([len(item["input_ids"]) for item in batch], dtype=torch.long)
    max_len = int(lengths.max())
    padded = torch.full((len(batch), max_len), pad_idx, dtype=torch.long)
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        padded[i, : len(ids)] = ids
    return {
        "ids": [item["id"] for item in batch],
        "languages": [item["language"] for item in batch],
        "language_ids": torch.stack([item["language_id"] for item in batch]),
        "index": torch.stack([item["index"] for item in batch]),
        "input_ids": padded,
        "lengths": lengths,
        "decoder_ids": padded[:, :-1],
        "targets": padded[:, 1:],
        "parent": torch.stack([item["parent"] for item in batch]),
        "depth": torch.stack([item["depth"] for item in batch]),
        "next": torch.stack([item["next"] for item in batch]),
        "child_count": torch.stack([item["child_count"] for item in batch]),
    }
