# Dataset

The project supports two dataset schemas:

```text
tractatus_structure_latents/data/tractatus.json            English-only legacy dataset
tractatus_structure_latents/data/tractatus_bilingual.json  aligned German/English dataset
```

Both are built from Project Gutenberg's public-domain Tractatus source.

## Bilingual Schema

The bilingual dataset contains one row per numbered proposition. Text realizations are stored under `texts`, while hierarchy-derived supervision is shared by proposition id across languages.

```json
{
  "id": "4.121",
  "parent_id": "4.12",
  "depth": 4,
  "texts": {
    "en": "Propositions cannot represent the logical form...",
    "de": "Die Sätze können die logische Form nicht darstellen..."
  },
  "next_id": "4.1211",
  "children": ["4.1211", "4.1212"],
  "siblings": ["4.122", "4.123"],
  "ancestor_chain": ["4", "4.1", "4.12"],
  "child_count": 2
}
```

Fields:

- `id`: Tractatus proposition number.
- `parent_id`: nearest existing parent proposition, or `null` for top-level propositions.
- `depth`: hierarchy depth inferred from the numbering system.
- `texts.en`: English proposition text.
- `texts.de`: German proposition text.
- `next_id`: next proposition in textual order, or `null` for proposition `7`.
- `children`: direct child propositions.
- `siblings`: propositions with the same parent.
- `ancestor_chain`: ordered ancestors from root to parent.
- `child_count`: number of direct children.

The training loader flattens bilingual rows into proposition-language samples. For example, proposition `4.121` produces one English sample and one German sample, but both share the same structural targets and proposition index.

## Legacy English Schema

The original English-only dataset uses one `text` field:

```json
{
  "id": "4.121",
  "parent_id": "4.12",
  "depth": 4,
  "text": "Propositions cannot represent the logical form...",
  "next_id": "4.1211",
  "children": ["4.1211", "4.1212"],
  "siblings": ["4.122", "4.123"],
  "ancestor_chain": ["4", "4.1", "4.12"],
  "child_count": 2
}
```

The loader remains backward-compatible with this schema and treats it as `en`.

## Dataset Statistics

Bilingual dataset:

```text
proposition rows:          526
flattened text samples:    1052
languages:                 de, en
first proposition:          1
last proposition:           7
```

Hierarchy statistics:

```text
non-leaf nodes:     168
maximum children:   14
```

Depth distribution:

```text
depth 1:   7
depth 2:   25
depth 3:   124
depth 4:   245
depth 5:   118
depth 6:   7
```

Top-level section distribution:

```text
1:   7
2:   79
3:   74
4:   109
5:   151
6:   105
7:   1
```

## Construction

Build the English-only dataset:

```bash
python3 -m tractatus_structure_latents.scripts.build_dataset \
  --output tractatus_structure_latents/data/tractatus.json \
  --languages en
```

Build the bilingual German/English dataset:

```bash
python3 -m tractatus_structure_latents.scripts.build_dataset \
  --output tractatus_structure_latents/data/tractatus_bilingual.json \
  --languages en,de
```

Build from a local source file:

```bash
python3 -m tractatus_structure_latents.scripts.build_dataset \
  --source raw_tractatus.tex \
  --output tractatus_structure_latents/data/tractatus_bilingual.json \
  --languages en,de
```

The TeX parser extracts `\PropositionE{...}` entries as English and `\PropositionG{...}` entries as German. For plain text sources, repeated proposition ids are assigned to languages in the order passed to `--languages`.

By default, requested languages must all be present for a proposition to be kept. Use `--allow-incomplete` to retain partial rows.
