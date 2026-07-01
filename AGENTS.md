# Agent Handoff Notes

Last updated: 2026-07-01

## Project Context

Repository path:

```text
D:\Code\unsia\enexre
```

Main research protocol:

```text
PENELITIAN_STEP.md
```

Current notebook:

```text
Protocol.ipynb
```

The research uses BC5CDR PubTator data for Chemical-Disease NER and relation extraction.

## Environment Status

Local virtual environment exists:

```text
.venv/
```

Jupyter kernel is registered as:

```text
Python (enexre)
```

Kernel metadata in `Protocol.ipynb` has been set to:

```json
{
  "display_name": "Python (enexre)",
  "name": "enexre"
}
```

Use this kernel for local validation, preprocessing, scripts, and documentation. Use Colab GPU later only for heavier model training if needed.

## Important Files Created or Updated

```text
.gitignore
README.md
requirements.txt
setup_venv.bash
configs/config_ner.yaml
configs/config_re.yaml
data/data_manifest.json
scripts/validate_bc5cdr.py
results/dataset_validation.json
Protocol.ipynb
```

`setup_venv.bash` installs dependencies from `requirements.txt`.

## Dataset Status

Official BC5CDR PubTator files were downloaded from:

```text
https://ftp.ncbi.nlm.nih.gov/pub/lu/BC5CDR/
```

Canonical local dataset path is:

```text
data/bc5cdr/train.txt
data/bc5cdr/dev.txt
data/bc5cdr/test.txt
```

Do not use the old notebook path:

```text
bc5cdr/train.txt
bc5cdr/dev.txt
bc5cdr/test.txt
```

There may be old leftover folders such as `bc5cdr/` or `-p/` from earlier notebook shell commands. They were not deleted. Treat `data/bc5cdr/` as the official path.

## Dataset Validation Result

Validation script:

```bash
.venv/Scripts/python.exe scripts/validate_bc5cdr.py
```

Output:

```text
results/dataset_validation.json
```

Validation passed:

```text
Passed: True
```

Summary:

```text
train: docs=500, Chemical=5203, Disease=4182, CID=1038
dev:   docs=500, Chemical=5347, Disease=4244, CID=1012
test:  docs=500, Chemical=5385, Disease=4424, CID=1066
duplicate_pmids_across_subsets=0
invalid_annotation_count=0
invalid_relation_count=0
duplicate_relation_count=0
```

Checksums are recorded in:

```text
data/data_manifest.json
```

## Notebook Download Cell Issue

The notebook previously had a cell:

```python
!wget -q ...
```

This looked stuck because `-q` is quiet mode and prints no progress. It also wrote to the old `bc5cdr/` folder.

The notebook was updated to use Python `urlretrieve` with progress output and to write to:

```text
PROJECT_DIR / "data" / "bc5cdr"
```

If VS Code still shows the old `!wget -q` cell, close and reopen `Protocol.ipynb` so the editor reloads the version from disk. Do not save the stale open notebook over the patched file.

Current expected download cell starts with:

```python
from urllib.request import urlretrieve

DATA_DIR = PROJECT_DIR / "data" / "bc5cdr"
DATA_DIR.mkdir(parents=True, exist_ok=True)
```

## Git Status at Handoff

At the time of this note, changed/untracked files included:

```text
M Protocol.ipynb
M README.md
M data/data_manifest.json
?? scripts/
```

There may now also be this new file:

```text
AGENTS.md
```

Before pushing, check:

```bash
git status --short
```

Then commit when ready:

```bash
git add .
git commit -m "Add BC5CDR validation workflow"
```

## Current Research Progress

Completed:

1. Initial repo scaffold.
2. Local venv and Jupyter kernel setup.
3. BC5CDR official PubTator data downloaded.
4. Dataset manifest updated with source and checksums.
5. Dataset validation script created.
6. Tahap 2 validation passed.

Current protocol position:

```text
Tahap 2 selesai.
Next: Tahap 3 - Membentuk Data NER / BIO labels.
```

## Recommended Next Step

Create a preprocessing script for NER:

```text
scripts/build_ner_dataset.py
```

Expected responsibilities:

1. Read `data/bc5cdr/train.txt`, `dev.txt`, and `test.txt`.
2. Combine title and abstract as `title + " " + abstract`.
3. Convert Chemical and Disease annotations to BIO labels:
   - `B-Chemical`
   - `I-Chemical`
   - `B-Disease`
   - `I-Disease`
   - `O`
4. Use PubMedBERT tokenizer with `offset_mapping`.
5. Ignore special tokens with label `-100`.
6. Save processed data under:

```text
data/processed/ner/
```

7. Save validation/report artifacts under:

```text
results/ner_preprocessing_report.json
```

Keep train/dev/test split unchanged.

