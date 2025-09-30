#### Quick Start 
use: 
 - uv sync
 - uv run -m src.main


### What it does
- Downloads the HF dataset `RJT1990/GeneralThoughtArchive` (first 100 rows by default).
- Loads the NER model `Ihor/gliner-biomed-large-v1.0`.
- Scans each question (first 150 characters), detects biomedical entities across predefined labels.
- Classifies a question as biomedical if:
  - it contains the token "bio" (case-insensitive), or
  - the model predicts at least one entity above the threshold (default 0.80).
- Writes matched records to `Final_Bio_terms.txt`.
- Reconstructs the matching rows from the original HF dataset and saves them to `hf_med_filtered_data/` via `datasets.save_to_disk`.


## How it works

- Entry point: `src/main.py`
  - Seeds  and picks device (`cuda` if available).
  - Iterates over `dataset['question']`, truncates to 150 chars, runs `model.predict_entities(...)`.
  - Keeps only `{"label", "score"}` per entity for logging .
  - Aggregates matched questions into `bio_term` and writes one JSON-like dict per line to `Final_Bio_terms.txt`.
  - Calls `filtered_med_data(...)` to map truncated questions back to full HF rows and saves to `hf_med_filtered_data/`.

- NER model: `src/ner_model.py`
  - Loads `Ihor/gliner-biomed-large-v1.0` to CPU/GPU and exposes `model` and `ner_labels`.

- Dataset: `src/hf_datasets.py`
  - Loads split `train` from `RJT1990/GeneralThoughtArchive` 

- Final- HF : `src/filtered_data.py`
  - Matches records by truncated question text to reconstruct the original rows.