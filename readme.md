## Academic Calendar OCR + Compact SLM QA

End-to-end pipeline to extract academic calendar tables from PDFs/images and answer questions about an uploaded calendar using a compact, scratch-trained transformer (no external LLMs). The answer is grounded strictly in the uploaded file.

### What this repo does
- Extracts table structure and text from academic calendar images/PDFs to `output_csv/` and `output_json/`.
- Trains a tiny character-level transformer (SLM) on synthetic Q→A pairs made from calendar rows.
- Runs a Streamlit app where users upload a calendar file and ask questions. Answers come only from that file’s content.

## Architecture

```
Uploaded PDF/Image
  → modules/TableExtractor.py (locate table, perspective correction)
  → modules/TableGridDetector.py (detect lines, compute cells)
  → modules/GridBasedOcrExtractor.py (OCR each cell)
  → output_json/<file>.json + output_csv/<file>.csv

Training (optional)
  → modules/GenerateQADataset.py (build JSONL from output_json/*.json)
  → model/TrainSLM.py (train MiniTransformer → model/saved_model.pt)

Inference (app)
  → app.py + inference.py
    - Extract table from uploaded file
    - Build compact context from that file
    - Lightweight retrieval over the file’s rows
    - Answer via SLM; deterministic fallback for date queries
```

## Unified pipeline (sequential)

Run the entire flow with one command:

```bash
source venv/bin/activate
python3 pipeline.py --epochs 10 --analyze --run-app
```

What runs (in order):
- PDFs in `Dataset/` → PNGs in `Dataset_Images/`
- OCR tables → `output_json/` and `output_csv/`
- Q&A dataset → `data/qa_data.jsonl`
- Train SLM → `model/saved_model.pt`
- (Optional) Analyze token stats → prints lengths and vocab size
- Optionally launches the Streamlit app

Flags:
- `--skip-pdf`  skip PDF→image conversion
- `--skip-ocr`  skip OCR table extraction
- `--skip-qa`   skip Q&A dataset generation
- `--skip-train` skip SLM training
- `--epochs N`  training epochs (default 10)
- `--run-app`   launch the app after training
- `--analyze`   print token statistics for `data/qa_data.jsonl`

## Key directories and files

- `modules/TableExtractor.py`: Finds the calendar table in the image, fixes perspective, adds padding.
- `modules/TableGridDetector.py`: Detects vertical/horizontal lines to build a grid of cells.
- `modules/GridBasedOcrExtractor.py`: OCR per cell using Tesseract, writes CSV/JSON. Returns a list-of-rows or list-of-dicts.
- `output_json/`: JSON tables produced by OCR; used as the source for training data and app context.
- `output_csv/`: CSV mirror of the table (debug/useful for inspection).
- `modules/GenerateQADataset.py`: Scans all `output_json/*.json` and creates `data/qa_data.jsonl` with synthetic Q→A pairs.
- SLM components (MiniTransformer): compact character-level model implementation.
- `model/tokenizer.py`: Simple char tokenizer with PAD handling.
- `model/TrainSLM.py`: Trains from scratch on Q→A JSONL, saves `model/saved_model.pt`.
- `inference.py`: Single-file OCR→JSON→context; heuristic retrieval; answer generation with SLM and date fallback.
- `app.py`: Streamlit UI to upload a calendar and ask questions.
- `modules/PdfToImage.py`: Batch PDF→PNG conversion for the pipeline.
- `modules/ProcessImages.py`: Batch image OCR runner.

## Processing pipeline (OCR)

1) PDF→Image (when needed)
- `modules/PdfToImage.py` converts each PDF in `Dataset/` to a single PNG in `Dataset_Images/` (first page).

2) Table extraction (`modules/TableExtractor.py`)
- Thresholds, inverts, dilates to find rectangular contours.
- Picks the largest rectangle as the table, applies perspective transform.
- Adds ~10% padding for safer OCR around edges.

3) Grid detection (`modules/TableGridDetector.py`)
- Detects lines, computes intersections, builds a 2D cell map with coordinates.

4) Cell OCR (`modules/GridBasedOcrExtractor.py`)
- Crops each cell (with inner padding), preprocesses (gray, otsu, denoise, resize small cells), calls Tesseract.
- Cleans symbols, collapses whitespace.
- Writes `output_csv/<name>.csv` and `output_json/<name>.json`.

Output JSON shapes supported
- List of dicts (preferred): `[{"Date": "...", "Day": "...", "Details": "..."}, ...]`
- Header+rows: `[["Date","Day","Details"], ["...","...","..."], ...]`

## Compact SLM (trained from scratch)

- Model: `MiniTransformer` (small embedding, 2 heads, 2 layers by default; character-level).
- Tokenizer: character vocabulary derived from training text (+ `<PAD>`), with `ignore_index` in loss.
- Training data: `modules/generate_qa_dataset.py` synthesizes Q→A pairs from all `output_json/*.json` files.

Training steps
```bash
pip install -r requirements.txt
brew install tesseract poppler  # macOS system deps

python /Users/noneofyourbusiness/Developer/Projects/nlp-ocr/modules/GenerateQADataset.py
python /Users/noneofyourbusiness/Developer/Projects/nlp-ocr/model/TrainSLM.py
```
This writes `data/qa_data.jsonl` and `model/saved_model.pt` (plus vocabulary state).

## Upload-and-answer app

- Starts Streamlit, loads `model/saved_model.pt`.
- On upload, runs the same OCR pipeline on the single file and shows a preview.
- Builds compact context lines from the extracted table.
- Lightweight retrieval: pick the most relevant rows to the question by lexical token overlap.
- If the question asks "when"/"date", directly returns the row’s Date (deterministic fallback).
- Otherwise, prompts the SLM with just the selected rows and the question; if output looks gibberish, falls back to the Date.

Run the app
```bash
streamlit run /Users/noneofyourbusiness/Developer/Projects/nlp-ocr/app.py
```

Why answers are grounded
- The app never consults training data at inference time.
- It conditions only on the uploaded file’s extracted table (or returns the row’s Date directly for date queries).
- No external or cloud LLMs are used.

## Data generation details

`modules/GenerateQADataset.py` does the following:
- Loads all JSON tables from `output_json/*.json`.
- Normalizes into rows with keys: `Date`, `Day`, `Details`.
- Emits three Q→A templates per row:
  - When does <details>?
  - What happens on <date>?
  - Which event is on <day>?
- Writes each as one line in `data/qa_data.jsonl`.

## Configuration and tuning

- You can adjust MiniTransformer hyperparameters in the model implementation (embedding size, heads, layers, block size).
- Increase epochs/batch size in `model/TrainSLM.py` for better fit (tradeoff vs overfitting/compute).
- The retrieval in `inference.py` is lexical; you can experiment with simple TF-IDF or fuzzy match while staying non-LLM.
- The deterministic date fallback ensures robust answers to most "when/date" queries.

## Limitations

- Character-level SLM can be brittle and produce noisy generations; thus the retrieval + fallback constraints.
- OCR accuracy depends on image quality, line detection, and Tesseract configuration.
- Multi-page PDFs: the provided converter only uses the first page by default.

## Common commands

Install deps
```bash
pip install -r requirements.txt
brew install tesseract poppler
```

Train SLM
```bash
python3 /Users/noneofyourbusiness/Developer/Projects/nlp-ocr/modules/GenerateQADataset.py
python3 /Users/noneofyourbusiness/Developer/Projects/nlp-ocr/model/TrainSLM.py
```

Run the app
```bash
streamlit run /Users/noneofyourbusiness/Developer/Projects/nlp-ocr/app.py
```