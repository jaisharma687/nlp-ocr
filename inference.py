import os
import json
import torch
from typing import List, Dict

from model.Model import MiniTransformer
from model.Tokenizer import SimpleTokenizer

import modules.TableExtractor as te
import modules.TableGridDetector as tgd
import modules.GridBasedOcrExtractor as gboe
from pdf2image import convert_from_path
from PIL import Image
import re


def _ensure_image(input_path: str) -> str:
    """
    Accepts a PDF or image path and returns a PNG image path for processing.
    For PDFs, converts the first page to PNG in a temp folder.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        return input_path

    if ext == ".pdf":
        os.makedirs(".tmp_infer", exist_ok=True)
        pages = convert_from_path(input_path, dpi=300)
        image: Image.Image = pages[0]
        out_path = os.path.join(".tmp_infer", os.path.basename(input_path) + ".png")
        image.save(out_path, "PNG")
        return out_path

    raise ValueError(f"Unsupported file type: {ext}")


def extract_calendar_json(input_path: str) -> Dict:
    """
    Runs the table extraction + OCR pipeline on a single input (PDF or image),
    and returns the parsed table content as a dict. Also writes a JSON file to output_json/.
    """
    image_path = _ensure_image(input_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # Ensure output directories exist
    os.makedirs("output_json", exist_ok=True)
    os.makedirs("output_csv", exist_ok=True)

    table_extractor = te.TableExtractor(image_path)
    perspective_corrected_image = table_extractor.execute()

    grid_detector = tgd.TableGridDetector(perspective_corrected_image)
    grid_result = grid_detector.execute()

    json_output_path = os.path.join("output_json", f"{base_name}.json")
    csv_output_path = os.path.join("output_csv", f"{base_name}.csv")

    ocr_extractor = gboe.GridBasedOcrExtractor(
        perspective_corrected_image,
        grid_result,
        csv_output_path=csv_output_path,
        json_output_path=json_output_path,
    )
    table_data = ocr_extractor.execute()

    # Return table_data (already a dict/list) for immediate use
    return table_data


def build_context_lines(table_data) -> List[str]:
    """
    Flattens table JSON into simple context lines that are easy to search/condition on.
    Expected table_data format: {"rows": [[cell, ...], ...]} or similar.
    """
    lines: List[str] = []
    # Be defensive about structure
    if isinstance(table_data, list):
        rows = table_data
    elif isinstance(table_data, dict):
        rows = table_data.get("rows") or table_data.get("table") or table_data.get("data") or []
    else:
        rows = []
    for row in rows:
        if isinstance(row, list):
            line = " | ".join([str(c).strip() for c in row if c is not None])
            if line:
                lines.append(line)
        elif isinstance(row, dict):
            # Join values in a stable order
            vals = [str(v).strip() for k, v in sorted(row.items()) if v is not None]
            line = " | ".join(vals)
            if line:
                lines.append(line)
    return lines


def _normalize_text(s: str) -> List[str]:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    tokens = [t for t in s.split() if t]
    return tokens


def select_best_rows(question: str, context_lines: List[str], top_k: int = 3) -> List[str]:
    # Skip headers if present
    lines = [ln for ln in context_lines if "|" in ln]
    if lines and lines[0].lower().startswith("date |"):
        lines = lines[1:]

    q_tokens = set(_normalize_text(question))
    scored = []
    for ln in lines:
        ln_tokens = set(_normalize_text(ln))
        score = len(q_tokens & ln_tokens)
        # Simple boost if any date-like pattern present
        if re.search(r"\d{2}[./-]\d{2}[./-]\d{4}", ln):
            score += 1
        scored.append((score, ln))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ln for _, ln in scored[:top_k] if _ > 0] or lines[:1]


def extract_date_from_row(row: str) -> str:
    # Expect "Date | Day | Details" format
    parts = [p.strip() for p in row.split("|")]
    if len(parts) >= 1:
        return parts[0]
    return ""


def load_model(checkpoint_path: str = "model/saved_model.pt"):
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    stoi, itos = ckpt["stoi"], ckpt["itos"]
    # Build tokenizer using recovered vocab
    tokenizer = SimpleTokenizer(["".join(itos.values())])
    model = MiniTransformer(vocab_size=len(stoi))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate(model: MiniTransformer, tokenizer: SimpleTokenizer, prompt: str, max_new_tokens: int = 120) -> str:
    input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
    # Respect model's positional embedding limit
    max_ctx = model.pos_embed.num_embeddings
    if input_ids.shape[1] > max_ctx:
        input_ids = input_ids[:, -max_ctx:]

    for _ in range(max_new_tokens):
        # Sliding window over context to avoid exceeding pos embeddings
        windowed = input_ids[:, -max_ctx:]
        logits = model(windowed)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        input_ids = torch.cat((input_ids, next_token.unsqueeze(0)), dim=1)
    return tokenizer.decode(input_ids[0].tolist())


def answer_question_from_context(question: str, context_lines: List[str], model, tokenizer) -> str:
    """
    Compose a compact prompt over the most relevant lines only.
    Provide a deterministic fallback using the table row if the model output looks invalid.
    """
    candidate_rows = select_best_rows(question, context_lines, top_k=3)

    # Deterministic rule: if the question asks "when" or mentions "date",
    # answer directly with the Date column from the best row.
    q_norm = question.strip().lower()
    if ("when" in q_norm) or ("date" in q_norm):
        row = candidate_rows[0] if candidate_rows else ""
        date = extract_date_from_row(row)
        if date:
            return date
    context = "\n".join(candidate_rows)
    prompt = (
        "You are a compact QA model. Answer ONLY using the row(s) below.\n" 
        "Return the exact date(s) if the question asks when.\n\n"
        f"Rows:\n{context}\n\nQ: {question}\nA:"
    )
    out = generate(model, tokenizer, prompt, max_new_tokens=64)
    anchor = out.rfind("A:")
    answer = out[anchor + 2:].strip() if anchor != -1 else out.strip()

    # Heuristic fallback: if answer looks like gibberish or has no digits/dates
    looks_bad = (
        len(answer) > 80 and not re.search(r"\d", answer)
    ) or re.search(r"([a-z])\1{4,}", answer)

    if looks_bad:
        # Try to extract a date from the best row
        row = candidate_rows[0] if candidate_rows else ""
        date = extract_date_from_row(row)
        if date:
            return date
    return answer


