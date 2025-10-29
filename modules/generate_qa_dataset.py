import os
import glob
import json


def _norm_key(key: str) -> str:
    return key.strip().lower()


def _extract_rows_from_json(obj):
    """
    Accepts content loaded from an output_json/*.json file.
    Returns a uniform list of dicts with keys: Date, Day, Details (best-effort).
    Handles both:
      - list of dicts (preferred, produced by GridBasedOcrExtractor)
      - list with first row headers then data rows
    """
    rows = []
    if isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict)):
        # Already a list of dicts
        for d in obj:
            if not isinstance(d, dict):
                continue
            # Use flexible key access
            mapped = {
                "Date": d.get("Date") or d.get("date") or d.get("DATE") or "",
                "Day": d.get("Day") or d.get("day") or d.get("DAY") or "",
                "Details": d.get("Details") or d.get("details") or d.get("DETAILS") or "",
            }
            rows.append(mapped)
        return rows

    if isinstance(obj, list) and len(obj) >= 2 and isinstance(obj[0], list):
        headers = [str(h).strip() for h in obj[0]]
        header_map = {_norm_key(h): i for i, h in enumerate(headers)}
        for r in obj[1:]:
            if not isinstance(r, list):
                continue
            def pick(name: str):
                idx = header_map.get(_norm_key(name))
                return r[idx] if idx is not None and idx < len(r) else ""
            rows.append({
                "Date": pick("Date"),
                "Day": pick("Day"),
                "Details": pick("Details"),
            })
        return rows

    return rows


def build_qa_pairs_from_rows(rows, source_name: str):
    qa_pairs = []
    for item in rows:
        date = str(item.get("Date", "")).strip()
        day = str(item.get("Day", "")).strip()
        details = str(item.get("Details", "")).strip()

        if not details:
            continue

        qa_pairs.extend([
            {"prompt": f"When does {details.lower()}?",
             "response": f"{details} happens on {date} ({day})."},
            {"prompt": f"What happens on {date}?",
             "response": f"On {date} ({day}), {details}."},
            {"prompt": f"Which event is on {day}?",
             "response": f"On {day}, {details} ({date})."}
        ])
    return qa_pairs


def main():
    os.makedirs("data", exist_ok=True)
    output_path = "data/qa_data.jsonl"

    all_json_files = sorted(glob.glob("output_json/*.json"))
    total_pairs = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for fp in all_json_files:
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                rows = _extract_rows_from_json(obj)
                qa_pairs = build_qa_pairs_from_rows(rows, os.path.basename(fp))
                for pair in qa_pairs:
                    json.dump(pair, out_f, ensure_ascii=False)
                    out_f.write("\n")
                total_pairs += len(qa_pairs)
            except Exception as e:
                print(f"Skipping {fp}: {e}")

    print(f"✅ Created {total_pairs} Q&A pairs from {len(all_json_files)} files")
    print(f"💾 Saved to {output_path}")


if __name__ == "__main__":
    main()
