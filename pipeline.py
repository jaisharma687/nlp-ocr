import argparse
import os
import subprocess
import sys

# Local imports
from modules.ProcessImages import convert_and_process_all_images
from modules.GenerateQADataset import main as generate_qa_main
from model.TrainSLM import train_model
from model.AnalyzeTokens import count_tokens


def step_ocr_tables():
    print("\n=== STEP 1/3: OCR Tables → output_csv/ & output_json/ ===")
    convert_and_process_all_images()


def step_generate_qa():
    print("\n=== STEP 2/3: Generate Q&A JSONL from output_json/*.json ===")
    os.makedirs("data", exist_ok=True)
    generate_qa_main()


def step_train_slm(epochs: int):
    print("\n=== STEP 3/3: Train compact SLM from scratch ===")
    train_model(jsonl_path="data/qa_data.jsonl", epochs=epochs)


def step_analyze_tokens(jsonl_path: str = "data/qa_data.jsonl"):
    print("\n=== ANALYZE: Token statistics for training data ===")
    count_tokens(jsonl_path=jsonl_path)

def maybe_run_app(run_app: bool):
    if not run_app:
        return
    print("\n=== LAUNCH: Streamlit app (Ctrl+C to stop) ===")
    # Launch streamlit in foreground so user can interact
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py"]
    subprocess.run(cmd, check=False)


def main():
    parser = argparse.ArgumentParser(description="End-to-end pipeline: OCR→QA gen→train→(optional app)")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR table extraction")
    parser.add_argument("--skip-qa", action="store_true", help="Skip Q&A dataset generation")
    parser.add_argument("--skip-train", action="store_true", help="Skip SLM training")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs for SLM")
    parser.add_argument("--run-app", action="store_true", help="Launch Streamlit app at the end")
    parser.add_argument("--analyze", action="store_true", help="Print token statistics after QA generation or training")
    args = parser.parse_args()

    if not args.skip_ocr:
        step_ocr_tables()
    else:
        print("[skip] OCR tables")

    if not args.skip_qa:
        step_generate_qa()
    else:
        print("[skip] Generate Q&A")

    if not args.skip_train:
        step_train_slm(args.epochs)
    else:
        print("[skip] Train SLM")

    if args.analyze:
        step_analyze_tokens("data/qa_data.jsonl")

    maybe_run_app(args.run_app)


if __name__ == "__main__":
    main()
