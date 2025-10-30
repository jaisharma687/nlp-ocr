import json
from model.Tokenizer import SimpleTokenizer

def count_tokens(jsonl_path="data/qa_data.jsonl"):
    # Read all lines
    with open(jsonl_path, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]

    # Combine all text to build tokenizer
    all_text = "\n".join(f"Q: {s['prompt']} A: {s['response']}" for s in samples)
    tokenizer = SimpleTokenizer([all_text])

    total_tokens = 0
    token_counts = []

    for s in samples:
        text = f"Q: {s['prompt']} A: {s['response']}"
        tokens = tokenizer.encode(text)
        token_counts.append(len(tokens))
        total_tokens += len(tokens)

    print(f"📊 Total samples: {len(samples)}")
    print(f"🧩 Total tokens: {total_tokens}")
    print(f"📏 Average tokens per sample: {total_tokens / len(samples):.2f}")
    print(f"⚙️ Vocabulary size: {tokenizer.vocab_size}")
    print(f"🔢 Min length: {min(token_counts)}, Max length: {max(token_counts)}")

if __name__ == "__main__":
    count_tokens()
