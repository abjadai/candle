
import subprocess
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

# Each entry pairs a model's output file with a display name for the table column.
# The column named RAW_DATA_COLUMN is used as the baseline for % improvement.
MODELS = [
    {"file_path": "benchmark_data/WildSAText/wildsatext_raw.txt",
     "model_name": "Raw Data"},
    {"file_path": "benchmark_data/WildSAText/wildsatext_groundtruth.txt",
     "model_name": "GroundTruth"},
    {"file_path": "benchmark_data/WildSAText/wildsatext_raw_dedup_candle.txt",
     "model_name": "CTC"},
    {"file_path": "benchmark_data/WildSAText/wildsatext_raw_dedup_candle_distilled.txt",
     "model_name": "CTC-Ditilled"},
]

# Each entry pairs an HF tokenizer ID with a display name for the table row.
TOKENIZERS = [
    {"hf_tokenizer": "inceptionai/jais-13b",        "tokenizer_name": "JAIS 13B"},
    {"hf_tokenizer": "google/gemma-4-31B-it",       "tokenizer_name": "Gemma4 it 31B"},
    {"hf_tokenizer": "meta-llama/Llama-3.3-70B-Instruct",       "tokenizer_name": "Llama 3.3 70B"},
    {"hf_tokenizer": "gpt2",                        "tokenizer_name": "GPT-2"},
    {"hf_tokenizer": "riotu-lab/Aranizer-SP-86k",   "tokenizer_name": "Aranizer-SP-86k"},
    {"hf_tokenizer": "ALLaM-AI/ALLaM-7B-Instruct-preview",   "tokenizer_name": "ALLaM 7B Instruct"},
    {"hf_tokenizer": "CohereForAI/c4ai-command-r-plus",   "tokenizer_name": "Command-R+"},
    {"hf_tokenizer": "Qwen/Qwen3.6-35B-A3B",   "tokenizer_name": "Qwen 3.6 35B"},
    # Add more as needed...
]

# Must match one of the model_name values above.
RAW_DATA_COLUMN = "Raw Data"

# CSV output path (set to None to skip saving).
CSV_OUTPUT = "fertility_results.csv"

BATCH_SIZE  = 1024
NUM_WORKERS = 30

# ── Helpers ───────────────────────────────────────────────────────────────────

def count_lines(path: str) -> int:
    result = subprocess.run(["wc", "-l", path], capture_output=True, text=True)
    return int(result.stdout.split()[0])


def process_batch(args: tuple[str, list[str]]) -> tuple[int, int]:
    """Runs in a worker process — each worker loads its own tokenizer instance."""
    from transformers import AutoTokenizer
    hf_tokenizer, batch = args
    tokenizer = AutoTokenizer.from_pretrained(hf_tokenizer, trust_remote_code=True)
    words  = sum(len(line.split()) for line in batch)
    tokens = sum(len(tokenizer.encode(line, add_special_tokens=False)) for line in batch)
    return words, tokens


def iter_batches(path: str, batch_size: int):
    """Yield non-empty stripped lines in chunks of batch_size."""
    with open(path, encoding="utf-8") as f:
        while True:
            batch = [line.strip() for line in islice(f, batch_size)]
            if not batch:
                break
            batch = [line for line in batch if line]
            if batch:
                yield batch


def compute_fertility(file_path: str, hf_tokenizer: str) -> float:
    """Return fertility score (tokens/word) for one file × tokenizer pair."""
    num_lines    = count_lines(file_path)
    total_words  = 0
    total_tokens = 0

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {
            executor.submit(process_batch, (hf_tokenizer, batch)): batch
            for batch in iter_batches(file_path, BATCH_SIZE)
        }
        with tqdm(total=num_lines, desc=f"  lines", unit="line", leave=False) as pbar:
            for future in as_completed(futures):
                words, tokens = future.result()
                total_words  += words
                total_tokens += tokens
                pbar.update(len(futures[future]))

    return total_tokens / total_words if total_words else -1.0 # return -1.0 to flag errors


# ── Main ──────────────────────────────────────────────────────────────────────

def pct_improvement(raw: float, model: float) -> float:
    """% improvement: positive means model is better (lower fertility) than raw."""
    return (raw - model) / raw * 100 if raw else 0.0


def format_cell(score: float, raw: float | None, is_raw_col: bool) -> str:
    """Format a terminal cell: score + optional % improvement."""
    if is_raw_col or raw is None:
        return f"{score:.4f}"
    pct = pct_improvement(raw, score)
    sign = "+" if pct >= 0 else ""
    return f"{score:.4f} ({sign}{pct:.2f}%)"


def print_table(results: dict[str, dict[str, float]]) -> None:
    """
    results[tokenizer_name][model_name] = fertility_score
    Rows = tokenizers, Columns = models.
    Non-Raw Data columns show score + % improvement vs Raw Data.
    """
    model_names     = [m["model_name"]     for m in MODELS]
    tokenizer_names = [t["tokenizer_name"] for t in TOKENIZERS]

    # Cell width must fit "1.2341 (+100.00%)" in the worst case
    cell_example_w = len("1.2341 (+100.00%)")
    col_w = max(cell_example_w, max(len(n) for n in model_names))
    row_label_w = max(len("Tokenizer \\ Model"), max(len(n) for n in tokenizer_names))

    sep = "-" * (row_label_w + 2 + (col_w + 3) * len(model_names))

    # After
    header_label = "Tokenizer \\ Model"
    header = f"{header_label:<{row_label_w}}  " + "  ".join(f"{n:>{col_w}}" for n in model_names)

    print()
    print(sep)
    print(header)
    print(sep)

    for tok_name in tokenizer_names:
        raw_score = results[tok_name].get(RAW_DATA_COLUMN)
        row = f"{tok_name:<{row_label_w}}  "
        cells = []
        for m_name in model_names:
            score      = results[tok_name][m_name]
            is_raw_col = (m_name == RAW_DATA_COLUMN)
            cell       = format_cell(score, raw_score, is_raw_col)
            cells.append(f"{cell:>{col_w}}")
        row += "  ".join(cells)
        print(row)

    print(sep)
    print(f"  % improvement = (Raw Data − Model) / Raw Data × 100  "
          f"(positive = lower fertility = better)")
    print(sep)


def save_csv(results: dict[str, dict[str, float]], path: str) -> None:
    """Save results to CSV with the same layout as the terminal table."""
    model_names     = [m["model_name"]     for m in MODELS]
    tokenizer_names = [t["tokenizer_name"] for t in TOKENIZERS]

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header row: blank corner + model names
        writer.writerow(["Tokenizer \\ Model"] + model_names)

        for tok_name in tokenizer_names:
            raw_score = results[tok_name].get(RAW_DATA_COLUMN)
            row = [tok_name]
            for m_name in model_names:
                score      = results[tok_name][m_name]
                is_raw_col = (m_name == RAW_DATA_COLUMN)
                row.append(format_cell(score, raw_score, is_raw_col))
            writer.writerow(row)

    print(f"\nCSV saved → {os.path.abspath(path)}")


if __name__ == "__main__":
    if RAW_DATA_COLUMN not in {m["model_name"] for m in MODELS}:
        raise ValueError(f"RAW_DATA_COLUMN '{RAW_DATA_COLUMN}' not found in MODELS.")

    total_pairs = len(MODELS) * len(TOKENIZERS)
    print(f"Computing fertility scores: {len(MODELS)} model(s) × {len(TOKENIZERS)} tokenizer(s) = {total_pairs} pair(s)\n")

    results: dict[str, dict[str, float]] = {t["tokenizer_name"]: {} for t in TOKENIZERS}

    pair_num = 0
    for model in MODELS:
        for tok in TOKENIZERS:
            pair_num += 1
            print(f"[{pair_num}/{total_pairs}] {model['model_name']} × {tok['tokenizer_name']}")
            score = compute_fertility(model["file_path"], tok["hf_tokenizer"])
            results[tok["tokenizer_name"]][model["model_name"]] = score
            print(f"        → fertility: {score:.4f}")

    print_table(results)

    if CSV_OUTPUT:
        save_csv(results, CSV_OUTPUT)

