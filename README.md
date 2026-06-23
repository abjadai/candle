# Candle — Arabic Character Deduplication Model

Candle is a Transformer-based model that corrects character elongation (*tamdeed*) in Arabic text. It takes noisy, user-generated input where characters are repeated (e.g. `الممملكككة`) and restores it to the correct form (e.g. `المملكة`). This is a common phenomenon in informal Arabic text on social media and in other user-generated content.

The model uses a CTC (Connectionist Temporal Classification) objective and is trained on Arabic newspaper text. A lighter student model is also provided, trained via knowledge distillation from the full teacher model.

---

## How It Works

Input text is first normalized by collapsing all character runs to exactly 2 repetitions (e.g. `الممملكككة` → `االلممللككةة`). This normalized form is fed into a Transformer encoder, and the CTC output is decoded greedily to produce the corrected text.

A word-level guard ensures that words with no elongation in the original input are never modified, preventing the model from accidentally altering already-clean words that appear alongside elongated ones in the same sequence.

---

## Repository Structure

```
candle/
├── candle_tokenizer.py       # Character-level tokenizer
├── candle_dataset.py         # Dataset and DataLoader with overlapping chunking
├── candle_pl.py              # Teacher model (CandleModel) — PyTorch Lightning
├── candle_pl_distill.py      # Student model (CandleModelDistill) with knowledge distillation
├── transformer.py            # Transformer Encoder implementation (with Flash Attention support)
├── xer.py                    # WER / CER metric utilities
├── train.py                  # Training script for the teacher model
├── train_distill.py          # Training script for the student model
├── predict_cli.py            # Inference script (supports both teacher and distilled models)
├── compute_xer_cli.py        # CLI tool for computing WER / CER / SER across benchmark files
├── compute_fertility.py      # Tokenizer fertility analysis across multiple models and tokenizers
├── run_predict_cli.sh        # Example shell script for running inference
└── run_eval.sh               # Example shell script for running evaluation
```

---

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Lightning
- tqdm
- kaldialign (for evaluation metrics)

```bash
pip install torch pytorch-lightning tqdm
pip install git+https://github.com/pzelasko/kaldialign.git
```

---

## Training

### Teacher Model

```bash
python train.py
```

The model is initialized from a pretrained CharBERT checkpoint ([CATT](https://github.com/abjadai/catt/releases/tag/v2)) and fine-tuned using a **phased unfreezing** strategy:

- **Phase 1** — Only the last two encoder layers and the linear head are trainable. Checkpoints saved to `models/candle_model_phase_1/`.
- **Phase 2** — Unfreeze one additional layer. Initialize from the best Phase 1 checkpoint. Checkpoints saved to `models/candle_model_phase_2/`.
- **Phase 3** — Unfreeze all layers. Initialize from the best Phase 2 checkpoint. Checkpoints saved to `models/candle_model_phase_3/`.

To switch phases, comment/uncomment the relevant `freeze`/`unfreeze` calls and `dirpath` in `train.py`.

**Key hyperparameters:**

| Parameter | Value | Description |
|---|---|---|
| `actual_max_seq_len` | 256 | Max sequence length during training |
| `model_max_seq_len` | 1024 | Max sequence length the model can handle at inference |
| `d_model` | 512 | Transformer hidden dimension |
| `n_layers` | 6 | Number of Transformer encoder layers |
| `n_heads` | 16 | Number of attention heads |
| `batch_size` | 256 | Training batch size |

**Expected data layout:**

```
train_data/
├── <your_train_file>.txt
└── <your_val_file>.txt
```

One Arabic sentence per line. Update the `train_text_file` and `val_text_file` paths in `train.py` accordingly.

Checkpoints are saved to the configured `dirpath`, monitored by `val_ser` (Sentence Error Rate).

---

### Student Model (Knowledge Distillation)

```bash
python train_distill.py
```

The student model has **2 encoder layers** instead of 6, and is trained with a combination of:

- **Hard loss** — CTC loss against ground-truth labels
- **Soft loss** — KL divergence against the teacher's output distributions, scaled by temperature²

Student weights are initialized by copying matching layers from the teacher: the teacher's first layer loads into the student's first layer, and the teacher's last layer loads into the student's second layer.

**Key distillation hyperparameters:**

| Parameter | Value | Description |
|---|---|---|
| `temperature` | 3.0 | Softens the teacher's probability distributions |
| `alpha` | 0.7 | Weight of the soft (distillation) loss; `1 - alpha` weights the hard (CTC) loss |

Update `ckpt_path` in `train_distill.py` to point to your best teacher checkpoint. Checkpoints are saved to `models/candle_model_distilled_v1/`.

---

## Inference

```bash
python predict_cli.py INPUT_FILE OUTPUT_FILE MODEL_PATH BATCH_SIZE IS_DISTILLED
```

| Argument | Description |
|---|---|
| `INPUT_FILE` | Path to input `.txt` file (one line per sample) |
| `OUTPUT_FILE` | Path to write deduplicated output |
| `MODEL_PATH` | Path to model checkpoint (`.ckpt`) |
| `BATCH_SIZE` | Number of lines to process per batch |
| `IS_DISTILLED` | `true` to load the 2-layer distilled model, `false` for the 6-layer teacher |

Example using the provided shell script (runs both teacher and distilled models on the same benchmarks):

```bash
bash run_predict_cli.sh
```

Lines longer than `max_seq_len` are automatically split on word boundaries and rejoined after inference, so the script handles arbitrarily long inputs gracefully.

---

## Evaluation

To compute WER, CER, and SER against a reference file:

```bash
python compute_xer_cli.py REF.txt HYP.txt
```

Results are reported separately for sentences that contain character duplications and those that do not, as well as overall. To run all benchmark evaluations at once:

```bash
bash run_eval.sh
```

---

## Tokenizer Fertility Analysis

`compute_fertility.py` measures the fertility (tokens per word) of various HuggingFace tokenizers across the raw, ground-truth, and model-deduplicated benchmark files. This is useful for quantifying how much character elongation increases token counts for downstream LLMs.

Configure the `MODELS` and `TOKENIZERS` lists at the top of the script, then run:

```bash
python compute_fertility.py
```

Results are printed as a formatted table and optionally saved to a CSV file.

---

## Validation Metrics

Both training scripts log the following metrics per epoch:

| Metric | Description |
|---|---|
| `val_loss` | CTC loss on the validation set |
| `val_ser` | Sentence Error Rate — fraction of sentences with any error |
| `val_wer` | Word Error Rate |
| `val_cer` | Character Error Rate |

Logs are written to TensorBoard under the configured `logs_path`.

---

## Data Format

Training and inference expect plain `.txt` files with one Arabic sentence per line. The preprocessing pipeline:

1. Strips tashkeel (diacritics) and tatweel (elongation marks)
2. Removes non-Arabic characters
3. Filters out lines shorter than `min_sentence_len` characters
4. Splits long lines into overlapping chunks with a configurable `stride_ratio` (default `0.5`)

---

## Acknowledgements

- Tokenizer adapted from [CATT (CharBERT)](https://github.com/abjadai/catt)
- Pretrained CharBERT weights from [CATT](https://github.com/abjadai/catt/releases/tag/v2)
- Transformer implementation based on [transformer-pytorch](https://github.com/gusdnd852/transformer-pytorch) by Hyunwoong Ko
- WER/CER metrics via [kaldialign](https://github.com/pzelasko/kaldialign)
