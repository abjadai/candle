
import os
import sys
import re
import math
import time
import torch
import subprocess
from tqdm import tqdm
from functools import cache
from candle_pl import CandleModel
from candle_tokenizer import CandleTokenizer

# Diacritics
FATHATAN = u'\u064b'
DAMMATAN = u'\u064c'
KASRATAN = u'\u064d'
FATHA = u'\u064e'
DAMMA = u'\u064f'
KASRA = u'\u0650'
SHADDA = u'\u0651'
SUKUN = u'\u0652'
TATWEEL = u'\u0640'

HARAKAT_PAT = re.compile(u"["+u"".join([FATHATAN, DAMMATAN, KASRATAN,
                                        FATHA, DAMMA, KASRA, SUKUN,
                                        SHADDA])+u"]")

def strip_tashkeel(text):
    text = HARAKAT_PAT.sub('', text)
    text = re.sub(u"[\u064E]", "", text,  flags=re.UNICODE) # fattha
    text = re.sub(u"[\u0671]", "", text,  flags=re.UNICODE) # waSla
    return text


def strip_tatweel(text):
    return re.sub(u'[%s]' % TATWEEL, '', text)


# remove non arabic chars + removing Tatweel + removing Tashkeel
def remove_non_arabic(text):
    text = strip_tashkeel(text)
    text = strip_tatweel(text)
    return ' '.join(re.sub(u"[^\u0621-\u063A\u0641-\u064A ]", " ", text,  flags=re.UNICODE).split())


def split_large_word(word, max_seq_len):
    max_subword_len = 10 # get a quarter of the seq len to be the max word size
    num_subwords = math.ceil(len(word) / max_subword_len)
    subwords = []
    for i in range(num_subwords):
        subwords.append(word[i*max_subword_len:(i+1)*max_subword_len])
    return subwords


def split_large_text(large_text, max_seq_len):
    words = large_text.split()
    split_lines = []
    new_split = ''
    is_long_word = False
    for word in words:
        if len(new_split + f' {word}')  < max_seq_len:
            new_split += f' {word}'
        else:
            split_lines.append(new_split)
            if len(word) > max_seq_len:
                is_long_word = True
                split_lines += split_large_word(word, max_seq_len)
                new_split = ''
            else:
                new_split = word
    split_lines.append(new_split)
    if is_long_word:
        assert ''.join(large_text.split()) == ''.join(''.join(split_lines).split())
    else:
        assert ' '.join(large_text.split()) == ' '.join(' '.join(split_lines).split())
    return split_lines


@cache
def get_num_lines(filename):
    command = f"wc -l {filename}"
    result = subprocess.check_output(command, shell=True, text=True)
    return int(result.split()[0])


def get_batches(X, batch_size=128):
    num_batches = math.ceil(len(X) / batch_size)
    for i in range(num_batches):
        yield X[i*batch_size:(i+1)*batch_size]


def process_file_with_deduplication_robust(f, out_f, model, batch_size, max_seq_len, split_large_text, num_lines, progress_callback=None):
    """
    Robust version with progress tracking and error handling.

    Args:
        f: Input file handle
        out_f: Output file handle
        model: CandleModel instance with deduplicate method
        batch_size: Number of lines to process in each batch
        max_seq_len: Maximum sequence length for the model
        split_large_text: Function to split large texts
        progress_callback: Optional function to call with progress updates
    """

    pbar = tqdm(total=num_lines)

    total_lines_processed = 0
    batch_number = 0
    try:
        while True:
            batch_number += 1
            batch_lines = []

            # Read batch with error handling
            for _ in range(batch_size):
                pbar.update(1)
                try:
                    line = f.readline()
                    if not line:
                        break
                    batch_lines.append(line.rstrip('\n\r'))
                except UnicodeDecodeError as e:
                    print(f"Warning: Skipping line due to encoding error: {e}")
                    continue

            if not batch_lines:
                break

            # Process batch
            texts_to_process = []
            line_mappings = []
            for i, line in enumerate(batch_lines):
                if len(line) <= max_seq_len:
                    texts_to_process.append(line)
                    line_mappings.append((i, 1))
                else:
                    try:
                        split_texts = split_large_text(line, max_seq_len)
                        texts_to_process.extend(split_texts)
                        line_mappings.append((i, len(split_texts)))
                    except Exception as e:
                        print(f"Warning: Error splitting line {total_lines_processed + i}: {e}")
                        # Fallback: truncate the line
                        texts_to_process.append(line[:max_seq_len])
                        line_mappings.append((i, 1))

            # Model inference with error handling
            try:
                if texts_to_process:
                    texts_to_process = [remove_non_arabic(t) for t in texts_to_process]
                    deduplicated_texts = []
                    for batch_texts in get_batches(texts_to_process, batch_size):
                        deduplicated_texts += model.deduplicate(batch_texts)

                    # Reconstruct results
                    result_index = 0
                    final_results = []

                    for original_line_idx, num_splits in line_mappings:
                        if num_splits == 1:
                            final_results.append(deduplicated_texts[result_index])
                            result_index += 1
                        else:
                            rejoined_text = ' '.join(deduplicated_texts[result_index:result_index + num_splits])
                            final_results.append(rejoined_text)
                            result_index += num_splits

                    # Write results
                    for result in final_results:
                        out_f.write(result + '\n')

                    out_f.flush()

            except Exception as e:
                print(f"Error processing batch {batch_number}: {e}")
                # Write original lines as fallback
                for line in batch_lines:
                    out_f.write(line + '\n')

            total_lines_processed += len(batch_lines)

            # Progress callback
            if progress_callback:
                progress_callback(batch_number, total_lines_processed)

    except Exception as e:
        print(f"Fatal error during processing: {e}")
        raise
    pbar.close()
    return total_lines_processed


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print('USAGE: python {} INPUT_FILE OUTPUT_FILE MODEL_PATH BATCH_SIZE IS_DISTILLED'.format(sys.argv[0]))
        sys.exit(1)


    f = sys.argv[1]
    out_f = sys.argv[2]
    ckpt_path = sys.argv[3]
    batch_size = int(sys.argv[4])
    is_distilled = sys.argv[5].lower().strip() == 'true'

    # IMPORTANT NOTE: keep the following list in the same order
    letters = [' ', 'ش', 'ؤ', 'ء', 'ذ', 'إ', 'أ', 'ا', 'ض', 'ع', 'ح', 'ص', 'ط', 'ى', 'ظ', 'ب', 'د', 'ف', 'غ', 'ه', 'ج', 'ك', 'ل', 'م', 'ن', 'ة', 'ق', 'ر', 'س', 'ت', 'ث', 'و', 'خ', 'ي', 'ز', 'آ', 'ئ']

    actual_max_seq_len = 256 # The actual length that the model will be trained on
    model_max_seq_len = 1024 # maximum length that the model can handle.

    tokenizer = CandleTokenizer(letters, max_seq_len=actual_max_seq_len)

    n_layers = 2 if is_distilled else 6
    blank_symbol = '_'
    model = CandleModel(tokenizer, blank_symbol, max_seq_len=model_max_seq_len, d_model=512, n_layers=n_layers, n_heads=16, drop_prob=0.1, learnable_pos_emb=False)

    print('ckpt_path:', ckpt_path)
    ckpt = torch.load(ckpt_path)
    state_dict = {}
    for k, v in ckpt['state_dict'].items():
        if not 'teacher' in k:
            state_dict[k] = v

    print('Loading models keys:', model.load_state_dict(state_dict))
    print(model.eval())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    print('model.training:', model.training)
    x = 'الممملكككة الأررررردنييية الللهههااااششميية'
    print(model.deduplicate([x])[0])

    x = 'الممملكككة الأررررردنييية ررراااننييياااا'
    print(model.deduplicate([x])[0])

    print('Device:', model.device)
    print('Sleeping for 10 secs...')
    time.sleep(10)

    num_lines = get_num_lines(f)
    max_seq_len = model.tokenizer.max_seq_len

    def progress_callback(batch_num, total_lines):
        if batch_num % 10 == 0:  # Print every 10 batches
            print(f"Processed batch {batch_num}, total lines: {total_lines}")

    f_path = f
    out_f_path = out_f
    with open(f_path, 'r') as f, open(out_f_path, 'w') as out_f:
        total_lines_processed = process_file_with_deduplication_robust(
                                    f=f,
                                    out_f=out_f,
                                    model=model,
                                    batch_size=batch_size,
                                    max_seq_len=max_seq_len,
                                    split_large_text=split_large_text,
                                    num_lines=num_lines
                                )
    print('total_lines_processed:', total_lines_processed)
