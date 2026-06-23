
import xer
import sys
import subprocess
import re
import os
from functools import cache
from tqdm import tqdm
import pandas as pd
from multiprocessing import Pool, cpu_count
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools

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




f = 'benchmark_data/NewsText/fixed_typos_dict.txt'

typos_dict = {}
with open(f) as f1:
    for line in f1:
        typo, correct = line.strip().split(' : ')
        typos_dict[typo] = correct


def has_char_duplication(text):
    pattern = r'(.)\1{1,}'
    return bool(re.search(pattern, text))

# Keep your original function
@cache
def get_num_lines(filename):
    command = f"wc -l {filename}"
    result = subprocess.check_output(command, shell=True, text=True)
    return int(result.split()[0])

def process_chunk(chunk_data):
    """Process a chunk of line pairs and return aggregated results"""

    ref_lines, hyp_lines = chunk_data
    computed_samples = []
    for ref_text, hyp_text in zip(ref_lines, hyp_lines):
        # Same processing as original
        ref_text, hyp_text = str(ref_text).strip(), str(hyp_text).strip()
        ref_text, hyp_text = remove_non_arabic(ref_text), remove_non_arabic(hyp_text)

        for typo, correct in typos_dict.items():
            ref_text = ref_text.replace(typo, correct)

        wer_err = xer.wer(ref_text, hyp_text)
        cer_err = xer.cer(ref_text, hyp_text)

        computed_sample = {}
        computed_sample['ref_text'] = ref_text
        computed_sample['hyp_text'] = hyp_text
        computed_sample['ser_ref_len'] = 1
        computed_sample['ser_distance'] = int(ref_text != hyp_text)
        computed_sample['wer_ref_len'] = wer_err['ref_length']
        computed_sample['wer_distance'] = wer_err['distance']
        computed_sample['cer_ref_len'] = cer_err['ref_length']
        computed_sample['cer_distance'] = cer_err['distance']

        computed_samples.append(computed_sample)

    return computed_samples


def read_file_in_chunks(filename, chunk_size=10000):
    """Generator to read file in chunks"""
    with open(filename, 'r', buffering=8192*4) as f:  # Larger buffer
        while True:
            chunk = list(itertools.islice(f, chunk_size))
            if not chunk:
                break
            yield chunk


def compute_metrics(computed_samples):

    # Initialize global counters
    total_ser_ref_len = 0
    total_ser_distance = 0
    total_wer_ref_len = 0
    total_wer_distance = 0
    total_cer_ref_len = 0
    total_cer_distance = 0

    for computed_sample in computed_samples:
        # Aggregate results
        total_ser_ref_len += computed_sample['ser_ref_len']
        total_ser_distance += computed_sample['ser_distance']
        total_wer_ref_len += computed_sample['wer_ref_len']
        total_wer_distance += computed_sample['wer_distance']
        total_cer_ref_len += computed_sample['cer_ref_len']
        total_cer_distance += computed_sample['cer_distance']

    result = {}
    if len(computed_samples):
        result['SER'] = total_ser_distance / total_ser_ref_len
        result['WER'] = total_wer_distance / total_wer_ref_len
        result['CER'] = total_cer_distance / total_cer_ref_len
    else:
        result['SER'] = -1
        result['WER'] = -1
        result['CER'] = -1

    return result


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print('USAGE: python {} REF.txt HYP.txt'.format(sys.argv[0]))
        sys.exit(1)

    ref_file = sys.argv[1]
    hyp_file = sys.argv[2]

    # Configuration
    chunk_size = 2000  # Adjust based on your memory and file size
    max_workers = min(cpu_count(), 100)  # Don't use too many processes

    print(f"Using {max_workers} processes with chunk size {chunk_size}")

    # Get total lines for progress bar
    ref_num_lines = get_num_lines(ref_file)
    hyp_num_lines = get_num_lines(hyp_file)
    assert ref_num_lines == hyp_num_lines, f'ref_num_lines: {ref_num_lines}, hyp_num_lines: {hyp_num_lines}'

    # Create chunk generators
    ref_chunks = read_file_in_chunks(ref_file, chunk_size)
    hyp_chunks = read_file_in_chunks(hyp_file, chunk_size)

    all_computed_samples = []
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Create progress bar with faster updates
        pbar = tqdm(total=ref_num_lines, desc="Processing lines", smoothing=0.1, mininterval=0.1)

        # Submit all chunks first, then process results as they complete
        future_to_chunk = {}
        for ref_chunk, hyp_chunk in zip(ref_chunks, hyp_chunks):
            chunk_data = (ref_chunk, hyp_chunk)
            future = executor.submit(process_chunk, chunk_data)
            future_to_chunk[future] = len(ref_chunk)

        # Use as_completed for immediate updates when any chunk finishes
        for future in as_completed(future_to_chunk):
            chunk_len = future_to_chunk[future]
            computed_samples = future.result()
            all_computed_samples += computed_samples

            # Update progress bar immediately when chunk completes
            pbar.update(chunk_len)

        pbar.close()

    with_duplicate_chars_samples = []
    without_duplicate_chars_samples = []
    for computed_sample in all_computed_samples:
        if has_char_duplication(computed_sample['ref_text']):
            with_duplicate_chars_samples.append(computed_sample)
        else:
            without_duplicate_chars_samples.append(computed_sample)

    print('-'*100)
    print(f'python {sys.argv[0]} {sys.argv[1]} {sys.argv[2]}')
    print('-'*50)
    print(f'Samples WITH Duplicate Chars Result ({len(with_duplicate_chars_samples)} samples):')
    print(compute_metrics(with_duplicate_chars_samples))
    print('-'*90)

    print(f'Samples WITHOUT Duplicate Chars Result ({len(without_duplicate_chars_samples)} samples):')
    print(compute_metrics(without_duplicate_chars_samples))
    print('-'*90)

    print(f'Overall Result ({len(all_computed_samples)} samples):')
    print(compute_metrics(all_computed_samples))
