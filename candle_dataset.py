
import os
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import re
from tqdm import tqdm

def repeat_characters(string):
    repeated_string = re.sub(r'(.)', r'\1\1', string)
    return repeated_string.replace('  ', ' ')

def reduce_to_1_char(text):
    p_longation = re.compile(r'(.)\1+')
    subst = r"\1"
    text = re.sub(p_longation, subst, text)
    return text

def reduce_to_2_chars(text):
    return repeat_characters(reduce_to_1_char(text))

# sequences is a list of tensors of shape TxH where T is the seqlen and H is the feats dim
def pad_seq(sequences, batch_first=True, padding_value=0.0, prepadding=True):
    lens = [i.shape[0] for i in sequences]
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=padding_value) # NxTxH
    if prepadding:
        for i in range(len(lens)):
            padded_sequences[i] = padded_sequences[i].roll(-lens[i])
    if not batch_first:
        padded_sequences = padded_sequences.transpose(0, 1) # TxNxH
    return padded_sequences


class PrePaddingDataLoader(DataLoader):
    def __init__(self, pad_token_id, *args, **kwargs):
        super(PrePaddingDataLoader, self).__init__(*args, **kwargs)
        self.pad_token_id = pad_token_id
        self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        input_ids_list = []
        target_ids_list = []
        input_lens = []
        target_lens = []
        for input_ids, target_ids in batch:
            input_lens.append(len(input_ids))
            target_lens.append(len(target_ids))
            input_ids_list.append(input_ids)
            target_ids_list.append(target_ids)
        batch_input_ids = pad_seq(input_ids_list, batch_first=True, padding_value=self.pad_token_id, prepadding=False)
        batch_target_ids = pad_seq(target_ids_list, batch_first=True, padding_value=self.pad_token_id, prepadding=False)
        batch_input_lens, batch_target_lens = torch.tensor(input_lens).long(), torch.tensor(target_lens).long()
        return batch_input_ids, batch_input_lens, batch_target_ids, batch_target_lens


class CandleDataset(Dataset):

    def __init__(self, txt_file_path, tokenizer, max_sentence_len=256, min_sentence_len=30, stride_ratio=0.5):

        self.tokenizer = tokenizer
        self.txt_file_path = txt_file_path
        self.stride_ratio = stride_ratio
        self.max_sentence_len = max_sentence_len
        self.min_sentence_len = min_sentence_len

        # Load all lines
        with open(self.txt_file_path) as f1:
            self.lines = [line.strip() for line in tqdm(f1, total=self._line_count(self.txt_file_path), desc='Loading file')]

        # Build index of chunks: list of (line_idx, chunk_start, chunk_end)
        self.chunks = self._build_chunks()

    def _line_count(self, file_path):
        return int(os.popen(f'wc -l {file_path}').read().split()[0])

    def _build_chunks(self):
        """Split long lines into overlapping chunks."""
        chunks = []
        max_len = self.max_sentence_len - 4
        stride = int(max_len * self.stride_ratio)

        for line_idx, line in enumerate(tqdm(self.lines, desc='Creating chunks')):
            if len(line) == 0:
                continue

            # Filter out sentences shorter than min_sentence_len
            if len(line) < self.min_sentence_len:
                continue

            if len(line) <= max_len:
                # Short line - one chunk covering the entire line
                chunks.append((line_idx, 0, len(line)))
            else:
                # Long line - split into overlapping chunks
                chunk_start = 0
                while chunk_start < len(line):
                    chunk_end = min(chunk_start + max_len, len(line))

                    # Only add chunk if it meets minimum length requirement
                    chunk_len = chunk_end - chunk_start
                    if chunk_len >= self.min_sentence_len:
                        chunks.append((line_idx, chunk_start, chunk_end))

                    if chunk_end >= len(line):
                        break

                    # Move by stride
                    chunk_start += stride

                    # If the remaining tail is shorter than `stride`, we snap back to capture
                    # the final `max_len` characters. This means the last two chunks of a long
                    # line will always overlap by more than stride_ratio.
                    if len(line) - chunk_start < stride:
                        chunk_start = len(line) - max_len

        return chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, index):
        line_idx, chunk_start, chunk_end = self.chunks[index]
        line = self.lines[line_idx]

        # Extract the chunk
        sample = line[chunk_start:chunk_end]

        # Adjust chunk boundaries to word boundaries for better context
        # Check if we're cutting in the middle of a word at the START
        if chunk_start > 0 and line[chunk_start - 1] != ' ':
            first_space = sample.find(' ')
            if first_space > 0:
                sample = sample[first_space + 1:]

        # Check if we're cutting in the middle of a word at the END
        if chunk_end < len(line) and line[chunk_end] != ' ':
            last_space = sample.rfind(' ')
            if last_space > 0:
                sample = sample[:last_space]

        sample = sample.strip()

        # Handle empty samples (edge case)
        if not sample:
            return self.__getitem__((index + 1) % len(self))

        input_text = reduce_to_2_chars(sample)
        input_ids = self.tokenizer.tokenize(input_text, bos=False, eos=False)
        target_ids = self.tokenizer.tokenize(sample, bos=False, eos=False)
        input_ids, target_ids = torch.tensor(input_ids).long(), torch.tensor(target_ids).long()
        return input_ids, target_ids


if __name__ == '__main__':
    from candle_tokenizer import CandleTokenizer
    letters = [' ', 'ش', 'ؤ', 'ء', 'ذ', 'إ', 'أ', 'ا', 'ض', 'ع', 'ح', 'ص', 'ط', 'ى', 'ظ', 'ب', 'د', 'ف', 'غ', 'ه', 'ج', 'ك', 'ل', 'م', 'ن', 'ة', 'ق', 'ر', 'س', 'ت', 'ث', 'و', 'خ', 'ي', 'ز', 'آ', 'ئ']

    tokenizer = CandleTokenizer(letters)
    text_file = 'file.txt'
    dataset = CandleDataset(text_file, tokenizer)
    print(dataset[0])
    dataloader = PrePaddingDataLoader(tokenizer.pad_token_id, dataset, batch_size=4)
    a = next(iter(dataloader))
    print(a)
