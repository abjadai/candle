
# Adapted from CharBERTTokenizer (originally char_bert_tokenizer.py)
# From: https://github.com/abjadai/catt


class CandleTokenizer:

    def __init__(self, letters, max_seq_len=2048,
                       mask_token='<MASK>', pad_token='<PAD>',
                       bos_token='<BOS>', eos_token='<EOS>'):
        self.max_seq_len = max_seq_len
        self.mask_token = mask_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.letters = [self.pad_token, self.bos_token, self.eos_token] + list(letters) + [self.mask_token]
        self.letters_map = {c:i for i,c in enumerate(self.letters)}
        self.mask_token_id = self.letters_map[self.mask_token]
        self.pad_token_id = self.letters_map[self.pad_token]
        self.bos_token_id = self.letters_map[self.bos_token]
        self.eos_token_id = self.letters_map[self.eos_token]
        self.special_tokens_ids = [self.pad_token_id, self.bos_token_id, self.eos_token_id, self.mask_token_id]


    def tokenize(self, text, bos=True, eos=True):
        token_ids = [self.letters_map[c] for c in text.strip()]
        if bos:
            token_ids = [self.bos_token_id] + token_ids
        if eos:
            token_ids = token_ids + [self.eos_token_id]
        return token_ids

    def detokenize(self, token_ids):
        text = ''.join([self.letters[i] for i in token_ids])
        text = text.replace(self.bos_token, ' ').replace(self.eos_token, ' ').replace(self.pad_token, ' ')
        text = ' '.join(text.strip().split())
        return text
