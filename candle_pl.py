
from transformer import Encoder, make_pad_mask
from candle_dataset import pad_seq, reduce_to_2_chars
from itertools import zip_longest
import torch.nn as nn
import torch
import pytorch_lightning as pl
import xer
import re

def has_consecutive_chars(text, n):
    return bool(re.search(r'(.)\1{' + str(n-1) + r'}', text))


class CandleModel(pl.LightningModule):

    def __init__(self, tokenizer, blank_symbol='_', max_seq_len=2048, d_model=512, n_layers=6, n_heads=16, drop_prob=0.1, learnable_pos_emb=True):
        super(CandleModel, self).__init__()
        self.tokenizer = tokenizer
        ffn_hidden = 4 * d_model
        self.blank_symbol = blank_symbol
        self.vocab_list = list(self.tokenizer.letters)
        self.vocab_list[self.vocab_list.index('<MASK>')] = blank_symbol # Replace the mask token with the blank token
        enc_voc_size = len(self.vocab_list) # 37 + pad token + bos token + eos token + blank symbol = 41
        self.pad_token_id = tokenizer.pad_token_id
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_heads,
                               max_len=max_seq_len,
                               ffn_hidden=ffn_hidden,
                               enc_voc_size=enc_voc_size,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               padding_idx=self.pad_token_id,
                               learnable_pos_emb=learnable_pos_emb)

        self.linear = nn.Linear(d_model, enc_voc_size, bias=False)
        self.criterion = nn.CTCLoss(blank=self.vocab_list.index(self.blank_symbol), reduction='sum', zero_infinity=True)

    def forward(self, x):
        x_mask = make_pad_mask(x, self.pad_token_id)
        y_pred = self.encoder(x, x_mask)
        y_pred = self.linear(y_pred)
        return y_pred

    def greedy_decode(self, input_ids_list):
        texts = []
        for input_ids in input_ids_list:
            ids = []
            prev_char = None
            for char_id in input_ids:
                if prev_char is None:
                    prev_char = char_id
                    ids.append(self.vocab_list[char_id])
                elif prev_char != char_id:
                    prev_char = char_id
                    ids.append(self.vocab_list[char_id])
            text = "".join(ids).replace(self.blank_symbol, '')
            texts.append(text.strip())
        return texts

    @torch.no_grad()
    def deduplicate(self, texts):
        # Separate texts that need deduplication (have consecutive chars) from those that don't.
        # We avoid running clean texts through the model unnecessarily.
        texts_needing_dedup = []   # list of (original_index, text)
        texts_already_clean = []   # list of (original_index, text)

        for idx, text in enumerate(texts):
            if has_consecutive_chars(text, 2):
                texts_needing_dedup.append((idx, text))
            else:
                texts_already_clean.append((idx, text))

        # Early return: if no text needs deduplication, skip the model entirely.
        # Without this guard, pad_seq crashes on an empty list of sequences.
        if not texts_needing_dedup:
            _, ordered_texts = zip(*texts_already_clean)
            return ordered_texts

        # Encode each text: collapse all runs to exactly 2 consecutive chars, then tokenize.
        # bos=False, eos=False because the CTC model doesn't use sequence boundary tokens.
        token_id_sequences = []
        for _, text in texts_needing_dedup:
            normalized_text = reduce_to_2_chars(text)
            token_ids = self.tokenizer.tokenize(normalized_text, bos=False, eos=False)
            token_id_sequences.append(
                torch.tensor(token_ids, device=self.device).long()
            )

        # Pad sequences to the same length so they form a batch tensor (N x T).
        seq_lengths = [seq.shape[0] for seq in token_id_sequences]
        padded_input = pad_seq(token_id_sequences, batch_first=True,
                               padding_value=self.tokenizer.pad_token_id,
                               prepadding=False)

        # Forward pass through the model → raw logits (N x T x vocab_size).
        # Greedy argmax gives the most likely token at each timestep.
        logits = self(padded_input)
        predicted_token_ids = logits.argmax(-1)  # N x T

        # Trim padding from each prediction before CTC decoding.
        trimmed_predictions = [
            predicted_token_ids[i].tolist()[:seq_lengths[i]]
            for i in range(len(seq_lengths))
        ]

        # CTC greedy decode: collapse repeated tokens and remove blanks.
        decoded_texts = self.greedy_decode(trimmed_predictions)

        # Word-level guard: if a word had no consecutive chars in the *original* text,
        # keep the original word regardless of what the model predicted.
        # This prevents the model from accidentally altering already-clean words
        # that happen to appear in the same sequence as elongated ones.
        # zip_longest is used instead of zip to avoid silently dropping words when
        # the decoded text has a different word count than the original (e.g. if
        # the CTC decoder merges tokens across a space boundary).
        corrected_texts = []
        for (original_idx, original_text), decoded_text in zip(texts_needing_dedup, decoded_texts):
            corrected_words = []
            for original_word, decoded_word in zip_longest(original_text.split(), decoded_text.split()):
                if original_word is None:
                    # decoded has extra words — shouldn't happen, but skip them
                    continue
                if decoded_word is None or not has_consecutive_chars(original_word, 2):
                    corrected_words.append(original_word)  # keep original clean word
                else:
                    corrected_words.append(decoded_word)   # use model's correction
            corrected_texts.append((original_idx, ' '.join(corrected_words)))

        # Merge deduped results with the untouched clean texts,
        # then sort by original index to restore the input order.
        all_results = corrected_texts + texts_already_clean
        all_results.sort(key=lambda pair: pair[0])

        _, ordered_texts = zip(*all_results)
        return ordered_texts

    def training_step(self, batch, batch_idx):
        x, x_lens, y, y_lens = batch
        x = self(x)
        x = x.transpose(0, 1)  # NxTxH --> TxNxH
        x = x.float().log_softmax(2)  # ensure float32 for loss and applying log_softmax since Pytorch CTCloss accepts log propabilites as inputs
        loss = self.criterion(x, y, x_lens, y_lens)

        self.log('train_loss', loss, prog_bar=True)
        sch = self.lr_schedulers()
        if not sch is None:
            sch.step()
            self.log('lr', sch.get_last_lr()[0], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, x_lens, y, y_lens = batch
        y_pred = self(x)
        y_pred = y_pred.transpose(0, 1)  # NxTxH --> TxNxH
        y_pred = y_pred.float().log_softmax(2)  # ensure float32 for loss and applying log_softmax since Pytorch CTCloss accepts log propabilites as inputs
        loss = self.criterion(y_pred, y, x_lens, y_lens)
        y_pred = y_pred.permute(1, 0, 2) # TxNxH --> NxTxH
        hyps = self.greedy_decode(y_pred.argmax(-1).tolist())
        refs = []
        for i in range(len(y)):
            ref = ''.join([self.vocab_list[idx] for idx in y[i][:y_lens[i]]])
            refs.append(ref)

        total_val_ser_distance = 0
        total_val_ser_ref_length = 0
        total_val_wer_distance = 0
        total_val_wer_ref_length = 0
        total_val_cer_distance = 0
        total_val_cer_ref_length = 0

        for i in range(len(y)):
            hyp = hyps[i]
            ref = refs[i]
            val_wer = xer.wer(ref, hyp)
            total_val_wer_distance += val_wer['distance']
            total_val_wer_ref_length += val_wer['ref_length']
            val_cer = xer.cer(ref, hyp)
            total_val_cer_distance += val_cer['distance']
            total_val_cer_ref_length += val_cer['ref_length']
            total_val_ser_distance += int(ref.strip() != hyp.strip())
            total_val_ser_ref_length += 1

        total_ser_error = total_val_ser_distance / total_val_ser_ref_length
        total_wer_error = total_val_wer_distance / total_val_wer_ref_length
        total_cer_error = total_val_cer_distance / total_val_cer_ref_length

        self.log('val_loss', loss, sync_dist=True)
        self.log('val_ser', torch.FloatTensor([total_ser_error]).to(self.device), sync_dist=True)
        self.log('val_ser_distance', torch.FloatTensor([total_val_ser_distance]).to(self.device), sync_dist=True)
        self.log('val_ser_ref_length', torch.FloatTensor([total_val_ser_ref_length]).to(self.device), sync_dist=True)
        self.log('val_wer', torch.FloatTensor([total_wer_error]).to(self.device), sync_dist=True)
        self.log('val_wer_distance', torch.FloatTensor([total_val_wer_distance]).to(self.device), sync_dist=True)
        self.log('val_wer_ref_length', torch.FloatTensor([total_val_wer_ref_length]).to(self.device), sync_dist=True)
        self.log('val_cer', torch.FloatTensor([total_cer_error]).to(self.device), sync_dist=True)
        self.log('val_cer_distance', torch.FloatTensor([total_val_cer_distance]).to(self.device), sync_dist=True)
        self.log('val_cer_ref_length', torch.FloatTensor([total_val_cer_ref_length]).to(self.device), sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        opts = {"optimizer": optimizer}
        return opts
