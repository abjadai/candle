
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from candle_tokenizer import CandleTokenizer
from candle_dataset import CandleDataset, PrePaddingDataLoader
from candle_pl import CandleModel


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def unfreeze(model):
    for param in model.parameters():
        param.requires_grad = True


if __name__ == '__main__':
    # IMPORTANT NOTE: keep the following list in the same order
    letters = [' ', 'ش', 'ؤ', 'ء', 'ذ', 'إ', 'أ', 'ا', 'ض', 'ع', 'ح', 'ص', 'ط', 'ى', 'ظ', 'ب', 'د', 'ف', 'غ', 'ه', 'ج', 'ك', 'ل', 'م', 'ن', 'ة', 'ق', 'ر', 'س', 'ت', 'ث', 'و', 'خ', 'ي', 'ز', 'آ', 'ئ']

    actual_max_seq_len = 256 # The actual length that the model will be trained on
    model_max_seq_len = 1024 # maximum length that the model can handle.
    tokenizer = CandleTokenizer(letters, max_seq_len=actual_max_seq_len)

    train_text_file = 'train_data/Ryiadh_n_SaudiYoum_utf_8_CLEAN_n_UNIFIED_train.txt'
    val_text_file = 'train_data/Ryiadh_n_SaudiYoum_utf_8_CLEAN_n_UNIFIED_val.txt'

    batch_size = 256
    num_workers = 5

    max_sentence_len = tokenizer.max_seq_len
    min_sentence_len = (6 + 1)*3 # minimum 3 words with average word len is 6 and 1 white space after each word

    train_dataset = CandleDataset(train_text_file, tokenizer, max_sentence_len=max_sentence_len, min_sentence_len=min_sentence_len)
    train_dataloader = PrePaddingDataLoader(tokenizer.pad_token_id, train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    val_dataset = CandleDataset(val_text_file, tokenizer, max_sentence_len=max_sentence_len, min_sentence_len=min_sentence_len)
    val_dataloader = PrePaddingDataLoader(tokenizer.pad_token_id, val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    blank_symbol = '_'
    model = CandleModel(tokenizer, blank_symbol, max_seq_len=model_max_seq_len, d_model=512, n_layers=6, n_heads=16, drop_prob=0.1, learnable_pos_emb=False)

#    unfreeze(model)                     # Phase 3
    freeze(model)
#    unfreeze(model.encoder.layers[-3])  # Phase 2
    unfreeze(model.encoder.layers[-2])  # Phase 1
    unfreeze(model.encoder.layers[-1])  # Phase 1
    unfreeze(model.linear)              # Phase 1

    dirpath = 'models/candle_model_phase_1/' # initialized from MLM, all frozen except last two layers
#    dirpath = 'models/candle_model_phase_2/' # initialized from best ckpt in phase 1, all frozen except last three layers
#    dirpath = 'models/candle_model_phase_3/' # initialized from best ckpt in phase 2, all model unfrozen

    checkpoint_callback = ModelCheckpoint(dirpath=dirpath, save_top_k=10, save_last=True,
                                          monitor='val_ser',
                                          filename='candle_model-{epoch:02d}-{val_loss:.5f}-{val_ser:.5f}-{val_wer:.5f}-{val_cer:.5f}')

    print('Creating Trainer...')

    logs_path = f'{dirpath}/logs'

    print('#'*100)
    print(model)
    print('#'*100)

    # CharBERT checkpoint taken from CATT: https://github.com/abjadai/catt/releases/tag/v2
    ckpt_path = 'char_bert_model_pretrained.pt'
    print('-'*89)
    print('-'*89)
    print('Loading models keys:', model.load_state_dict(torch.load(ckpt_path)))
    print('-'*89)
    print('-'*89)

    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        max_epochs=300,
        callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint_callback],
        logger=TensorBoardLogger(logs_path),
        strategy="ddp_find_unused_parameters_false"
        )

    trainer.fit(model, train_dataloader, val_dataloader)
    print(f'Save model to: {dirpath}/last_saved_after_keyboard_interruption.ckpt')
    trainer.save_checkpoint(f'{dirpath}/last_saved_after_keyboard_interruption.ckpt')
    print('DONE!!!!')
