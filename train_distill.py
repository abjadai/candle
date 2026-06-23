
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from candle_tokenizer import CandleTokenizer
from candle_dataset import CandleDataset, PrePaddingDataLoader
from candle_pl import CandleModel
from candle_pl_distill import CandleModelDistill

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

    ckpt_path = 'models/candle_model_phase_3/candle_model-epoch=04-val_loss=84.50557-val_ser=0.09816-val_wer=0.00318-val_cer=0.00066.ckpt'

    print('Loading Teacher Model:')
    print(ckpt_path)
    # 1. Load the pre-trained teacher model
    teacher_model = CandleModel.load_from_checkpoint(
        ckpt_path,
        tokenizer=tokenizer,
        blank_symbol=blank_symbol,
        max_seq_len=model_max_seq_len,
        d_model=512,
        n_layers=6,
        n_heads=16,
        drop_prob=0.1,
        learnable_pos_emb=False
    )

    print('Teacher model successfully loaded!')

    print('Initializing Student Model')
    # 2. Create student model with distillation
    student_model = CandleModelDistill(
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        blank_symbol=blank_symbol,
        max_seq_len=model_max_seq_len,
        d_model=512,
        n_layers=2,  # Smaller student
        n_heads=16,
        drop_prob=0.1,
        learnable_pos_emb=False,
        temperature=3.0,  # Distillation temperature
        alpha=0.7  # Weight for soft targets (0.7) vs hard targets (0.3)
    )

    print('Load as much as you can from the Teacher\'s weights into the Student model...')
    print(student_model.load_state_dict(teacher_model.state_dict(), strict=False))

    print('Load the weights of some special layers from the teacher in the student...')
    print(student_model.encoder.layers[0].load_state_dict(teacher_model.encoder.layers[0].state_dict()))
    print(student_model.encoder.layers[1].load_state_dict(teacher_model.encoder.layers[-1].state_dict()))
    print(student_model.linear.load_state_dict(teacher_model.linear.state_dict()))

    print('Weights successfully loaded!')

    teacher_model.eval()

    dirpath = 'models/candle_model_distilled_v1/'

    checkpoint_callback = ModelCheckpoint(dirpath=dirpath, save_top_k=10, save_last=True,
                                          monitor='val_ser',
                                          filename='candle_model_distilled-{epoch:02d}-{val_loss:.5f}-{val_ser:.5f}-{val_wer:.5f}-{val_cer:.5f}')

    print('Creating Trainer...')

    logs_path = f'{dirpath}/logs'

    print('#'*100)
    print(student_model)
    print('#'*100)

    trainer = Trainer(
        accelerator="gpu",
        devices=-1,
        max_epochs=300,
        callbacks=[TQDMProgressBar(refresh_rate=1), checkpoint_callback],
        logger=TensorBoardLogger(logs_path, name='candle_model_distilled_v1'),
        strategy="ddp_find_unused_parameters_false"
        )

    trainer.fit(student_model, train_dataloader, val_dataloader)
    print(f'Save model to: {dirpath}/last_saved_after_keyboard_interruption.ckpt')
    trainer.save_checkpoint(f'{dirpath}/last_saved_after_keyboard_interruption.ckpt')
    print('DONE!!!!')
