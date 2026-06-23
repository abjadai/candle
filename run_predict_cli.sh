#!/bin/bash

# RUN 6-layer Candle CTC
ckpt_path='models/candle_model_phase_3/candle_model-epoch=04-val_loss=84.50557-val_ser=0.09816-val_wer=0.00318-val_cer=0.00066.ckpt'
is_distilled=false

input_file='benchmark_data/NewsText/newstext.txt'
output_file='benchmark_data/NewsText/newstext_dedup_candle.txt'
batch_size=1024
python predict_cli.py $input_file $output_file $ckpt_path $batch_size $is_distilled

input_file='benchmark_data/AmbigText/ambigtext.txt'
output_file='benchmark_data/AmbigText/ambigtext_dedup_candle.txt'
batch_size=16
python predict_cli.py $input_file $output_file $ckpt_path $batch_size $is_distilled

input_file='benchmark_data/WildSAText/wildsatext_raw.txt'
output_file='benchmark_data/WildSAText/wildsatext_raw_dedup_candle.txt'
batch_size=1024
python predict_cli.py $input_file $output_file $ckpt_path $batch_size $is_distilled


# RUN 2-layer distilled Candle CTC
ckpt_path='models/candle_model_distilled_v1/candle_model_distilled-epoch=43-val_loss=29.99644-val_ser=0.11001-val_wer=0.00363-val_cer=0.00075.ckpt'
is_distilled=true

input_file='benchmark_data/NewsText/newstext.txt'
output_file='benchmark_data/NewsText/newstext_dedup_candle_distilled.txt'
batch_size=1024
python predict_cli.py $input_file $output_file $ckpt_path $batch_size $is_distilled

input_file='benchmark_data/AmbigText/ambigtext.txt'
output_file='benchmark_data/AmbigText/ambigtext_dedup_candle_distilled.txt'
batch_size=16
python predict_cli.py $input_file $output_file $ckpt_path $batch_size $is_distilled

input_file='benchmark_data/WildSAText/wildsatext_raw.txt'
output_file='benchmark_data/WildSAText/wildsatext_raw_dedup_candle_distilled.txt'
batch_size=1024
python predict_cli.py $input_file $output_file $ckpt_path $batch_size $is_distilled
