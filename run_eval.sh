#!/bin/bash

########################### 6-layer Candle CTC ###############################
python compute_xer_cli.py \
    benchmark_data/NewsText/newstext.txt \
    benchmark_data/NewsText/newstext_dedup_candle.txt

python compute_xer_cli.py \
    benchmark_data/AmbigText/ambigtext.txt \
    benchmark_data/AmbigText/ambigtext_dedup_candle.txt

python compute_xer_cli.py \
    benchmark_data/WildSAText/wildsatext_groundtruth.txt \
    benchmark_data/WildSAText/wildsatext_raw_dedup_candle.txt


########################### 2-layer distilled Candle CTC ###############################
python compute_xer_cli.py \
    benchmark_data/NewsText/newstext.txt \
    benchmark_data/NewsText/newstext_dedup_candle_distilled.txt

python compute_xer_cli.py \
    benchmark_data/AmbigText/ambigtext.txt \
    benchmark_data/AmbigText/ambigtext_dedup_candle_distilled.txt

python compute_xer_cli.py \
    benchmark_data/WildSAText/wildsatext_groundtruth.txt \
    benchmark_data/WildSAText/wildsatext_raw_dedup_candle_distilled.txt
