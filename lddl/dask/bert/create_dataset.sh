#!/bin/bash

# Paths for raw non-shuffled and non-split datasets
BOOKS_PATH="${BOOKS_PATH-/workspace/bert/data/bookcorpus/source/}"
WIKI_PATH="${WIKI_PATH-/workspace/bert/data/wikipedia_dataset/source/en}"
C4_PATH="${C4_PATH-/workspace/bert/data/c4/source/}"

# Whether to create raw train and test datasets. Set to false if they already exist.
CREATE_BASE_SHARDS="${CREATE_BASE_SHARDS:-true}"

# Proportions of how to split data between train and test
N_TRAIN_SHARDS="${N_TRAIN_SHARDS:-256}"
N_TEST_SHARDS="${N_TEST_SHARDS:-16}"

# Directories for writing formatted and processed, but still not properly shuffled
# and balanced datasets.
OUTPUT_TRAIN_DIR="${OUTPUT_TRAIN_DIR:-$(pwd)/output_train}"
OUTPUT_TEST_DIR="${OUTPUT_TEST_DIR:-$(pwd)/output_test}"

# Number of duplicating runs reusing the same raw datasets for calculating LLM
# training input data. Set to 0 if you don't want to process a dataset at all.
N_TRAIN_RUNS="${N_TRAIN_RUNS:-10}"
N_TEST_RUNS="${N_TEST_RUNS:-1}"

# Useful if there's a need to create different datasets from same root data
# (e.g. when data is limited).
INITIAL_SEED="${INITIAL_SEED:-0}"
N_WORKERS="${N_WORKERS:-"$(nproc)"}"

MAX_SEQ_LEN="${MAX_SEQ_LEN:-128}"

# At test time we always use untrimmed sequences of MAX_SEQ_LEN and always replace
# tested tokens with "[MASK]" as opposed to 20% probability of replacing by random
# or the same word in train data. Thus, 0.15 * 0.8 = 0.12 as MASKED_LM_RATIO_TEST.
SHORT_SEQ_PROB_TRAIN="${SHORT_SEQ_PROB_TRAIN:-"0.1"}"
SHORT_SEQ_PROB_TEST="${SHORT_SEQ_PROB_TEST:-"0.0"}"
MASKED_LM_RATIO_TRAIN="${MASKED_LM_RATIO_TRAIN:-"0.15"}"
MASKED_LM_RATIO_TEST="${MASKED_LM_RATIO_TEST:-"0.12"}"
P_MASK_TOKEN_TRAIN="${P_MASK_TOKEN_TRAIN:-"0.8"}"
P_MASK_TOKEN_TEST="${P_MASK_TOKEN_TRAIN:-"1.0"}"


TRAIN_BASE_DIR="${TRAIN_BASE_DIR:-$(pwd)/raw_train}"
TEST_BASE_DIR="${TEST_BASE_DIR:-$(pwd)/raw_test}"
mkdir -p "$TRAIN_BASE_DIR/bookcorpus" "$TRAIN_BASE_DIR/wikipedia/en" "$TRAIN_BASE_DIR/c4"
mkdir -p "$TEST_BASE_DIR/bookcorpus" "$TEST_BASE_DIR/wikipedia/en" "$TEST_BASE_DIR/c4"


function create_base_shards()
{
  local input_path="$1"
  local dataset_subdir="$2"
  local nfiles="$3"
  shuffle_split.sh \
    --inputdir "$input_path" \
    --outputdir "$TRAIN_BASE_DIR/$dataset_subdir" \
    --nfiles $nfiles

  echo "Created base train shards for $dataset_subdir"

  for shard in $(seq $N_TRAIN_SHARDS $(($N_TRAIN_SHARDS + $N_TEST_SHARDS - 1)))
  do
    mv "$TRAIN_BASE_DIR/$dataset_subdir/part_${shard}.txt" "$TEST_BASE_DIR/$dataset_subdir/"
  done
  echo "Created base test shards for $dataset_subdir"
}

if [[ "$CREATE_BASE_SHARDS" == "true" ]]; then
  echo "Started to create base train and test shards"
  if [[ -n  "$BOOKS_PATH" ]]; then
    create_base_shards "$BOOKS_PATH" "bookcorpus" $(($N_TRAIN_SHARDS + $N_TEST_SHARDS))
  fi

  if [[ -n "$WIKI_PATH" ]]; then
    create_base_shards "$WIKI_PATH" "wikipedia/en" $((2 * $N_TRAIN_SHARDS + 2 * $N_TEST_SHARDS))
  fi

  if [[ -n "$C4_PATH" ]]; then
    create_base_shards "$C4_PATH" "c4" $(($N_TRAIN_SHARDS + $N_TEST_SHARDS))
  fi
  echo "Finished to create base train and test shards"
fi

function create_dataset()
{
  local BASE_IN_DIR="$1"
  local BASE_OUT_DIR="$2"
  local N_RUNS="$3"
  local SHORT_SEQ_PROB="$4"
  local MASKED_LM_RATIO="$5"
  local P_MASK_TOKEN="$6"

  for i in $(seq 1 $N_RUNS)
  do
    echo "Starting run $i"
    RUN_OUTDIR="$BASE_OUT_DIR/run_$i"
    mkdir -p $RUN_OUTDIR
    # Pre-shuffle each of input datasets so different documents end up
    # in one shard during distinct runs.
    RUN_SEED=$((i + INITIAL_SEED))
    if [[ -n "$BOOKS_PATH" ]]; then
      BOOKS="$RUN_OUTDIR/bookcorpus"
      shuffle_split.sh --inputdir "$BASE_IN_DIR/bookcorpus"  --outputdir "$BOOKS" \
      --nfiles $N_TRAIN_SHARDS --seed "$RUN_SEED"
    fi
    if [[ -n "$WIKI_PATH" ]]; then
      WIKI="$RUN_OUTDIR/wikipedia"
      shuffle_split.sh --inputdir "$BASE_IN_DIR/wikipedia/en"  --outputdir "$WIKI/en" \
      --nfiles $N_TRAIN_SHARDS --seed "$RUN_SEED"
    fi
    if [[ -n "$C4_PATH" ]]; then
      C4="$RUN_OUTDIR/c4"
      shuffle_split.sh --inputdir "$BASE_IN_DIR/c4"  --outputdir "$C4" \
      --nfiles $N_TRAIN_SHARDS --seed "$RUN_SEED"
    fi
    # Combine datasets and make training/ test sequences out of them
    # -x LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libjemalloc.so \
    mpirun \
    -np $N_WORKERS \
    --oversubscribe \
    --allow-run-as-root \
      preprocess_bert_pretrain \
        --output-format=hdf5 \
        --wikipedia="$WIKI" \
        --books="$BOOKS" \
        --common-crawl="$C4" \
        --sink="$RUN_OUTDIR/dataset" \
        --target-seq-length="$MAX_SEQ_LEN" \
        --short-seq-prob="$SHORT_SEQ_PROB" \
        --masked-lm-ratio="$MASKED_LM_RATIO" \
        --p-mask-token="$P_MASK_TOKEN" \
        --seed="$RUN_SEED"
    rm -rf "$RUN_OUTDIR/bookcorpus" "$RUN_OUTDIR/wikipedia/" "$RUN_OUTDIR/c4"

    echo "Finished run $i"
  done
}

# Create dataset for training
if [[ -n "$N_TRAIN_RUNS" && "$N_TRAIN_RUNS" -gt 0 ]]; then
  create_dataset "$TRAIN_BASE_DIR" "$OUTPUT_TRAIN_DIR" "$N_TRAIN_RUNS" \
    "$SHORT_SEQ_PROB_TRAIN" "$MASKED_LM_RATIO_TRAIN" "$P_MASK_TOKEN_TRAIN"
fi

# Create dataset for evaluation/ test
if [[ -n "$N_TEST_RUNS" && "$N_TEST_RUNS" -gt 0 ]]; then
  create_dataset "$TEST_BASE_DIR" "$OUTPUT_TEST_DIR" "$N_TEST_RUNS" \
    "$SHORT_SEQ_PROB_TEST" "$MASKED_LM_RATIO_TEST" "$P_MASK_TOKEN_TEST"
fi



