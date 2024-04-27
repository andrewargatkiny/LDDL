# Dataset Preparation for LLM training with BERT-like objective.

This directory contains scripts to prepare and process datasets for training a BERT-like language model using a masked language modeling (MLM) and next sentence prediction (NSP) objectives. 

The process involves creating initial training and test splits of the data, creating out of texts datapoints for MLM and NSP training, and shuffling the data several times during the process to ensure proper randomness even when one raw text is used several times for datapoints creation.


## Scripts Description

1. **create_dataset.sh** - This script is used to generate datasets from raw text sources. It randomly splits the raw original data into training and test datasets. Then for predefined number of runs repeatedly gathers (train/ test) data from different sources (Wikipedia, Bookcorpus, Common Crawl), reshuffles it and creates formatted samples for BERT pretraining using the `preprocess_bert_pretrain` command (**pretrain.py** script). The script organizes the data into shards files and stores outputs for distinct runs in different directories.

2. **shuffle_gather_output.py** - This Python script randomly shuffles and distributes the data from previously created shards into new files, ensuring that each new shard contains datapoints from all runs over the raw original dataset, and has the same number of samples, except the last shard. It uses multi-processing for faster execution.

## Usage

## 1. create_dataset.sh

### Overview

The `create_dataset.sh` script is responsible for generating a dataset from source texts such as books, Wikipedia, and Common Crawl data (C4). It preprocesses, divides and formats these texts to create datapoints for train and test datasets with MLM and NSP objectives like those used in BERT.

### Parameters

- `BOOKS_PATH`: Directory containing book corpus data. Default: `/workspace/bert/data/bookcorpus/source/`
- `WIKI_PATH`: Directory containing Wikipedia dataset. Default: `/workspace/bert/data/wikipedia_dataset/source/en`
- `C4_PATH`: Directory containing Common Crawl data. Default: `/workspace/bert/data/c4/source/`
- `N_TRAIN_SHARDS`: Number of shards for training data. Default: `256`
- `N_TEST_SHARDS`: Number of shards for testing data. Default: `16`
- `OUTPUT_TRAIN_DIR`: Output directory for processed training data. Default: `$(pwd)/output_train`
- `OUTPUT_TEST_DIR`: Output directory for processed testing data. Default: `$(pwd)/output_test`
- `N_TRAIN_RUNS`: Number of times the training dataset is processed to enhance variability. Default: `10`
- `N_TEST_RUNS`: Number of times the testing dataset is processed. Default: `1`
- `MAX_SEQ_LEN`: Maximum sequence length of data points. Default: `128`
- `INITIAL_SEED`: Initial random seed for data shuffling. Default: `0`
- `SHORT_SEQ_PROB_TRAIN`: Probability of creating a short sequence during training data generation. Default: `0.1`
- `SHORT_SEQ_PROB_TEST`: Probability of creating a short sequence during test data generation. Default: `0.0`
- `MASKED_LM_RATIO_TRAIN`: Ratio of tokens that are masked in the training data. Default: `0.15`
- `MASKED_LM_RATIO_TEST`: Ratio of tokens that are masked in the testing data. Default: `0.12`
- `P_MASK_TOKEN_TRAIN`: Probability of replacing a masked token with a random token during training. Default: `0.8`
- `P_MASK_TOKEN_TEST`: Probability of replacing a masked token with a random token during testing. Default `1.0` (all masked tokens are replaced).
### Example Usage

```bash
# In this example we don't use c4 dataset.
# We want to reuse the raw data 5 times for training dataset 
# Also we don't want to create processed test dataset at all
# But to still create raw test texts split for later preparation.
BOOKS_PATH=/path/to/bookcorpus \
WIKI_PATH=/path/to/wikipedia \
C4_PATH=/"" \
N_TRAIN_RUNS=5 \
N_TEST_RUNS=0 \
./create_dataset.sh
```

## 2. shuffle_gather_output.py

### Overview

The shuffle_gather_output.py script shuffles and gathers the outputs from the create_dataset.sh script into a dataset ready for pretraining. It randomly distributes a run's samples across files, shuffles contents within files, and splits into final output shards of a specified length.

### Parameters

### Parameters

- `--input_hdf5`: Path to input HDF5 files. Default: `output_train`
- `--output_hdf5`: Path for final shuffled HDF5 files. Default: `final_data`
- `--intermediate_hdf5`: Directory for intermediate shuffling stages. Default: `intermediate_hdf5_shards`
- `--batch_shards`: Number of input files processed before distribution to intermediate files. Default: `128`
- `--n_out_files`: Number of intermediate output files during shuffling. Default: `64`
- `--shard_length`: Number of samples per output shard file. Default: `524288` (2^19)
- `--seed`: Seed for random number generation. Default: `12345`
- `--masking`: Enables reading and writing static label information if set. Default: `True`
- `--max_seq_length`: Maximum sequence length of inputs. Default: `128`
- `--masked-lm-ratio`: The ratio of the number of tokens to be masked. Default: `0.15`

### Example Usage

```bash
python shuffle_gather_output.py --input_hdf5 /path/to/input --output_hdf5 /path/to/output
```
