import argparse
import glob
from itertools import cycle
import json
import logging
import multiprocessing as mp
import os
from time import asctime
from typing import List, Dict

import h5py
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
parser = argparse.ArgumentParser(
    description="Training data sharding for BERT. It 1) Randomly distributes "
                "samples from input files to a small number of intermediate "
                "output files, 2) shuffles contents of each intermediate file,"
                "3) splits these files so each output file except, possibly, "
                "the last one has --shard_length samples."
)
parser.add_argument(
    '--input_hdf5',
    type=str,
    default='output_train',
    help='Input hdf5_file path')
parser.add_argument(
    '--output_hdf5',
    type=str,
    default='final_data',
    help='Output hdf5_dir path')
parser.add_argument(
    '--intermediate_hdf5',
    type=str,
    default='intermediate_hdf5_shards',
    help='Output dir path for stages 1 and 2')
parser.add_argument(
    '--batch_shards',
    type=int,
    default=128,
    help='Number of input files stored in RAM before randonly assigning to '
         'intermediate output dumps')
parser.add_argument(
    '--n_out_files',
    type=int,
    default=64,
    help='Number of intermediate output files stored to which data gets '
         'randomly distributed')
parser.add_argument(
    '--shard_length',
    type=int,
    default=2 ** 19,
    help='Length of an output shard')
parser.add_argument(
    '--seed',
    type=int,
    default=12345,
    help='random seed')
parser.add_argument(
    '--masking',
    action='store_true',
    default=True,
    help='Read/ Write information about static labels')
parser.add_argument(
    '--max_seq_length',
    type=int,
    default=128,
    help='Max sequence length of input')
parser.add_argument(
    '--masked-lm-ratio',
    type=float,
    default=0.15,
    help='The ratio of the number of tokens to be masked')
args = parser.parse_args()
logging.info('args: {}'.format(args))

# Using this algo:
# https://blog.janestreet.com/how-to-shuffle-a-big-dataset/

keys = ['input_ids', 'input_mask', 'segment_ids', 'next_sentence_labels',
        'filled_lengths']
if args.masking:
    keys.extend(['masked_lm_positions', 'masked_lm_ids'])
n_out_files = args.n_out_files
max_predictions_per_seq = int(round(args.masked_lm_ratio * args.max_seq_length))
np.random.seed(args.seed)


# STAGE 1

def random_distribute_hdf5():
    input_files = sorted(glob.glob(args.input_hdf5 + '/**/part_*.hdf5', recursive=True))
    os.makedirs(args.intermediate_hdf5, exist_ok=True)
    num_shards = len(input_files)
    logging.info('n_input_shards = {}'.format(num_shards))
    init_length = 0
    out_files = []
    for part_idx in range(n_out_files):
        inputs_shape = (init_length, args.max_seq_length)
        max_input_shape = (None, args.max_seq_length)
        labels_shape = (init_length, max_predictions_per_seq)
        max_labels_shape = (None, max_predictions_per_seq)
        f = h5py.File(args.intermediate_hdf5 + '/part{:02d}.hdf5'.format(part_idx), 'w')
        f.create_dataset("input_ids", shape=inputs_shape, maxshape=max_input_shape,
                         dtype='i4', compression='gzip')
        f.create_dataset("input_mask", shape=inputs_shape, maxshape=max_input_shape,
                         dtype='i1', compression='gzip')
        f.create_dataset("segment_ids", shape=inputs_shape, maxshape=max_input_shape,
                         dtype='i1', compression='gzip')
        f.create_dataset("next_sentence_labels", shape=init_length,
                         maxshape=(None,), dtype='i1', compression='gzip')
        f.create_dataset("filled_lengths", shape=init_length,
                         maxshape=(None,), dtype='i4', compression='gzip')
        if args.masking:
            f.create_dataset("masked_lm_positions", shape=labels_shape,
                             maxshape=max_labels_shape, dtype='i4', compression='gzip')
            f.create_dataset("masked_lm_ids", shape=labels_shape,
                             maxshape=max_labels_shape, dtype='i4', compression='gzip')
        out_files.append(f)


    content = {}
    for key in keys:
        content[key] = []

    for ifile_idx in tqdm(range(num_shards)):
        with h5py.File(f'{input_files[ifile_idx]}', 'r') as f:
            for key in keys:
                content[key].append(f[key][:])

        if ((ifile_idx + 1) % args.batch_shards == 0
            or ifile_idx == num_shards - 1):
            logging.info('Started dumping shard no {} at {}'
                         .format(ifile_idx, asctime()))
            for key in keys:
                content[key] = np.concatenate(content[key], axis=0)
            file_length = len(content['input_ids'])
            inds = np.random.permutation(file_length)
            inds_chunks = np.array_split(inds, n_out_files)
            # Append to each existing output file its random chunk of data
            # of the current input file.
            for i in range(n_out_files):
                prev_len = len(out_files[i]['input_ids'])
                chunk = inds_chunks[i]
                new_len = prev_len + chunk.shape[0]
                for key in keys:
                    out_files[i][key].resize(new_len, axis=0)
                    out_files[i][key][prev_len:new_len] = content[key][chunk]
                out_files[i].flush()
            for key in keys:
                content[key] = []
            logging.info('Finished dumping shard no {} at {}'
                         .format(ifile_idx, asctime()))

    for fname in out_files:
        fname.close()
# STAGE 2


def shuffle_hdf5_file(filename: str, keys: List[str], seed: int) -> None:
    """Shuffles all samples inside a one HDF5 file"""
    np.random.seed(seed)
    logging.info('Started shuffling file {} at {}'.format(filename, asctime()))
    with h5py.File(filename, 'r+') as f:
        n_samples = len(f['input_ids'])
        new_order = np.random.permutation(n_samples)
        for key in keys:
            f[key][:] = f[key][:][new_order]
            logging.info('Finished writing {} dataset in {} at {}'
                         .format(key, filename, asctime()))
    logging.info('Finished shuffling file {} at {}'.format(filename, asctime()))


# STAGE 3

def create_empty_hdf5(filename: str, shard_length: int) -> h5py.File:
    inputs_shape = (shard_length, args.max_seq_length)
    labels_shape = (shard_length, max_predictions_per_seq)
    f = h5py.File(filename, 'w')
    f.create_dataset("input_ids", shape=inputs_shape,
                     dtype='i4', compression='gzip')
    f.create_dataset("input_mask", shape=inputs_shape,
                     dtype='i1', compression='gzip')
    f.create_dataset("segment_ids", shape=inputs_shape,
                     dtype='i1', compression='gzip')
    f.create_dataset("next_sentence_labels", shape=shard_length,
                     dtype='i1', compression='gzip')
    f.create_dataset("filled_lengths", shape=shard_length,
                     dtype='i4', compression='gzip')
    if args.masking:
        f.create_dataset("masked_lm_positions", shape=labels_shape,
                         dtype='i4', compression='gzip')
        f.create_dataset("masked_lm_ids", shape=labels_shape,
                         dtype='i4', compression='gzip')
    return f


def split_files(
    fnames: List[str], outdir: str,
    proc_id: int, keys: List[str], mark_leftovers=True
) -> None:
    remainder_len = 0
    shard_len = args.shard_length
    buffer = {key: [] for key in keys}
    idx_out = 0

    os.makedirs(outdir, exist_ok=True)
    for in_fname in fnames:
        logging.info('Started splitting file {} at {}'
                     .format(in_fname, asctime()))
        with h5py.File(in_fname, 'r') as f:
            n_samples = len(f['input_ids'])
            n_used = 0
            while n_samples - n_used >= shard_len - remainder_len:
                out_fname = outdir + f'/part_{proc_id:02d}_{idx_out:03d}.hdf5'
                out_file = create_empty_hdf5(out_fname, shard_len)
                logging.info('Started writing output file {} at {}'
                             .format(out_fname, asctime()))
                upper_bound = n_used + shard_len - remainder_len
                for key in keys:
                    buffer[key].append(f[key][n_used:upper_bound])
                    buffer[key] = np.concatenate(buffer[key])
                    out_file[key][:] = buffer[key]
                    buffer[key] = []
                out_file.close()
                idx_out += 1
                logging.info('Finished writing output file {} at {}'
                             .format(out_fname, asctime()))

                remainder_len = 0
                n_used = upper_bound
            else:
                for key in keys:
                    buffer[key].append(f[key][n_used:n_samples])
                remainder_len += n_samples - n_used
    else:
        # Dealing with chunks with length less than `shard_length`. When
        # flushing them, we can mark them as "leftovers" or give a usual name.
        if remainder_len != 0:
            if mark_leftovers:
                out_fname = outdir + f'/part_{proc_id:02d}_leftover.hdf5'
            else:
                out_fname = outdir + f'/part_{proc_id:02d}_{idx_out:03d}.hdf5'
            out_file = create_empty_hdf5(out_fname, remainder_len)
            logging.info('Started writing output file {} at {}'
                         .format(out_fname, asctime()))
            for key in keys:
                buffer[key] = np.concatenate(buffer[key])
                out_file[key][:] = buffer[key]
                buffer[key] = []
            out_file.close()
            logging.info('Finished writing output file {} at {}'
                         .format(out_fname, asctime()))

def wrapper(arguments):
    filename, keys, proc_id = arguments
    shuffle_hdf5_file(filename, keys, seed=proc_id)
    split_files([filename], args.output_hdf5, proc_id, keys,)


file_names = []
for n in range(n_out_files):
    file_names.append(args.intermediate_hdf5 + f'/part{n:02d}.hdf5')

if __name__ == '__main__':
    random_distribute_hdf5()
    with mp.get_context("spawn").Pool(
            min(mp.cpu_count(), n_out_files)
    ) as pool:
        pool.map(wrapper, zip(file_names, cycle([keys]), range(n_out_files)))
    # We aim for all but the last file to have the same length so we treat
    # leftovers in the second pass.
    leftovers = sorted(glob.glob(args.output_hdf5 + '/*_leftover.hdf5'))
    split_files(fnames=leftovers, outdir=args.output_hdf5, proc_id=n_out_files,
                keys=keys, mark_leftovers=False)
    for file in leftovers:
        os.remove(file)
