#!/bin/bash

# This script combines all .txt files in a directory, shuffles their strings
# and outputs shuffled contents as sharded files.

set -x
function usage()
{
   cat << HEREDOC

   Usage: $progname [-o|--outputdir PATH] [-h|--help TIME_STR]

   optional arguments:
     -h, --help            show this help message and exit
     -i, --inputdir PATH   pass in a localization of input dataset directory
     -o, --outputdir PATH  pass in a localization of output dataset directory
     -n, --nfiles          number of split output shards
     -s, --seed            random seed for shuffling lines of text across all input shards

HEREDOC
}
function get_fixed_random()
{
  openssl enc -aes-256-ctr -pass pass:"$1" -nosalt </dev/zero 2>/dev/null
}

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
    -h|--help)
      usage
      exit 0
      ;;
    -i|--inputdir)
      INPUTDIR="$2"
      shift # past argument
      shift # past value
      ;;
    -o|--outputdir)
      OUTPUTDIR="$2"
      shift # past argument
      shift # past value
      ;;
    -n|--nfiles)
      NSPLITS="$2"
      shift # past argument
      shift # past value
      ;;
    -s|--seed)
      SEED="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      usage
      exit 1
      ;;
  esac
done

# Defaults
: ${NSPLITS:="$((128+8))"}
: ${SEED:=12345}
# OUTPUTDIR="$(dirname "$(readlink -f "$0")")/shuffled"
: ${OUTPUTDIR:="$(dirname "$INPUTDIR")/shuffled"}

mkdir -p "$OUTPUTDIR"
cat $INPUTDIR/*.txt | shuf --random-source=<(get_fixed_random "$SEED") > tmp_shuffled.txt
zcat $INPUTDIR/*.json.gz | shuf --random-source=<(get_fixed_random "$SEED") >> tmp_shuffled.txt


LINE_SPLITS=$(($(wc -l tmp_shuffled.txt | awk '{print $1}') / $NSPLITS))
if (( $LINE_SPLITS == 0 )); then
  echo "--nfiles is $NSPLITS, which is more than total lines in dataset"
  exit 1
fi

split -l $LINE_SPLITS -d -a 3 --additional-suffix ".txt" tmp_shuffled.txt "$OUTPUTDIR/part_"
rm tmp_shuffled.txt