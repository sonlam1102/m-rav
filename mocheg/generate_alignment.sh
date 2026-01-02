#!/usr/bin/bash


#PBS -N test_p4
#PBS -j oe -l ngpus=2
#PBS -q GPU-S
#PBS -o pbs_infer-sp1.log
#PBS -e pbs_error-sp1.log
#PBS -M s2320014@jaist.ac.jp
#PBS -m e

source ~/.bashrc
SOURCE_PATH=/home/s2320014/multimodal-fact-checking

export PATH="/home/s2320014/miniconda3/bin:$PATH"
cd $SOURCE_PATH

python llms_generate_alignment.py --path="/home/data/mocheg" --system --test --limit --start=0 --end=200
