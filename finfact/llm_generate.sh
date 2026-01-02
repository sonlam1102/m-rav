#!/usr/bin/bash


#PBS -N finfact-llava34
#PBS -j oe -l ngpus=2
#PBS -q GPU-S
#PBS -o pbs_infer-sp1.log
#PBS -e pbs_error-sp1.log
#PBS -M s2320014@jaist.ac.jp
#PBS -m e

source ~/.bashrc
SOURCE_PATH=/home/s2320014/finfact

export PATH="/home/s2320014/miniconda3/bin:$PATH"
cd $SOURCE_PATH

python llm_generate.py --path="/home/s2320014/data/finfact/images_new"