#!/usr/bin/bash


#PBS -N factify-llava34-direct
#PBS -j oe -l ngpus=1
#PBS -q GPU-1
#PBS -o pbs_infer-sp1d.log
#PBS -e pbs_error-sp1d.log
#PBS -M s2320014@jaist.ac.jp
#PBS -m e

source ~/.bashrc
SOURCE_PATH=/home/s2320014/factify

export PATH="/home/s2320014/miniconda3/bin:$PATH"
cd $SOURCE_PATH

python llms_verify.py --path="/home/s2320014/data/factify/images_set/test" 