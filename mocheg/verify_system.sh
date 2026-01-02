#!/usr/bin/bash


#PBS -N mocheg-system-llava1.6-direct
#PBS -j oe -l ngpus=2
#PBS -q GPU-S
#PBS -o pbs_infer-lv16-d.log
#PBS -e pbs_error-lv16-d.log
#PBS -M s2320014@jaist.ac.jp
#PBS -m e

source ~/.bashrc
SOURCE_PATH=/home/s2320014/multimodal-fact-checking

export PATH="/home/s2320014/miniconda3/bin:$PATH"
cd $SOURCE_PATH

python llms_verify_system.py --path="/home/s2320014/data/mocheg" 