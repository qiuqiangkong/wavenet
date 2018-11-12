#!/bin/bash
DATASET_DIR="/vol/vssp/datasets/audio/VCTK/vctk/VCTK-Corpus"
WORKSPACE="/vol/vssp/msos/qk/workspaces/wavenet"

CUDA_VISIBLE_DEVICES=1 python main.py train --dataset=vctk --dataset_dir=$DATASET_DIR --workspace=$WORKSPACE [--condition] [--cuda]

CUDA_VISIBLE_DEVICES=2 python main.py generate --dataset=vctk --workspace=$WORKSPACE --iteration=10000 --samples=100 [--global_condition=311] [--fast_generation] [--cuda]

