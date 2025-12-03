#!/bin/bash
module load conda
conda activate $PSCRATCH/ccl-bench

export PATH=~/CS5470/assignment_1/nsys_new/opt/nvidia/nsight-systems-cli/2025.5.1/bin:$PATH
export HF_HOME=$PSCRATCH/huggingface

