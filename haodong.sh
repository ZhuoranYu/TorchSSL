alg=$1
GPU=$2
DATASET=$3
LABELS=$4
SEED=$5

export CUDA_VISIBLE_DEVICES=${GPU}

python debiased.py  --c haodong/${alg}_${DATASET}_${LABELS}_${SEED}.yaml