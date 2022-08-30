alg=$1
GPU=$2
export CUDA_VISIBLE_DEVICES=${GPU}
python ${alg}.py --c config/${alg}/${alg}_cifar10_lt_10_100_0.yaml
