alg=$1
GPU=$2
export CUDA_VISIBLE_DEVICES=${GPU}
python ${alg}.py  --c config/${alg}/${alg}_cifar6_ood_400_0.yaml
