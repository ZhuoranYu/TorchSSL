alg=$1
GPU=$2
export CUDA_VISIBLE_DEVICES=0,1,2,3
python ${alg}.py  --c config/${alg}/${alg}_cifar100_400_0.yaml
