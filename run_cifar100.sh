alg=$1
export CUDA_VISIBLE_DEVICES=4,5,6,7
python ${alg}.py --c config/${alg}/${alg}_cifar100_400_0.yaml
