alg=$1
GPU=$2
export CUDA_VISIBLE_DEVICES=${GPU}
python ${alg}.py --world-size 1 --rank 0 --gpu ${GPU} --c config/${alg}/${alg}_svhn_40_0.yaml
