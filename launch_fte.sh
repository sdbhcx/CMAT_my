#!/bin/bash
# 启动 V1-A.FTE 三档消融：B1(mean) / B2(conditional) / B3(concat)
# GPU 分配：B1->0, B2->2, B3->3（GPU 1 被他人占用）
cd /home/junbo/wyn/codes/CMAT_my || exit 1

export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:/usr/local/cuda-11.8/targets/x86_64-linux/lib
PY=/home/datasets/ljk/miniconda3/envs/cmat/bin/python
mkdir -p runs_fte

for pair in "b1 0" "b2 2" "b3 3"; do
    set -- $pair
    tag=$1
    gpu=$2
    CUDA_VISIBLE_DEVICES=$gpu nohup $PY train.py --config configs/piadv2_fte_$tag.yaml \
        > runs_fte/$tag.log 2>&1 &
    echo "launched $tag on GPU $gpu (pid $!)"
done

sleep 5
echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
