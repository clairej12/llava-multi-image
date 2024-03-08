#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-13b-baseline"
SPLIT="mm-vet-eval"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_multi \
        --model-path /mnt/sdrgstd01scus/user/v-clairejin/llava_1.5_13B_finetune_v2 \
        --question-file /mnt/icodeeval01wus3/user/v-clairejin/mm-vet/$SPLIT.jsonl \
        --image-folder /mnt/icodeeval01wus3/user/v-clairejin/mm-vet/images/ \
        --answers-file /mnt/icodeeval01wus3/user/v-clairejin/mm-vet/results/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/mnt/icodeeval01wus3/user/v-clairejin/mm-vet/results/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /mnt/icodeeval01wus3/user/v-clairejin/mm-vet/results/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

# python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT --dir /mnt/aimsfrontierresearcheu01/gpt-rad/data/llava-1.5/visual_instruction_tuning/eval/vqav2

