#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

CKPT="llava-v1.5-13b-gpt4-finetune"
SPLIT="llava_gqa_testdev_balanced"
GQADIR="/mnt/aimsfrontierresearcheu01/gpt-rad/data/llava-1.5/visual_instruction_tuning/eval/gqa/data/"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_multi \
        --model-path /mnt/sdrgstd01scus/user/v-clairejin/llava_1.5_13B_gpt4_finetune_v2/ \
        --question-file /mnt/aimsfrontierresearcheu01/gpt-rad/data/llava-1.5/visual_instruction_tuning/eval/gqa/$SPLIT.jsonl \
        --image-folder  /mnt/aimsfrontierresearcheu01/gpt-rad/data/llava-1.5/visual_instruction_tuning/eval/gqa/data/images \
        --answers-file /mnt/aimsfrontierresearcheu01/gpt-rad/data/llava-1.5/visual_instruction_tuning/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode vicuna_v1 &
done

wait

output_file=/mnt/aimsfrontierresearcheu01/gpt-rad/data/llava-1.5/visual_instruction_tuning/eval/gqa/answers/$SPLIT/$CKPT/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat /mnt/aimsfrontierresearcheu01/gpt-rad/data/llava-1.5/visual_instruction_tuning/eval/gqa/answers/$SPLIT/$CKPT/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python scripts/convert_gqa_for_eval.py --src $output_file --dst $GQADIR/testdev_balanced_predictions.json

cd $GQADIR
python eval/eval.py --tier testdev_balanced
