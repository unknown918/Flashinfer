#!/usr/bin/env bash

export TRANSFORMERS_VERBOSITY=error
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export HF_HUB_VERBOSITY=error
export HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_DATASETS_DISABLE_PROGRESS_BARS=1
export DATASETS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

MODEL_PATH="/home1/csu_yqj/Llama-3.1-8B-Instruct/"
PROMPT_PATH="./paper/prompt.txt"
BATCH_SIZES=(1 2 4 8)
SEQ_LENS=(4000 8000 16000 32000)
BUDGETS=(1024 2048)

run_and_cleanup() {
    setsid "$@" &
    pid=$!

    wait "$pid"
    exit_code=$?

    kill -TERM -- "-$pid" 2>/dev/null || true
    sleep 1
    kill -KILL -- "-$pid" 2>/dev/null || true

    echo "Process group $pid cleaned up. Sleeping for 10 seconds..."
    sleep 10

    return "$exit_code"
}

for seq in "${SEQ_LENS[@]}"; do
    for batch_size in "${BATCH_SIZES[@]}"; do
        echo "Dense Attention, with seq_len $seq and batch_size $batch_size"
        run_and_cleanup \
            python3 -m paper.tpot \
            --model-path "$MODEL_PATH" \
            --prompt-path "$PROMPT_PATH" \
            --seq-len "$seq" \
            --batch-size "$batch_size" \
            --num-dense-layers 32


        for budget in "${BUDGETS[@]}"; do
            echo "Sparse Attention, with seq_len $seq, budget $budget and batch_size $batch_size"
            run_and_cleanup \
                python3 -m paper.tpot \
                --model-path "$MODEL_PATH" \
                --prompt-path "$PROMPT_PATH" \
                --budget "$budget" \
                --seq-len "$seq" \
                --batch-size "$batch_size" \
                --num-dense-layers 2
        done
    done
done