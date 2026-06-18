#!/bin/bash

SEQ_LENS=(8192 16384 32768 128000)
BUDGETS=(512 1024 2048)

for seq in "${SEQ_LENS[@]}"; do

  echo "=============================="
  echo "Running baseline | seq=$seq"
  python3 -m paper.baseline --seq_len $seq

  for budget in "${BUDGETS[@]}"; do

    echo "Running e2e | seq=$seq | budget=$budget"
    python3 -m paper.e2e --seq_len $seq --budget $budget

  done
done