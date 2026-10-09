#!/bin/bash
# B11 privacy heads (record-level DP-SGD + journal-level client DP) on CPU
cd ~/workstorage/Reza-Project/federated-retraction
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore
.venv/bin/python scripts/v2/b11_dp_heads.py --only record --seeds 5 --jobs 5 --epochs 10 > logs/b11_record.log 2>&1
.venv/bin/python scripts/v2/b11_dp_heads.py --only journal --seeds 5 --jobs 5 --rounds 200 > logs/b11_journal.log 2>&1
echo CHAIN6_DONE > logs/chain6.done
