#!/bin/bash
cd ~/workstorage/Reza-Project/federated-retraction
export OMP_NUM_THREADS=1
until [ -f data_v2/certainty_abstracts.parquet ]; do sleep 120; done
.venv/bin/python scripts/v2/b7_certainty_eval.py > logs/b7_certainty.log 2>&1
