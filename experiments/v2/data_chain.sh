#!/bin/bash
# Unattended data pipeline, stage 2: waits for Phase A (download + parse), then builds the
# full-text dataset (A6) and audits it for leakage (D1). Writes data_v2/FT_READY when done.
cd ~/workstorage/Reza-Project/federated-retraction
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
log=logs/data_chain.log
echo "$(date -u +%FT%TZ) waiting for Phase A" >> $log
until grep -q PHASE_A_DONE logs/phaseA.log; do sleep 120; done
echo "$(date -u +%FT%TZ) A6 start" >> $log
until .venv/bin/python scripts/v2/a6_ft_dataset.py >> logs/a6_ft_dataset.log 2>&1; do
  echo "$(date -u +%FT%TZ) A6 failed, retry in 5 min" >> $log; sleep 300; done
echo "$(date -u +%FT%TZ) A6 done; D1 fulltext audit" >> $log
.venv/bin/python scripts/v2/d1_leak_audit.py --source fulltext > logs/d1_fulltext.log 2>&1
touch data_v2/FT_READY
echo "$(date -u +%FT%TZ) FT_READY" >> $log
