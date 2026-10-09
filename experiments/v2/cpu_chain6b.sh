#!/bin/bash
# rerun B11 record-level part with the clip-only reference once chain 6 is done
cd ~/workstorage/Reza-Project/federated-retraction
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore
until [ -f logs/chain6.done ]; do sleep 120; done
.venv/bin/python scripts/v2/b11_dp_heads.py --only record --seeds 5 --jobs 5 --epochs 10 > logs/b11_record.log 2>&1
echo DONE > logs/chain6b.done
