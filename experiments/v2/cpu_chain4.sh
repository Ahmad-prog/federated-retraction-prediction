#!/bin/bash
# Full-text embedding heads (CPU) as soon as each GPU embedding file appears.
cd ~/workstorage/Reza-Project/federated-retraction
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
P=".venv/bin/python scripts/v2/c2_heads.py --source fulltext --seeds 10 --jobs 10"
for e in fulltext_modernbert_sections fulltext_qwen3emb4b_sections; do
  until [ -f data_v2/emb/$e.npy ] && [ -f data_v2/FT_READY ]; do sleep 180; done
  while [ ! -f logs/chain3.done ] && ps -eo args | grep -q "[c]2_heads.py --source abstracts"; do sleep 120; done
  $P --emb $e > logs/c2_$e.log 2>&1
  $P --emb $e --with-tab > logs/c2_${e}_tab.log 2>&1
done
echo CHAIN4_DONE > logs/chain4.done
