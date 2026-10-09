cd ~/workstorage/Reza-Project/federated-retraction
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
while ps -eo args | grep -q "[b]2_main_benchmark"; do sleep 60; done
.venv/bin/python scripts/v2/c2_heads.py --source abstracts --emb abstracts_modernbert --seeds 10 --jobs 10 > logs/c2_abs_modernbert_emb.log 2>&1
.venv/bin/python scripts/v2/c2_heads.py --source abstracts --emb abstracts_modernbert --with-tab --seeds 10 --jobs 10 > logs/c2_abs_modernbert_tab.log 2>&1
while [ ! -f data_v2/emb/abstracts_qwen3emb8b.npy ]; do sleep 120; done
.venv/bin/python scripts/v2/c2_heads.py --source abstracts --emb abstracts_qwen3emb8b --seeds 10 --jobs 10 > logs/c2_abs_qwen8b_emb.log 2>&1
.venv/bin/python scripts/v2/c2_heads.py --source abstracts --emb abstracts_qwen3emb8b --with-tab --seeds 10 --jobs 10 > logs/c2_abs_qwen8b_tab.log 2>&1
echo CHAIN3_DONE > logs/chain3.done
