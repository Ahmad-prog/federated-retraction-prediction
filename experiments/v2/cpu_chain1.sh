cd ~/workstorage/Reza-Project/federated-retraction
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/v2/b2_main_benchmark.py --seeds 10 --jobs 30 > logs/b2_main.log 2>&1
.venv/bin/python scripts/v2/b3_ablations.py --seeds 10 --jobs 10 > logs/b3_ablations.log 2>&1
echo CHAIN1_DONE >> logs/b3_ablations.log
