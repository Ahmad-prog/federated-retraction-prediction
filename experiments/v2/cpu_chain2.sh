cd ~/workstorage/Reza-Project/federated-retraction
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python scripts/v2/b4_scale_dp.py --seeds 10 --jobs 10 > logs/b4_scale_dp.log 2>&1
echo CHAIN2_DONE >> logs/b4_scale_dp.log
