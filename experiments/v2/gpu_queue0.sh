#!/bin/bash
# Sequential GPU job runner pinned to GPU 0. Jobs are shell snippets in queue0/pending,
# executed in name order; logs in logs/gpu_<job>.log. Add jobs by dropping files in.
cd ~/workstorage/Reza-Project/federated-retraction
export CUDA_VISIBLE_DEVICES=0 HF_HOME=$PWD/hf_home PYTHONUNBUFFERED=1
while true; do
  j=$(ls queue0/pending 2>/dev/null | sort | head -1)
  if [ -z "$j" ]; then sleep 30; continue; fi
  mv queue0/pending/$j queue0/running/$j
  echo "$(date -u +%FT%TZ) START $j" >> logs/gpu0_queue.log
  if bash queue0/running/$j > logs/gpu0_$j.log 2>&1; then
    mv queue0/running/$j queue0/done/$j; echo "$(date -u +%FT%TZ) DONE $j" >> logs/gpu0_queue.log
  else
    mv queue0/running/$j queue0/failed/$j; echo "$(date -u +%FT%TZ) FAIL $j" >> logs/gpu0_queue.log
  fi
done
