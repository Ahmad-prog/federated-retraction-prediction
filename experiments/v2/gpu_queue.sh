#!/bin/bash
# Sequential GPU job runner pinned to GPU 1. Jobs are shell snippets in queue/pending,
# executed in name order; logs in logs/gpu_<job>.log. Add jobs by dropping files in.
cd ~/workstorage/Reza-Project/federated-retraction
export CUDA_VISIBLE_DEVICES=1 HF_HOME=$PWD/hf_home PYTHONUNBUFFERED=1
while true; do
  j=$(ls queue/pending 2>/dev/null | sort | head -1)
  if [ -z "$j" ]; then sleep 30; continue; fi
  mv queue/pending/$j queue/running/$j
  echo "$(date -u +%FT%TZ) START $j" >> logs/gpu_queue.log
  if bash queue/running/$j > logs/gpu_$j.log 2>&1; then
    mv queue/running/$j queue/done/$j; echo "$(date -u +%FT%TZ) DONE $j" >> logs/gpu_queue.log
  else
    mv queue/running/$j queue/failed/$j; echo "$(date -u +%FT%TZ) FAIL $j" >> logs/gpu_queue.log
  fi
done
