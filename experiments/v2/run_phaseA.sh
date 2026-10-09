set -e
cd ~/workstorage/Reza-Project/federated-retraction
PY=.venv/bin/python
for s in a2_map_pmcid a3_pmc_controls a4_fetch_fulltext a5_parse_clean; do
  echo "=== $s $(date)"
  until $PY scripts/v2/$s.py; do echo "retrying $s in 60s"; sleep 60; done
done
echo "PHASE_A_DONE $(date)"
