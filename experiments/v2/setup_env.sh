set -e
cd ~/workstorage/Reza-Project/federated-retraction
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip uv
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install "transformers>=4.48" accelerate peft datasets sentence-transformers scikit-learn xgboost pandas pyarrow requests pyyaml textstat matplotlib tqdm scipy lxml opacus "flwr>=1.15" safetensors
python - <<"PY"
import torch, transformers, peft, xgboost, sklearn, opacus
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("devices", torch.cuda.device_count(), [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); print("matmul ok", (x @ x).float().abs().mean().item())
print("transformers", transformers.__version__, "peft", peft.__version__, "xgb", xgboost.__version__, "opacus", opacus.__version__)
PY
echo SETUP_DONE
