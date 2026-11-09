#!/usr/bin/env bash
set -euo pipefail
# Run in a CPU preparation session. This installs into an isolated environment.
# It does not allocate a GPU or install into the Colab CLI virtual environment.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi --list-gpus 2>/dev/null | grep -q '^GPU '; then
  echo 'Prepare dependencies on CPU before starting the budgeted G4 session.' >&2
  exit 2
fi
task_env_root="${H3_ENV_ROOT:-/content/colab-lab/h3-environment}"
task_source_dir="$task_env_root/FastVideo"
task_commit='ce970eb715215c79a169658d7aa6c8a96e64891e'
mkdir -p "$task_env_root"
python3 -m pip install -q uv
export UV_PYTHON_INSTALL_DIR="$task_env_root/python"
uv python install 3.12
uv venv --python 3.12 --managed-python --allow-existing --seed "$task_env_root/.venv"
if [ ! -d "$task_source_dir/.git" ]; then
  git clone --filter=blob:none --no-checkout https://github.com/hao-ai-lab/FastVideo.git "$task_source_dir"
fi
git -C "$task_source_dir" fetch --depth 1 origin "$task_commit"
git -C "$task_source_dir" checkout --detach "$task_commit"
# Remove obsolete optional CuTe 4.8 split-library packages left by an earlier
# resolver pass; the pinned FA4 build uses the self-contained 4.6 base library.
uv pip uninstall --python "$task_env_root/.venv/bin/python" nvidia-cutlass-dsl-libs-cu12 nvidia-cutlass-dsl-libs-cu13
UV_TORCH_BACKEND=cu130 uv pip install --python "$task_env_root/.venv/bin/python" -e "$task_source_dir" 'quack-kernels==0.5.3' 'flash-attn-4 @ git+https://github.com/Dao-AILab/flash-attention.git@14c377950125c70b7a9dabf9c561fca53715ac7d#subdirectory=flash_attn/cute'
"$task_env_root/.venv/bin/python" -m pip check
# CPU/GPU Colab sessions do not share /content. Preserve a compatible wheel/source
# cache on Drive and reproduce this environment in the G4 VM before inference;
# this CPU install alone is not proof that Blackwell kernels work.
