#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <apinn_pid>" >&2
  exit 2
fi

apinn_pid="$1"

while kill -0 "${apinn_pid}" 2>/dev/null; do
  sleep 15
done

source /home/mingshengpeng/miniconda3/etc/profile.d/conda.sh
conda activate torch_gpu
exec python /home/mingshengpeng/PhD/ADD-PINNs-JCP/ADD-PINNs_BeamGood/main_reduced.py
