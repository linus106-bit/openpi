#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  scripts/run_multinode_from_hostfile.sh <config_name> --exp_name <run_name> [train args...]

Example:
  NPROC_PER_NODE=8 MASTER_PORT=29500 \
    scripts/run_multinode_from_hostfile.sh pi0_aloha_sim --exp_name pytorch_multinode

This script reads HOSTFILE (default: /horovod/generated/hostfile), uses the first
host as MASTER_ADDR, infers NNODES from the host list, and computes NODE_RANK by
matching the current machine's hostname/IP against the hostfile entries.
Run the same command on every node in the hostfile.

Expected hostfile formats:
  10.0.0.1
  10.0.0.1 slots=8
  worker-0 slots=8

Environment overrides:
  HOSTFILE          Path to the Horovod/OpenMPI hostfile.
  NPROC_PER_NODE   Processes to launch per node. Defaults to detected GPU count, or 1.
  GPUS_PER_NODE    Alias used only when NPROC_PER_NODE is not set.
  NODE_RANK        Explicit node rank. Use this if auto-detection cannot match the node.
  MASTER_ADDR      Explicit master address. Defaults to the first host in HOSTFILE.
  MASTER_PORT      Master port. Defaults to 29500.
  TRAIN_ENTRYPOINT Training script. Defaults to scripts/train_pytorch.py.
  USE_UV           Set to 0 to call torchrun directly instead of "uv run torchrun".
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -lt 1 ]]; then
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." >/dev/null 2>&1 && pwd)"
cd "$REPO_ROOT"

HOSTFILE="${HOSTFILE:-/horovod/generated/hostfile}"
MASTER_PORT="${MASTER_PORT:-29500}"
TRAIN_ENTRYPOINT="${TRAIN_ENTRYPOINT:-scripts/train_pytorch.py}"
USE_UV="${USE_UV:-1}"

if [[ ! -f "$HOSTFILE" ]]; then
  echo "Hostfile not found: $HOSTFILE" >&2
  exit 1
fi

mapfile -t HOSTS < <(
  awk '
    {
      sub(/#.*/, "")
      if ($1 == "") {
        next
      }
      host = $1
      sub(/,$/, "", host)
      if (host ~ /^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+:[0-9]+$/) {
        sub(/:[0-9]+$/, "", host)
      }
      if (!seen[host]++) {
        print host
      }
    }
  ' "$HOSTFILE"
)

if [[ "${#HOSTS[@]}" -eq 0 ]]; then
  echo "No hosts found in $HOSTFILE" >&2
  exit 1
fi

NNODES="${NNODES:-${#HOSTS[@]}}"
MASTER_ADDR="${MASTER_ADDR:-${HOSTS[0]}}"

detect_gpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local count
    count="$(nvidia-smi -L 2>/dev/null | awk 'END { print NR }')"
    if [[ "$count" =~ ^[0-9]+$ && "$count" -gt 0 ]]; then
      echo "$count"
      return
    fi
  fi
  echo 1
}

NPROC_PER_NODE="${NPROC_PER_NODE:-${GPUS_PER_NODE:-}}"
if [[ -z "$NPROC_PER_NODE" ]]; then
  NPROC_PER_NODE="$(detect_gpus)"
fi

contains() {
  local needle="$1"
  shift

  local item
  for item in "$@"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

resolve_host() {
  local host="$1"

  if command -v getent >/dev/null 2>&1; then
    getent ahostsv4 "$host" 2>/dev/null | awk '{ print $1 }' || true
    getent hosts "$host" 2>/dev/null | awk '{ print $1 }' || true
  fi
}

local_identity_tokens() {
  hostname 2>/dev/null || true
  hostname -s 2>/dev/null || true
  hostname -f 2>/dev/null || true
  hostname -i 2>/dev/null || true
  hostname -I 2>/dev/null || true

  if command -v ip >/dev/null 2>&1; then
    ip -o -4 addr show scope global 2>/dev/null | awk '{ split($4, a, "/"); print a[1] }' || true
  fi
}

infer_node_rank() {
  local -a local_tokens
  mapfile -t local_tokens < <(local_identity_tokens | tr ' ' '\n' | awk 'NF && !seen[$0]++')

  local index host resolved
  for index in "${!HOSTS[@]}"; do
    host="${HOSTS[$index]}"

    if contains "$host" "${local_tokens[@]}"; then
      echo "$index"
      return 0
    fi

    while IFS= read -r resolved; do
      if [[ -n "$resolved" ]] && contains "$resolved" "${local_tokens[@]}"; then
        echo "$index"
        return 0
      fi
    done < <(resolve_host "$host" | awk 'NF && !seen[$0]++')
  done

  return 1
}

if [[ -z "${NODE_RANK:-}" ]]; then
  if ! NODE_RANK="$(infer_node_rank)"; then
    echo "Could not infer NODE_RANK from $HOSTFILE." >&2
    echo "Set NODE_RANK explicitly, for example: NODE_RANK=0 $0 $*" >&2
    exit 1
  fi
fi

if [[ "$USE_UV" == "0" ]]; then
  launcher=("${TORCHRUN_BIN:-torchrun}")
else
  launcher=("${UV_BIN:-uv}" run "${TORCHRUN_BIN:-torchrun}")
fi

cmd=(
  "${launcher[@]}"
  "--nnodes=$NNODES"
  "--nproc_per_node=$NPROC_PER_NODE"
  "--node_rank=$NODE_RANK"
  "--master_addr=$MASTER_ADDR"
  "--master_port=$MASTER_PORT"
  "$TRAIN_ENTRYPOINT"
  "$@"
)

echo "HOSTFILE=$HOSTFILE"
echo "HOSTS=${HOSTS[*]}"
echo "NNODES=$NNODES NPROC_PER_NODE=$NPROC_PER_NODE NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

exec "${cmd[@]}"
