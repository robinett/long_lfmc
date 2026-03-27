#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer"
session_name="${1:-long_lfmc_viewer}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required but was not found on PATH." >&2
    exit 1
fi

if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "tmux session ${session_name} already exists." >&2
    echo "Attach with: tmux attach -t ${session_name}" >&2
    exit 1
fi

tmux new-session -d -s "${session_name}"
tmux send-keys -t "${session_name}:0.0" "cd ${script_dir}" C-m
tmux send-keys -t "${session_name}:0.0" "bash ${script_dir}/run_viewer_api.sh" C-m
tmux split-window -h -t "${session_name}:0"
tmux send-keys -t "${session_name}:0.1" "cd ${script_dir}" C-m
tmux send-keys -t "${session_name}:0.1" "bash ${script_dir}/run_viewer_frontend.sh" C-m
tmux select-layout -t "${session_name}:0" even-horizontal

hostname_short="$(hostname -s)"

cat <<EOF
Started tmux session: ${session_name}

If you have not built the native-grid viewer assets yet, run this first in another shell:
bash ${script_dir}/run_viewer_build.sh

Attach with:
tmux attach -t ${session_name}

From your laptop, create the SSH tunnel with:
ssh -J ${USER}@sherlock.stanford.edu -L 4173:127.0.0.1:4173 ${USER}@${hostname_short}

Then open:
http://127.0.0.1:4173
EOF
