#!/usr/bin/env bash

set -euo pipefail

script_dir="/home/users/trobinet/long_lfmc/lfmc_model/scripts/viewer_3857"
session_name="${1:-long_lfmc_viewer_3857}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required but was not found on PATH." >&2
    exit 1
fi

if tmux has-session -t "${session_name}" 2>/dev/null; then
    echo "Restarting existing tmux session ${session_name}."
    tmux kill-session -t "${session_name}"
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

If you have not built the 3857 viewer dataset and assets yet, run these first in another shell:
bash ${script_dir}/run_viewer_build_dataset.sh
bash ${script_dir}/run_viewer_build.sh

Attach with:
tmux attach -t ${session_name}

From your laptop, create the SSH tunnel with:
ssh -J ${USER}@sherlock.stanford.edu -L 4174:127.0.0.1:4174 ${USER}@${hostname_short}

Then open:
http://127.0.0.1:4174
EOF
