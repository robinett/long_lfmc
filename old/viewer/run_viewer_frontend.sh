#!/usr/bin/env bash

set -eo pipefail

script_dir="/home/users/trobinet/long_lfmc/old/viewer"
frontend_dir="${script_dir}/frontend"
node_bin_dir="/share/software/user/open/nodejs/25.3.0/bin"
node_lib_dir="/share/software/user/open/nodejs/25.3.0/lib"
gcc_lib64_dir="/share/software/user/open/gcc/14.2.0/lib64"
gcc_lib_dir="/share/software/user/open/gcc/14.2.0/lib"
gcc_libgcc_dir="/share/software/user/open/gcc/14.2.0/lib/gcc/x86_64-pc-linux-gnu"

cd "${frontend_dir}"

source ~/.bashrc || true
set -u
export PATH="${node_bin_dir}:${PATH}"
export LD_LIBRARY_PATH="${node_lib_dir}:${gcc_lib64_dir}:${gcc_libgcc_dir}:${gcc_lib_dir}:${LD_LIBRARY_PATH:-}"

if ! command -v node >/dev/null 2>&1; then
    echo "Could not find node on PATH after adding ${node_bin_dir}" >&2
    exit 1
fi

if ! command -v npm >/dev/null 2>&1; then
    echo "Could not find npm on PATH after adding ${node_bin_dir}" >&2
    exit 1
fi

if [[ ! -d "${frontend_dir}/node_modules" ]] || [[ "${frontend_dir}/package.json" -nt "${frontend_dir}/node_modules" ]] || [[ "${frontend_dir}/package-lock.json" -nt "${frontend_dir}/node_modules" ]]; then
    echo "Installing frontend dependencies in ${frontend_dir}"
    npm install
fi

npm run dev -- --host 127.0.0.1 --port 4173
