#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p .references
fetch() {
    local name="$1" url="$2" revision="$3"
    if [ ! -d ".references/$name/.git" ]; then
        git init ".references/$name"
        git -C ".references/$name" remote add origin "$url"
        git -C ".references/$name" fetch --depth 1 --filter=blob:none origin "$revision"
        git -C ".references/$name" checkout --detach FETCH_HEAD
    else
        echo "$name already exists; leaving its checkout untouched"
        git -C ".references/$name" rev-parse HEAD
    fi
}
fetch vllm https://github.com/vllm-project/vllm.git abf98672ce7e822701f7a062bf240efb2c9900d1
fetch sglang https://github.com/sgl-project/sglang.git a73010b648819b4125458888956ce697a0a6274f
fetch nano-vllm https://github.com/GeeeekExplorer/nano-vllm.git bb823b3e06983d71485a8e1f23715ebd87d98ef8
fetch Quest https://github.com/mit-han-lab/Quest.git 01c1623bf9395009520874e989e29f683203b357
