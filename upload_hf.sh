#!/bin/bash
# ============================================================
# One-click HuggingFace upload script
# Run after server reboot:  bash /root/dfu/upload_hf.sh
# ============================================================
# Prerequisites:
#   - /root/.proxy/sing-box (binary)
#   - /root/.proxy/config.json (VMess WS config)
#   - HF token in ~/.cache/huggingface/token
# ============================================================

set -e

echo "=== Step 1: Start proxy ==="
/root/.proxy/sing-box run -c /root/.proxy/config.json &
sleep 3

echo "=== Step 2: Test proxy ==="
HTTP_CODE=$(curl -x http://127.0.0.1:7890 -s --connect-timeout 10 \
  https://huggingface.co -o /dev/null -w "%{http_code}")
if [ "$HTTP_CODE" != "200" ]; then
  echo "ERROR: Proxy not working (HTTP $HTTP_CODE)"
  exit 1
fi
echo "Proxy OK (HTTP $HTTP_CODE)"

echo "=== Step 3: Upload models/ ==="
python3 -c "
import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import HfApi
api = HfApi()
print('Uploading models/...')
api.upload_folder('/root/dfu/models', 'models', 'cl-666/dfu-project', repo_type='model')
print('models/ Done!')
"

echo "=== Step 4: Upload data/ ==="
python3 -c "
import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'
os.environ['HF_HUB_DISABLE_XET'] = '1'
from huggingface_hub import HfApi
api = HfApi()
print('Uploading data/...')
api.upload_folder('/root/dfu/data', 'data', 'cl-666/dfu-project', repo_type='model')
print('data/ Done!')
"

echo "=== Done! ==="
echo "https://huggingface.co/cl-666/dfu-project"
