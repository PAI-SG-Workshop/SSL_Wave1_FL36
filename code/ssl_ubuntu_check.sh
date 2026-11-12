#!/usr/bin/env bash
# ssl_ubuntu_check.sh - READ-ONLY check of the RTX 4070 Ubuntu laptop for SSL Wave 1a/1b. Changes nothing.
#   bash ssl_ubuntu_check.sh 2>&1 | tee ~/SSL_Wave1/logs/ubuntu_check.txt
echo "=== ssl_ubuntu_check $(date '+%F %T')  host $(hostname)"

echo "--- GPU"
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,power.limit,clocks.sm,clocks.max.sm --format=csv
echo "compute processes on the GPU (empty = free):"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
echo "--- tmux sessions / python processes"
tmux ls 2>&1
pgrep -af python | grep -v pgrep | head -10

echo "--- machine"
lscpu | grep -m1 "Model name"; echo "threads $(nproc)"
free -g | head -2
df -h "$HOME" | tail -1

echo "--- existing work (vseg / OD) locations"
echo "PHYSICAL_AI_ROOT=$PHYSICAL_AI_ROOT"
find "$HOME" -maxdepth 6 -type d -name FashionLetters36 -not -path "*/SSL_Wave1/*" 2>/dev/null | head -5
find "$HOME" -maxdepth 6 -name fl36_count.json -not -path "*/SSL_Wave1/*" 2>/dev/null | head -3
find "$HOME" -maxdepth 6 -name train_image_centernet.py -not -path "*/SSL_Wave1/*" 2>/dev/null | head -5
find "$HOME" -maxdepth 8 -name "image_centernet_fl36_last.pth" 2>/dev/null | head -3

echo "--- conda envs and their torch"
CONDA_BASE=$(conda info --base 2>/dev/null)
echo "conda base: $CONDA_BASE"
for e in $(conda env list 2>/dev/null | awk '!/^#/ && NF {print $1}'); do
  printf "%-14s " "$e"
  conda run -n "$e" python -c "
import sys
try:
    import torch
    t = torch.__version__ + ' cuda ' + str(torch.version.cuda) + ' avail ' + str(torch.cuda.is_available())
except Exception:
    t = 'no torch'
mods = []
for m in ('torchvision', 'cv2', 'sklearn', 'scipy'):
    try:
        __import__(m); mods.append(m)
    except Exception:
        pass
print('py', sys.version.split()[0], '|', t, '|', ' '.join(mods))
" 2>/dev/null || echo "(python not runnable)"
done

echo "--- extracted SSL files"
ls -la "$HOME/SSL_Wave1" "$HOME/SSL_Wave1/datasets/ssl" "$HOME/SSL_Wave1/code/ssl" 2>&1 | head -40
python3 - <<'EOF'
import hashlib, json, pathlib
d = pathlib.Path.home() / "SSL_Wave1" / "datasets" / "ssl"
try:
    info = json.loads((d / "ssl_data.json").read_text())
except Exception as e:
    print("ssl_data.json not readable:", e); raise SystemExit
for name in ("crop36_train.npz", "crop36_test.npz", "fl36_train_gray.npy"):
    p = d / name
    if not p.is_file():
        print(f"{name}: MISSING"); continue
    h = hashlib.md5()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    want = info[name]["md5"]
    print(f"{name}: {'MATCH' if h.hexdigest() == want else 'DIFFERENT'}  ({h.hexdigest()[:8]} vs Windows {want[:8]})")
EOF
echo "=== done"
