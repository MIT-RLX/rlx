#!/usr/bin/env bash
# RLX — XDNA NPU direct-path exec-hang diagnostic.
#
# Runs the feature=direct `direct_gemm` example (which currently hangs at the
# completion fence) while capturing the kernel/firmware state that a normal user
# can't see. Run it AS YOUR NORMAL USER (it holds the render-group fd to the NPU)
# — it uses `sudo` internally for the root-only bits, so it'll prompt for your
# password once.
#
#   ./npu_direct_debug.sh
#
# Paste the whole output back. The single most important line is "DYNDBG:" —
# ENABLED means the driver will narrate the failure into dmesg (step 5).
set -u

# ---- resolve the invoking user's home (works even under `sudo`) ---------------
REAL_USER=${SUDO_USER:-$USER}
REAL_HOME=$(getent passwd "$REAL_USER" 2>/dev/null | cut -d: -f6)
REAL_HOME=${REAL_HOME:-$HOME}

# ---- config (override via env) ------------------------------------------------
# Auto-find the rlx checkout if not given (handles run-as-user or run-as-root).
if [ -z "${RLX:-}" ]; then
  for d in "$REAL_HOME/rlx" "$HOME/rlx" /home/*/rlx; do
    [ -d "$d" ] && RLX="$d" && break
  done
fi
RLX=${RLX:-$REAL_HOME/rlx}
B=${B:-$REAL_HOME/mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/build}
XCLBIN=${XCLBIN:-$B/final_512x512x512_32x32x32_4c.xclbin}
INSTS=${INSTS:-$B/insts_512x512x512_32x32x32_4c.bin}
M=${M:-512}; K=${K:-512}; N=${N:-512}; ITERS=${ITERS:-1}
SNAP_DELAY=${SNAP_DELAY:-4}     # seconds to wait before snapshotting live state
RUN_TIMEOUT=${RUN_TIMEOUT:-20}  # hard cap on the hung run

echo "==================================================================="
echo "RLX XDNA direct-path exec-hang diagnostic   $(date '+%F %T')"
echo "  xclbin=$XCLBIN"
echo "  insts =$INSTS   MxKxN=${M}x${K}x${N}"
echo "==================================================================="

# ---- cache sudo creds up front (one prompt) -----------------------------------
echo "[*] Caching sudo credentials (enter password if prompted)..."
sudo -v || { echo "!! sudo failed — cannot capture root-only state"; exit 1; }

# ---- 0) environment facts -----------------------------------------------------
echo; echo "===== 0. environment ====="
echo -n "lockdown: "; sudo cat /sys/kernel/security/lockdown 2>/dev/null || echo "(n/a)"
echo -n "amdxdna params: "; ls /sys/module/amdxdna/parameters/ 2>/dev/null | tr '\n' ' '; echo
echo -n "iommu group type: "; cat /sys/bus/pci/devices/0000:c6:00.1/iommu_group/type 2>/dev/null

# ---- 1) THE key test: enable the driver's own tracing -------------------------
echo; echo "===== 1. enable amdxdna dynamic_debug (lockdown vs perms test) ====="
if echo 'module amdxdna +p' | sudo tee /sys/kernel/debug/dynamic_debug/control >/dev/null 2>&1; then
  echo "DYNDBG: ENABLED ✅  (driver will narrate into dmesg — see step 5)"
  DYNDBG=1
else
  echo "DYNDBG: BLOCKED (Secure Boot lockdown) — dmesg will only have ERROR lines"
  DYNDBG=0
fi

# ---- find (or build) the example, robustly ------------------------------------
echo; echo "===== finding direct_gemm (RLX=$RLX) ====="
find_bin() { find "$RLX/target" -name direct_gemm -type f -perm -u+x 2>/dev/null | head -1; }
BIN=$(find_bin)
if [ -z "$BIN" ]; then
  echo "[*] not found — building (as $REAL_USER) ..."
  CARGO_ENV="$REAL_HOME/.cargo/env"
  sudo -u "$REAL_USER" bash -lc "source '$CARGO_ENV' 2>/dev/null; cd '$RLX' && cargo build -q -p rlx-xdna --features direct --example direct_gemm" 2>&1 | tail -15
  BIN=$(find_bin)
fi
[ -n "$BIN" ] && [ -x "$BIN" ] || { echo "!! direct_gemm not found under $RLX/target (set RLX=... to your checkout)"; exit 1; }
echo "using: $BIN"

# ---- 2) clear kernel log ------------------------------------------------------
echo; echo "===== 2. clear dmesg ====="
sudo dmesg -C && echo "cleared"

# ---- 3) launch the hanging run (as this user) ---------------------------------
echo; echo "===== 3. run direct_gemm (expected to hang at the fence) ====="
source "$HOME/.cargo/env" 2>/dev/null
XCLBIN="$XCLBIN" INSTS="$INSTS" M="$M" K="$K" N="$N" ITERS="$ITERS" \
  timeout "$RUN_TIMEOUT" "$BIN" > /tmp/rlx_direct_gemm.log 2>&1 &
DG=$!
echo "launched pid=$DG; sleeping ${SNAP_DELAY}s to catch it mid-hang..."

# ---- 4) snapshot LIVE driver/firmware state while it's stuck ------------------
sleep "$SNAP_DELAY"
echo; echo "===== 4. LIVE amdxdna debugfs (job stuck) ====="
sudo sh -c '
  base=/sys/kernel/debug/accel
  if [ ! -d "$base" ]; then echo "(no $base)"; exit 0; fi
  find "$base" -type f 2>/dev/null | while read -r f; do
    echo "--- $f ---"; head -c 800 "$f" 2>/dev/null; echo
  done'

# ---- wait for the run to finish/timeout ---------------------------------------
wait "$DG" 2>/dev/null
echo; echo "===== 3b. direct_gemm output ====="
cat /tmp/rlx_direct_gemm.log

# ---- 5) dump everything the driver logged ------------------------------------
echo; echo "===== 5. dmesg for this run ====="
sudo dmesg | grep -iE 'amdxdna|aie2|aie |mailbox|drm_sched|iommu|sva|pasid' | tail -120

echo; echo "===== done. dyndbg=$DYNDBG — paste all output above back. ====="
