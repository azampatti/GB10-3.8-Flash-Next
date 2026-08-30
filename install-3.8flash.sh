#!/usr/bin/env bash
# ============================================================================
#  Qwen3.8-Flash-Next on ONE DGX Spark (GB10)  --  reproducible setup
#
#  Produces ~/launch-3.8flash.sh, which serves the model at ~40 tok/s single
#  stream (Q&A 35 / Code 38 / JSON 44 / Math 41 / LongCode 43) with a 91/100
#  tool-eval-bench score.
#
#  What this configuration is:
#    * RadixArk NVFP4 checkpoint (experts NVFP4, everything else BF16 on disk)
#    * flashnext-one-spark patches -- REQUIRED on sm_121, upstream corrupts
#    * HashK R=4 n-gram table: 51.2 GB -> 12.8 GB, built locally (~7 min)
#    * FP8 cast of the dense path at load time: +27% decode, no quality cost
#    * NEXTN speculative decoding depth 3 (worth ~1.8x on its own)
#
#  Requires: DGX Spark / GB10, Docker + NVIDIA Container Toolkit, ~40 GB free
#  disk, and the checkpoint ALREADY in your HuggingFace cache.
# ============================================================================
set -uo pipefail

MODEL_ID="RadixArk/Qwen3.8-Flash-Next-NVFP4"
IMAGE="lmsysorg/sglang:qwen38flashnext"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
SNAP_DIR="$HF_CACHE/hub/models--RadixArk--Qwen3.8-Flash-Next-NVFP4"
WORK="${WORK:-$HOME/qwen38-flash-spark}"
LAUNCHER="$HOME/launch-3.8flash.sh"

G=$'\033[32m'; R=$'\033[31m'; N=$'\033[0m'
STEP=0; NSTEPS=7
T_START=$(date +%s)
# inline phase: "  [n/7] Now doing... OK -- detail"
step(){  STEP=$((STEP+1)); printf '  [%d/%d] %s' "$STEP" "$NSTEPS" "$1"; }
ok(){    printf ' %sOK%s%s\n' "$G" "$N" "${1:+  --  $1}"; }
# phase that prints its own output: banner, output, then a standalone OK
stepn(){ STEP=$((STEP+1)); printf '  [%d/%d] %s\n' "$STEP" "$NSTEPS" "$1"; }
okn(){   printf '         %sOK%s%s\n' "$G" "$N" "${1:+  --  $1}"; }
die(){   printf '\n  %sFAIL%s %s\n' "$R" "$N" "$1" >&2; exit 1; }

cat <<'HEAD'
=== Qwen3.8-Flash-Next on one DGX Spark -- setup v1.1 ===

  phase                                          approx time
  ------------------------------------------------------------
  1  verify checkpoint in HuggingFace cache           ~20 s
  2  verify docker, GPU and disk                      ~40 s
  3  pull container image (~14 GB)               0 - 15 min   (skipped if present)
  4  clone runtime patches                            ~10 s
  5  apply FP8 dense cast                              ~1 s
  6  build HashK n-gram table on GPU                 ~6 min   (skipped if present)
  7  write the launcher                                ~1 s
  ------------------------------------------------------------
  TOTAL                                          ~7 - 25 min

  Then  ~/launch-3.8flash.sh  boots the server in ~10 min.

HEAD

# ---- 1. the model must already be in the HF cache -------------------------
step "Now checking for the model checkpoint (135 GB scan)..."
[ -d "$SNAP_DIR" ] || die "$MODEL_ID not in $HF_CACHE.
       Fetch it first (135 GB):
         huggingface-cli download $MODEL_ID"

NST=$(find "$SNAP_DIR" -name '*.safetensors' | wc -l)
BYTES=$(du -sbL "$SNAP_DIR" 2>/dev/null | cut -f1)
GB=$(awk "BEGIN{printf \"%.1f\", $BYTES/1e9}")
[ "$NST" -eq 206 ] || die "checkpoint incomplete: $NST/206 safetensors (${GB} GB). Re-run the download."
awk "BEGIN{exit !($BYTES > 134.5e9)}" || die "checkpoint incomplete: ${GB} GB, expected ~135.2. Re-run the download."
[ "$(find "$SNAP_DIR" -name '*.incomplete' | wc -l)" -eq 0 ] || die "download still in progress (.incomplete files present)"
ok "206/206 safetensors, ${GB} GB"

# ---- 2. environment -------------------------------------------------------
step "Now checking docker, GPU and free disk..."
command -v docker >/dev/null || die "docker not found"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon"
CAP=$(docker run --rm --gpus all --entrypoint python3 "$IMAGE" -c \
      "import torch;print('%d%d'%torch.cuda.get_device_capability(0))" 2>/dev/null | tail -1)
FREE=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$FREE" -ge 40 ] || die "need ~40 GB free for the HashK artifact, have ${FREE} GB"
ok "sm_$CAP, ${FREE} GB free"
[ "$CAP" = "121" ] || echo "         NOTE: sm_$CAP, expected sm_121 (GB10). Untested on this arch."

# ---- 3. container image ---------------------------------------------------
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  step "Now checking the container image..."; ok "already present, skipping the 14 GB pull"
else
  stepn "Now pulling the container image, ~14 GB -- this is the long one..."
  docker pull "$IMAGE" || die "image pull failed"
  okn "$IMAGE"
fi

# ---- 4. patched runtime ---------------------------------------------------
step "Now cloning the sm_121 runtime patches..."
mkdir -p "$WORK" && cd "$WORK"
if [ ! -d flashnext-one-spark ]; then
  git clone -q https://github.com/deathbyorderfill/flashnext-one-spark.git \
    || die "git clone failed"
fi
REPO="$WORK/flashnext-one-spark"
chmod +x "$REPO"/launch.sh "$REPO"/tools/*.sh 2>/dev/null || true
for p in qwen4_exp_nvfp4.py flash_fwd.py qwen_sparse_attn_backend.py sparse_attn.py; do
  [ -f "$REPO/patches/$p" ] || die "missing patch $p -- clone is incomplete"
done
ok "4 patch files in $WORK"

# ---- 5. FP8 dense cast ----------------------------------------------------
step "Now applying the runtime patches (FP8 cast, PLE-off mode)..."
cat > "$WORK/patch_fp8.py" <<'Q4XEOF'
#!/usr/bin/env python3
"""Step 1: post-load FP8 cast for lm_head (and, later, other dense modules).

Anchored edit against flashnext-one-spark/patches/qwen4_exp_nvfp4.py, following
that repo's generator convention. Idempotent; keeps a .orig backup.

Enable at runtime with  Q4X_FP8=lm_head  in the container env. Unset = no-op,
so the patched file is safe to leave bind-mounted for a baseline re-run.

Why lm_head: NEXTN reads it 4x per iteration (3 draft steps + verify).
248320 x 2560 = 0.636B params, 1.27 GB BF16 -> 0.64 GB FP8, saving 2.5 GB/iteration.
"""
import os, pathlib, shutil, sys

TARGET = pathlib.Path(os.environ["Q4X_TARGET"])

BLOCK = '''

# ==== Q4X step 1: post-load FP8 cast of dense weights (env Q4X_FP8) ====
# Weight-only FP8 e4m3, per-output-channel absmax. No calibration data needed.
# Layout note: sglang's apply_fp8_linear wants weight as (K, N) with
# weight_scale.numel() == weight.shape[1]; nn.Linear stores (N, K). We transpose.
# Getting that backwards produces silent garbage, not an error.


class Q4XFp8LinearMethod:
    """Minimal weight-only FP8 apply(). The class NAME matters: logits_processor
    skips quant methods listed in _UNQUANTIZED_LM_HEAD_METHODS, and this is not
    one of them, so lm_head routes through apply() instead of the bf16 matmul."""

    def __init__(self, cutlass_ok: bool):
        self.cutlass_ok = cutlass_ok

    def process_weights_after_loading(self, layer):
        # Required: sglang's loader calls this on every module carrying a
        # quant_method. Our weights are already final -- cast in load_weights.
        return

    def create_weights(self, *args, **kwargs):  # never used; we attach post-load
        raise NotImplementedError("Q4XFp8LinearMethod is applied after loading")

    def apply(self, layer, x, bias=None):
        from sglang.srt.layers.quantization.fp8_utils import apply_fp8_linear

        return apply_fp8_linear(
            input=x,
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            input_scale=None,
            bias=bias,
            cutlass_fp8_supported=self.cutlass_ok,
            use_per_token_if_dynamic=True,
        )


def _q4x_fp8_targets():
    import os

    return {t.strip() for t in os.environ.get("Q4X_FP8", "").split(",") if t.strip()}


def _q4x_quantize_to_fp8_t(w: torch.Tensor, chunk: int = 8192):
    """(N, K) bf16 -> ((K, N) COLUMN-MAJOR fp8, (N, 1) fp32 scale).

    Layout is load-bearing: the cutlass fp8 GEMM rejects a row-major B with
    "mat_b must be a column major tensor". Build a contiguous (N, K) buffer and
    return its .t() view -- strides (1, K), i.e. column major -- which is exactly
    what sglang's own Fp8LinearMethod does (`Parameter(weight.t())`).
    Chunked over N so we never hold a second bf16 copy.
    """
    N, K = w.shape
    buf = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=w.device)
    scales = torch.empty((N, 1), dtype=torch.float32, device=w.device)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        blk = w[i:j].to(torch.float32)
        s = blk.abs().amax(dim=1, keepdim=True).clamp_min(1e-12) / 448.0
        buf[i:j] = (blk / s).clamp_(-448.0, 448.0).to(torch.float8_e4m3fn)
        scales[i:j] = s
        del blk, s
    out = buf.t()                      # (K, N) column-major view, no copy
    assert out.stride() == (1, K), f"expected column-major B, got {out.stride()}"
    return out, scales


def _q4x_apply_fp8(model):
    targets = _q4x_fp8_targets()
    if not targets:
        return
    from sglang.srt.layers.quantization.fp8_utils import cutlass_fp8_supported

    ok = cutlass_fp8_supported()

    if "lm_head" in targets:
        lm = getattr(model, "lm_head", None)
        if lm is None or not hasattr(lm, "weight"):
            logger.warning("Q4X_FP8: lm_head requested but not found -- SKIPPED")
        elif lm.weight.dtype == torch.float8_e4m3fn:
            logger.info("Q4X_FP8: lm_head already fp8, nothing to do")
        else:
            before = lm.weight.numel() * lm.weight.element_size()
            q, s = _q4x_quantize_to_fp8_t(lm.weight.data)
            lm.weight = torch.nn.Parameter(q, requires_grad=False)
            lm.weight_scale = torch.nn.Parameter(s, requires_grad=False)
            lm.input_scale = None
            lm.quant_method = Q4XFp8LinearMethod(ok)
            torch.cuda.empty_cache()
            after = q.numel() * q.element_size()
            # LOUD, because a silent no-op here is the whole failure mode: the
            # server would serve fine at baseline speed and we would misread it
            # as "fp8 did not help".
            logger.info(
                "Q4X_FP8 ACTIVE: lm_head %s -> fp8 %s  (%.2f -> %.2f GB, cutlass=%s)",
                tuple(lm.weight_scale.shape[:1]) + (q.shape[0],),
                tuple(q.shape), before / 1e9, after / 1e9, ok,
            )

    # --- hyper-connections: plain nn.Linear, no quant_method to swap ---
    # The compute takes raw weight TENSORS, so we attach packed fp8 tuples to the
    # GatedResidual module and let the patched hyperconnection.py pick them up.
    # ⚠ This DISABLES the fused Triton hc_mix kernel, whose guard requires
    # w.dtype == activation dtype. Expected to be a regression; measured on purpose.
    if "hyper_connection" in targets:
        n_hc = 0
        for name, mod in model.named_modules():
            if not hasattr(mod, "input_mix_weight_down"):
                continue
            try:
                mod._q4x_down = _q4x_quantize_to_fp8_t(mod.input_mix_weight_down.weight.data)
                mod._q4x_up = _q4x_quantize_to_fp8_t(mod.input_mix_weight_up.weight.data)
                if getattr(mod, "block_inject_weight", None) is not None:
                    mod._q4x_inject = _q4x_quantize_to_fp8_t(mod.block_inject_weight.weight.data)
                n_hc += 1
            except Exception as e:
                logger.warning("Q4X_FP8: hyper_connection %s FAILED (%s)", name, e)
        torch.cuda.empty_cache()
        if n_hc:
            logger.info(
                "Q4X_FP8 ACTIVE: %d hyper-connection blocks -> fp8 "
                "(fused Triton hc_mix is now BYPASSED for these)", n_hc,
            )
        else:
            logger.warning("Q4X_FP8: target 'hyper_connection' matched NO modules")

    # --- generic dense sweep: any sglang Linear whose qualified name matches ---
    # Selected by NAME MATCH over named_modules() rather than a hardcoded list,
    # and every conversion is logged, so a target that silently matches nothing
    # is visible instead of being scored as "fp8 did not help".
    module_targets = targets - {"lm_head", "hyper_connection"}
    if module_targets:
        converted, skipped, total_before, total_after = [], [], 0, 0
        for name, mod in model.named_modules():
            if not any(t in name for t in module_targets):
                continue
            w = getattr(mod, "weight", None)
            if w is None or w.dim() != 2 or w.dtype != torch.bfloat16:
                continue                      # norms (1-D), conv1d (3-D), already-quantized
            qm = getattr(mod, "quant_method", None)
            if qm is None:
                # plain nn.Linear (e.g. the hyper-connections): no quant_method to
                # swap, forward goes straight to F.linear. Needs a different
                # mechanism -- report it rather than pretending it was done.
                skipped.append((name, "no quant_method (plain nn.Linear)"))
                continue
            if type(qm).__name__ not in ("UnquantizedLinearMethod", "UnquantizedEmbeddingMethod"):
                skipped.append((name, f"already {type(qm).__name__}"))
                continue
            before = w.numel() * w.element_size()
            q, s = _q4x_quantize_to_fp8_t(w.data)
            mod.weight = torch.nn.Parameter(q, requires_grad=False)
            mod.weight_scale = torch.nn.Parameter(s, requires_grad=False)
            mod.input_scale = None
            mod.quant_method = Q4XFp8LinearMethod(ok)
            total_before += before
            total_after += q.numel() * q.element_size()
            converted.append(name)
            # Free eagerly and compact periodically: 200+ alloc/free pairs
            # otherwise fragment the caching allocator and SHRINK the KV pool,
            # which is sized from free memory after loading.
            del w
            if len(converted) % 16 == 0:
                torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        logger.info(
            "Q4X_FP8 ACTIVE: %d modules -> fp8 (%.2f -> %.2f GB, saved %.2f GB) for targets %s",
            len(converted), total_before / 1e9, total_after / 1e9,
            (total_before - total_after) / 1e9, sorted(module_targets),
        )
        for t_ in sorted(module_targets):
            n = sum(1 for c in converted if t_ in c)
            if n == 0:
                logger.warning("Q4X_FP8: target %r matched NO modules -- check the name", t_)
            else:
                logger.info("Q4X_FP8:   %-20s %4d modules", t_, n)
        for name, why in skipped[:12]:
            logger.warning("Q4X_FP8: SKIPPED %s (%s)", name, why)
        if len(skipped) > 12:
            logger.warning("Q4X_FP8: ... and %d more skipped", len(skipped) - 12)


# ==== end Q4X step 1 ====
'''

ANCHOR_HOOK = """        for module in self.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.finalize_fused_in_proj()

        return loaded_params"""

REPLACE_HOOK = """        for module in self.modules():
            if isinstance(module, Qwen3_5GatedDeltaNet):
                module.finalize_fused_in_proj()

        _q4x_apply_fp8(self)

        return loaded_params"""

ANCHOR_BLOCK = "\nEntryClass = [Qwen4ExpForConditionalGeneration]"


def main():
    src = TARGET.read_text()
    if "Q4X step 1" in src:
        print("already patched; nothing to do")
        return 0
    if ANCHOR_HOOK not in src:
        print("FATAL: hook anchor not found -- refusing to guess", file=sys.stderr)
        return 1
    if ANCHOR_BLOCK not in src:
        print("FATAL: block anchor not found -- refusing to guess", file=sys.stderr)
        return 1

    backup = TARGET.with_suffix(".py.orig")
    if not backup.exists():
        shutil.copy2(TARGET, backup)
        print(f"backup -> {backup.name}")

    out = src.replace(ANCHOR_HOOK, REPLACE_HOOK, 1)
    assert out != src
    out = out.replace(ANCHOR_BLOCK, BLOCK + ANCHOR_BLOCK, 1)
    TARGET.write_text(out)

    import ast
    ast.parse(out)          # refuse to ship a file that will not import
    print(f"patched {TARGET.name}: +{len(out) - len(src)} bytes, syntax OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

Q4XEOF
Q4X_TARGET="$REPO/patches/qwen4_exp_nvfp4.py" python3 "$WORK/patch_fp8.py" \
  || die "FP8 patch failed to apply"
cat > "$WORK/patch_ple_off.py" <<'PLEEOF'
#!/usr/bin/env python3
"""Step 2: PLE_MODE=off -- drop the n-gram table entirely.

Anchored edit against flashnext-one-spark/patches/qwen4_exp_nvfp4.py, mirroring
the existing hashk path (which already deletes the embedding parameter, so the
mechanism is proven -- we only add "and load nothing in its place").

Three hooks:
  1. __init__      delete the weight parameter, flag the module
  2. gather        return zeros instead of looking anything up
  3. weight loader consume + discard the 10 model-plefp8-* shards, never allocate

The injection is `hidden_states = hidden_states + self.ple(...)`, so zeros make it
an exact no-op. RMSNorm(0) = 0/sqrt(eps) = 0, no NaN.

Frees 12.8 GB vs hashk (51.2 GB vs an unmodified checkpoint).

  --check   write to a temp copy and validate only; do NOT touch the live file
"""
import argparse, ast, os, pathlib, shutil, sys

TARGET = pathlib.Path(os.environ["Q4X_TARGET"])

HELPER = '''

def _ple_off() -> bool:
    """PLE_MODE=off: no n-gram table at all. Not a speed lever -- the table is
    ~5 KB/token -- but it frees 12.8 GB and answers what the table is worth."""
    import os

    return os.environ.get("SGLANG_QWEN4_PLE_OFF", "0") == "1"

'''
HELPER_ANCHOR = "\ndef _nvfp4_ple_enabled() -> bool:"

INIT_ANCHOR = """        if _hashk_path():
            if getattr(config, "ple_offload_embedding", False):
                config.ple_offload_embedding = False"""
INIT_NEW = """        if _ple_off():
            if getattr(config, "ple_offload_embedding", False):
                config.ple_offload_embedding = False
            del self.ngram_embedding._parameters["weight"]
            torch.cuda.empty_cache()
            self.ngram_embedding.ple_off = True
            logger.info(
                "PLE OFF: n-gram embedding deleted, injection zeroed "
                "(frees 12.8 GB vs hashk, 51.2 GB vs the raw checkpoint)"
            )
        elif _hashk_path():
            if getattr(config, "ple_offload_embedding", False):
                config.ple_offload_embedding = False"""

GATHER_ANCHOR = """        if getattr(self.ngram_embedding, "hashk_mode", False):
            embeddings = _hashk_gather(self.ngram_embedding, lookup_ids)"""
GATHER_NEW = """        if getattr(self.ngram_embedding, "ple_off", False):
            # [T, heads] -> [T, heads, head_dim] of zeros. The PLE output is added
            # to the residual stream, so zeros make the whole injection a no-op.
            embeddings = torch.zeros(
                (*lookup_ids.shape, self.head_dim_per_ngram),
                dtype=torch.bfloat16,
                device=lookup_ids.device,
            )
        elif getattr(self.ngram_embedding, "hashk_mode", False):
            embeddings = _hashk_gather(self.ngram_embedding, lookup_ids)"""

LOADER_ANCHOR = """            if getattr(emb, "hashk_mode", False):
                loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
                return True"""
LOADER_NEW = """            if getattr(emb, "ple_off", False) or getattr(emb, "hashk_mode", False):
                # Consume and discard: the 10 model-plefp8-* shards are read off
                # disk and dropped, never allocated.
                loaded_shard_params.add(f"{mod_prefix}.ngram_embedding.weight")
                return True"""

EDITS = [("helper", HELPER_ANCHOR, HELPER + HELPER_ANCHOR),
         ("init", INIT_ANCHOR, INIT_NEW),
         ("gather", GATHER_ANCHOR, GATHER_NEW),
         ("loader", LOADER_ANCHOR, LOADER_NEW)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true",
                    help="validate against a temp copy; leave the live file alone")
    args = ap.parse_args()

    src = TARGET.read_text()
    if "_ple_off" in src:
        print("already patched; nothing to do")
        return 0

    out = src
    for name, anchor, repl in EDITS:
        if anchor not in out:
            print(f"FATAL: anchor {name!r} not found -- refusing to guess", file=sys.stderr)
            return 1
        out = out.replace(anchor, repl, 1)
        print(f"  applied: {name}")

    ast.parse(out)
    print("syntax OK")

    if args.check:
        tmp = pathlib.Path("/tmp/qwen4_exp_ple_off_preview.py")
        tmp.write_text(out)
        print(f"--check: wrote preview to {tmp} (+{len(out) - len(src)} bytes)")
        print("LIVE FILE UNTOUCHED -- rerun without --check when the server is down")
        return 0

    backup = TARGET.with_suffix(".py.prestep2")
    if not backup.exists():
        shutil.copy2(TARGET, backup)
        print(f"backup -> {backup.name}")
    TARGET.write_text(out)
    print(f"patched {TARGET.name}: +{len(out) - len(src)} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

PLEEOF
Q4X_TARGET="$REPO/patches/qwen4_exp_nvfp4.py" python3 "$WORK/patch_ple_off.py" \
  || die "PLE-off patch failed to apply"
ok "FP8 cast + PLE-off mode available"

# ---- 6. HashK R=4 artifact ------------------------------------------------
if [ ! -f "$REPO/ple_hashk_R4.pt" ]; then
  stepn "Now building the HashK n-gram table on GPU, ~6 min (128 shards, progress below)..."
  docker run --rm --privileged --gpus all \
    -v "$HF_CACHE":/root/.cache/huggingface -v "$REPO":/out \
    --entrypoint python3 "$IMAGE" /out/tools/build_hashk_ple.py \
    || die "HashK build failed"
  okn "51.2 GB table -> $(du -h "$REPO/ple_hashk_R4.pt" | cut -f1)"
else
  step "Now checking the HashK n-gram table..."; ok "already built, skipping the 6 min build"
fi
[ -f "$REPO/ple_hashk_R4.pt" ] || die "HashK artifact missing after build"

# ---- 7. the launcher ------------------------------------------------------
step "Now writing the launcher..."
cat > "$LAUNCHER" <<'LAUNCHEOF'
#!/usr/bin/env bash
# Qwen3.8-Flash-Next on one DGX Spark. Generated by install-3.8flash.sh.
#   ~40 tok/s single stream, 91/100 tool-eval-bench, 200k context.
# Env overrides: PORT, CTX, MEM_FRACTION, THINKING (off|low|medium|xhigh)
set -uo pipefail
REPO="@@REPO@@"
HF_CACHE="@@HF_CACHE@@"
IMAGE="@@IMAGE@@"
MODEL_ID="@@MODEL_ID@@"
PORT="${PORT:-30000}"
MEM_FRACTION="${MEM_FRACTION:-0.90}"
THINKING="${THINKING:-medium}"

# ---- mode -----------------------------------------------------------------
#   1  n-gram table at HashK R=4, 200k context   (default; best measured quality)
#   2  no n-gram table, 256k context             (8.6x the KV cache, ~2 pts lower
#                                                 on tool-eval -- within run noise)
MODE="${MODE:-}"
if [ -z "$MODE" ]; then
  if [ -t 0 ]; then
    echo "  Which configuration?"
    echo "    1) R=4 n-gram table  --  200k context   (default, best quality)"
    echo "    2) PLE OFF           --  256k context, 4+ concurrent long sessions"
    printf "  Choose [1]: "; read -r MODE
  fi
  MODE="${MODE:-1}"
fi

case "$MODE" in
  1) PLE_ENV=(-e "SGLANG_QWEN4_PLE_HASHK=/patches/ple_hashk_R4.pt")
     CTX="${CTX:-200000}"
     [ -f "$REPO/ple_hashk_R4.pt" ] || { echo "HashK artifact missing -- re-run install-3.8flash.sh" >&2; exit 1; }
     echo "  mode 1: HashK R=4 table, ${CTX} context" ;;
  2) PLE_ENV=(-e "SGLANG_QWEN4_PLE_OFF=1")
     CTX="${CTX:-256000}"
     echo "  mode 2: PLE OFF (no n-gram table), ${CTX} context" ;;
  *) echo "unknown MODE=$MODE (use 1 or 2)" >&2; exit 1 ;;
esac

# ---- memory advisory ------------------------------------------------------
# Reference: on the box this was tuned on, with nothing else loaded, there was
# 118 GiB available RAM and 32 GiB swap. The server holds ~91 GB of weights
# inside a ~108 GB static budget (mem-fraction 0.90), so a short shortfall here
# means a failed boot or a hard lock, not just a slow one.
MIN_RAM_GIB=105
MIN_SWAP_GIB=16
RAM_GIB=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
SWAP_GIB=$(awk '/SwapTotal/{printf "%d", $2/1048576}' /proc/meminfo)
if [ "${RAM_GIB:-0}" -lt "$MIN_RAM_GIB" ] || [ "${SWAP_GIB:-0}" -lt "$MIN_SWAP_GIB" ]; then
  Y=$'\033[33m'; N=$'\033[0m'
  echo "${Y}"
  echo "  ============================ WARNING ============================"
  echo "  Low memory. This configuration needs about ${MIN_RAM_GIB} GiB of AVAILABLE"
  echo "  RAM and ${MIN_SWAP_GIB} GiB of swap. You currently have:"
  echo ""
  [ "${RAM_GIB:-0}" -lt "$MIN_RAM_GIB" ] \
    && echo "      available RAM : ${RAM_GIB} GiB   <-- BELOW the ${MIN_RAM_GIB} GiB needed" \
    || echo "      available RAM : ${RAM_GIB} GiB   (ok)"
  [ "${SWAP_GIB:-0}" -lt "$MIN_SWAP_GIB" ] \
    && echo "      swap          : ${SWAP_GIB} GiB   <-- BELOW the ${MIN_SWAP_GIB} GiB recommended" \
    || echo "      swap          : ${SWAP_GIB} GiB   (ok)"
  echo ""
  echo "  The model holds ~91 GB of weights. Close your desktop session and any"
  echo "  other GPU or memory-heavy process first. Launching short of this can"
  echo "  hard-lock the machine rather than failing cleanly."
  echo "  Continuing in 10s -- Ctrl-C to abort."
  echo "  ================================================================="
  echo "${N}"
  sleep 10
fi
if [ "$THINKING" = "off" ]; then
  KW='{"enable_thinking": false}'
else
  KW="{\"enable_thinking\": true, \"reasoning_effort\": \"$THINKING\"}"
fi

docker rm -f flashnext >/dev/null 2>&1 || true
docker run -d --name flashnext --privileged --gpus all --network host --ipc=host --shm-size 32g \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -v "$REPO":/patches \
  -v "$REPO/patches/qwen4_exp_nvfp4.py":/sgl-workspace/sglang/python/sglang/srt/models/qwen4_exp.py:ro \
  -v "$REPO/patches/flash_fwd.py":/usr/local/lib/python3.12/dist-packages/flash_attn/cute/flash_fwd.py:ro \
  -v "$REPO/patches/qwen_sparse_attn_backend.py":/sgl-workspace/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py:ro \
  -v "$REPO/patches/sparse_attn.py":/sgl-workspace/sglang/python/sglang/srt/layers/attention/qsa/sparse_attn.py:ro \
  "${PLE_ENV[@]}" \
  -e "Q4X_FP8=lm_head,linear_attn,qkv_proj,o_proj,shared_expert" \
  "$IMAGE" \
  python3 -m sglang.launch_server \
    --model-path "$MODEL_ID" --trust-remote-code --language-only \
    --quantization modelopt_fp4 --fp4-gemm-backend flashinfer_cutlass \
    --kv-cache-dtype fp8_e4m3 --page-size 64 \
    --mamba-scheduler-strategy extra_buffer --mamba-track-interval 64 \
    --chunked-prefill-size 8192 --max-prefill-tokens 32768 \
    --max-running-requests 8 --max-mamba-cache-size 24 --mamba-ssm-dtype bfloat16 \
    --context-length "$CTX" --mem-fraction-static "$MEM_FRACTION" \
    --default-chat-template-kwargs "$KW" \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --strip-thinking-cache \
    --speculative-algorithm NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4 \
    --host 0.0.0.0 --port "$PORT"

echo "Booting (~9 min). Watch: docker logs -f flashnext"
for i in $(seq 1 120); do
  curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "READY on port $PORT"; exit 0; }
  docker ps --format '{{.Names}}' | grep -q '^flashnext$' || { echo "container died -- docker logs flashnext" >&2; exit 1; }
  sleep 15
done
echo "not ready after 30 min -- docker logs flashnext" >&2; exit 1
LAUNCHEOF
sed -i -e "s|@@REPO@@|$REPO|" -e "s|@@HF_CACHE@@|$HF_CACHE|" \
       -e "s|@@IMAGE@@|$IMAGE|" -e "s|@@MODEL_ID@@|$MODEL_ID|" "$LAUNCHER"
chmod +x "$LAUNCHER"
ok "$LAUNCHER"

T_END=$(date +%s); ELAPSED=$(( T_END - T_START ))
cat <<DONE

=== setup complete in $((ELAPSED/60))m $((ELAPSED%60))s ===

  Start:   $LAUNCHER
  Test:    curl localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \\
             -d '{"model":"m","messages":[{"role":"user","content":"Hello"}]}'
  Stop:    docker rm -f flashnext

  NOTE: the endpoint is /v1/chat/completions -- /chat/completions returns 404.

  KNOWN ISSUE: greedy decoding is NOT bit-reproducible on sm_121 -- QSA
  block-selection top-k is approximate on this arch, so temperature 0 can give
  different outputs across identical runs. Benchmark scores vary by a couple of
  points run to run; do not read a single run as exact.
DONE
