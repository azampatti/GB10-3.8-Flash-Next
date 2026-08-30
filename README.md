# GB10-3.8-Flash-Next

A 180B model that officially needs two DGX Sparks, running on one — at **~43 tok/s**
with **91/100** on tool-eval-bench.

## One Liner: 
**curl -fsSL https://raw.githubusercontent.com/azampatti/GB10-3.8-Flash-Next/main/install-3.8flash.sh | bash && ~/launch-3.8flash.sh**

| workload | tok/s |
|---|---:|
| Q&A | 35.2 |
| Code | 38.4 |
| JSON | 43.9 |
| Math | 42.7 |
| LongCode | 43.6 |

*Single stream, 200k context. `sgbench` Run 2/2 on a GB10 DGX Spark.*

---

## Requirements

- **A DGX Spark (GB10 / sm_121), 128 GB unified memory**
- **~105 GiB free RAM and 16 GiB swap at launch** (the script warns if you are short)
- **Docker + NVIDIA Container Toolkit**
- **~40 GB free disk**
- **The checkpoint already downloaded (135 GB):**
  **https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4**
  ```bash
  huggingface-cli download RadixArk/Qwen3.8-Flash-Next-NVFP4
  ```

The installer **exits** if the checkpoint is missing or incomplete.

---

## Install and run

```bash
./install-3.8flash.sh      # ~16 min: clone, patch, build the n-gram table
~/launch-3.8flash.sh       # ~9 min boot, then READY on port 30000
```

```bash
curl localhost:30000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"m","messages":[{"role":"user","content":"Hello"}]}'
```

Stop with `docker rm -f flashnext`.

---

## What it does

The model is 180B total: a 125B-A6B MoE plus a **51.2 GB hashed n-gram embedding table**
that no quantizer touches — it is a lookup table, not a matmul. At full size the weights
come to ~129 GB and simply do not fit. Four things make it work:

1. **[flashnext-one-spark](https://github.com/deathbyorderfill/flashnext-one-spark)
   patches** — mandatory on sm_121. Stock kernels silently corrupt on this chip.
2. **HashK R=4** — compresses the n-gram table 51.2 GB → 12.8 GB, built locally in ~7 min.
3. **FP8 dense cast** — 256 modules converted at load time. **+27% decode, no measurable
   quality cost.** Includes the Gated DeltaNet layers that both public quantizers skip.
4. **NEXTN speculative decoding, depth 3** — worth **1.78×** on its own. Depth 3 / 4 draft
   tokens measured optimal; 1, 2 and 5 are all worse.

---

## Known issues

- **The endpoint is `/v1/chat/completions`.** Bare `/chat/completions` returns 404.
- **Greedy decoding is not bit-reproducible on sm_121.** QSA block-selection top-k is
  approximate on this architecture, so temperature 0 can give different output across
  identical runs. Benchmark scores move a couple of points run to run — do not read a
  single run as exact.
- **Context is set to 200k**, not the model's 262k. At 262k the KV pool does not fit
  alongside the weights.

---

## Credits

[Qwen](https://huggingface.co/Qwen) · [RadixArk](https://huggingface.co/RadixArk) (NVFP4
checkpoint) · [flashnext-one-spark](https://github.com/deathbyorderfill/flashnext-one-spark)
(sm_121 patches, HashK) · [SGLang](https://github.com/sgl-project/sglang)

