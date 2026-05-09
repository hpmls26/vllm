# Simamba vLLM Integration Notes

This fork contains a test-focused native vLLM integration for Simamba. The
integration is named `SimambaForCausalLM` and was used for batch-size-1
profiling of the 10M SlimPajama checkpoints.

The current VM layout used for testing was:

```bash
SIMAMBA_REPO=/home/david/simamba
VLLM_FORK=/tmp/hpmls26_vllm
SIMAMBA_MODEL=/home/david/simamba/outputs/improved_simamba_10m_slimpajama500m_20260508_013540/vllm_export
MAMBA2_MODEL=soumil1/mamba2-10m-slimpajama-500m
```

## What Changed

### Native Simamba model path

The fork adds a native vLLM model implementation for Simamba:

- `vllm/model_executor/models/simamba.py`
  - Defines `SimambaForCausalLM`, `SimambaModel`, and `SimambaDecoderLayer`.
  - Registers state shapes and state dtypes for vLLM's Mamba state cache.
  - Loads the checkpoint's fused `mixer.in_proj.weight` into both the original
    split projection parameters and the optimized fused projection.

- `vllm/model_executor/layers/mamba/simamba_mixer.py`
  - Implements the Simamba mixer as a vLLM custom op.
  - Supports the local Simamba Triton backend and the reference backend.
  - Maintains six state tensors: angle, SSM, two key-history states, and two
    value-history states.

### Compatibility fixes

The integration needed a few compatibility fixes to run with the installed
vLLM/Torch stack on this VM:

- `make_layers` callback now accepts `prefix=...`, matching installed vLLM's
  helper signature.
- `B_bias` and `C_bias` loading accepts checkpoint tensors shaped
  `[heads, 1, state]` and squeezes the singleton group dimension.
- Runtime KV cache lookup handles stock vLLM's per-virtual-engine cache layout.
- The Simamba chunked prefill path no longer passes `cu_seqlens` for the tested
  batch-size-1 path, because the local Triton Simamba op does not support that
  varlen argument.
- The profiling overlay sets `chunk_size` and `mamba_chunk_size` from
  `config.ssm_cfg["chunk_size"]`. This is required for this checkpoint because
  the trained chunk size is 16, while vLLM otherwise may default to a much larger
  chunk size.

### Performance fixes

Two changes were added after NCU/NSYS showed the decode path was launch-bound:

- TP=1 fused input projection:
  - `SimambaMixer` keeps a single `in_proj` layer and slices its output into
    `z`, `x`, `b`, `c`, `dt`, `a`, `simpson`, optional `midpoint`, and `angles`.
  - This avoids launching one small GEMV per projection slice during decode.

- Batch-size-1 persistent decode state:
  - For non-prefix-cache decode, the first token gathers/scatters vLLM cache
    state normally.
  - Subsequent single-request decode steps keep the active states in layer
    memory and update them directly.
  - This removes most repeated `index_select`/`index_copy_` state-cache kernels
    from batch-size-1 decode.

CUDA graphs were then enabled in the profiling scripts to reduce remaining
Python/CUDA launch overhead.

### Prefix-cache graph fix

Prefix caching uses vLLM's Mamba cache `"all"` mode. That path creates different
metadata from normal decode, so the persistent-state fast path is disabled there.

A graph-capture issue was fixed in the cache-all decode path by avoiding
creation of a new CUDA tensor from the Python `q_lens` list during single-token
decode. This allows prefix-cache decode to pass a CUDA graph smoke test.

## Tested Results

Batch size 1, 32 prompt words, 32 generated tokens, prefix cache off:

| Model | Mode | Median tok/s | Median TPOT | Median TTFT |
| --- | --- | ---: | ---: | ---: |
| Mamba2 | CUDA graph | 343.98 | 2.196 ms | 23.67 ms |
| Simamba | CUDA graph, optimized native path | 284.18 | 2.403 ms | 37.37 ms |

For the 512-token repeated-prefix prompt:

| Model | Mode | Median tok/s | Median TPOT | Median TTFT |
| --- | --- | ---: | ---: | ---: |
| Mamba2 | CUDA graph | 348.20 | 2.193 ms | 23.62 ms |
| Simamba | CUDA graph, optimized native path | 303.31 | 2.468 ms | 29.11 ms |

The main remaining gap is TTFT/prefill overhead. Decode TPOT is now close to
Mamba2 for this small checkpoint.

## Run Mamba2

From the Simamba repo, run the same benchmark sweep used for the comparison:

```bash
cd /home/david/simamba

PYTHONPATH=/home/david/simamba \
python profiling/vllm_sweep.py \
  --models soumil1/mamba2-10m-slimpajama-500m \
  --prompt-words 32 \
  --batch-sizes 1 \
  --max-tokens 32 \
  --prefix-cache-modes off \
  --repeats 5 \
  --warmup 1 \
  --trust-remote-code \
  --model-impl auto \
  --use-cudagraph \
  --out results/vllm_mamba2_cudagraph_b1_summary.csv \
  --raw-out results/vllm_mamba2_cudagraph_b1_raw.csv \
  --plot-out results/vllm_mamba2_cudagraph_b1.png
```

To serve Mamba2 directly with the vLLM CLI:

```bash
vllm serve soumil1/mamba2-10m-slimpajama-500m \
  --served-model-name mamba2 \
  --tokenizer EleutherAI/gpt-neox-20b \
  --trust-remote-code \
  --dtype float16 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.25 \
  --hf-overrides '{"num_attention_heads":10,"num_key_value_heads":10,"max_position_embeddings":1024}'
```

Send a demo request:

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mamba2","prompt":"State space models are","max_tokens":32,"temperature":0}'
```

## Run Simamba

The native Simamba path needs this fork's model files and the local Simamba
kernel package. On the current VM, the tested path uses the Simamba repo's
profiling overlay helper, which loads this fork's Simamba modules into the
installed vLLM package.

Run the optimized Simamba sweep:

```bash
cd /home/david/simamba

PYTHONPATH=/home/david/simamba \
python profiling/vllm_sweep.py \
  --models /home/david/simamba/outputs/improved_simamba_10m_slimpajama500m_20260508_013540/vllm_export \
  --prompt-words 32 \
  --batch-sizes 1 \
  --max-tokens 32 \
  --prefix-cache-modes off \
  --repeats 5 \
  --warmup 1 \
  --trust-remote-code \
  --model-impl auto \
  --simamba-vllm-fork /tmp/hpmls26_vllm \
  --simamba-native-backend triton \
  --use-cudagraph \
  --out results/vllm_hpmls26_simamba_cudagraph_fused_inproj_state_b1_summary.csv \
  --raw-out results/vllm_hpmls26_simamba_cudagraph_fused_inproj_state_b1_raw.csv \
  --plot-out results/vllm_hpmls26_simamba_cudagraph_fused_inproj_state_b1.png
```

Run one direct generation smoke test:

```bash
cd /home/david/simamba

VLLM_ENABLE_V1_MULTIPROCESSING=0 \
PYTHONPATH=/home/david/simamba \
python profiling/vllm_decode_once.py \
  --model /home/david/simamba/outputs/improved_simamba_10m_slimpajama500m_20260508_013540/vllm_export \
  --simamba-vllm-fork /tmp/hpmls26_vllm \
  --simamba-native-backend triton \
  --prompt-words 32 \
  --max-tokens 64 \
  --warmup 2 \
  --use-cudagraph
```

### Simamba CLI serving

The normal `vllm serve` CLI cannot call the profiling overlay hook. To serve
Simamba with the CLI, this fork must be the active vLLM package in the Python
environment, and the local Simamba repo must be importable:

```bash
export PYTHONPATH=/home/david/simamba:/tmp/hpmls26_vllm

vllm serve /home/david/simamba/outputs/improved_simamba_10m_slimpajama500m_20260508_013540/vllm_export \
  --served-model-name simamba \
  --tokenizer EleutherAI/gpt-neox-20b \
  --trust-remote-code \
  --model-impl auto \
  --dtype float16 \
  --max-model-len 1024 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 1024 \
  --gpu-memory-utilization 0.25 \
  --hf-overrides '{"num_attention_heads":5,"num_key_value_heads":5,"max_position_embeddings":1024,"chunk_size":16,"mamba_chunk_size":16}'
```

If this fails with an import error from the full vLLM fork, use the tested
overlay scripts above. The current VM benchmarks were produced through that
overlay path.

Send a demo request:

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"simamba","prompt":"State space models are","max_tokens":32,"temperature":0}'
```

## Prefix Caching

Prefix caching uses vLLM Mamba cache mode `"all"`:

```bash
cd /home/david/simamba

VLLM_ENABLE_V1_MULTIPROCESSING=0 \
PYTHONPATH=/home/david/simamba \
python profiling/vllm_decode_once.py \
  --model /home/david/simamba/outputs/improved_simamba_10m_slimpajama500m_20260508_013540/vllm_export \
  --simamba-vllm-fork /tmp/hpmls26_vllm \
  --simamba-native-backend triton \
  --prompt-words 32 \
  --max-tokens 8 \
  --warmup 1 \
  --prefix-cache \
  --use-cudagraph
```

The cache-all path is still experimental. The persistent single-request state
fast path is intentionally disabled when prefix caching is on, because cache-all
must write intermediate states at block boundaries.

## Profiling Commands

Target the Simamba recurrence kernel with Nsight Compute:

```bash
cd /home/david/simamba

sudo -n env VLLM_ENABLE_V1_MULTIPROCESSING=0 PYTHONPATH=/home/david/simamba \
/usr/local/cuda/bin/ncu \
  --set full \
  --kernel-name 'regex:mamba3_siso_step_kernel' \
  --launch-count 1 \
  --target-processes all \
  --csv \
  --page raw \
  --force-overwrite \
  --export profiling/results/vllm_decode_nsys/simamba_step_after_fastpath_sudo_ncu_full \
  /opt/conda/bin/python profiling/vllm_decode_once.py \
    --model /home/david/simamba/outputs/improved_simamba_10m_slimpajama500m_20260508_013540/vllm_export \
    --simamba-vllm-fork /tmp/hpmls26_vllm \
    --simamba-native-backend triton \
    --prompt-words 32 \
    --max-tokens 64 \
    --warmup 2
```

Capture the final optimized vLLM path with Nsight Systems:

```bash
cd /home/david/simamba

VLLM_ENABLE_V1_MULTIPROCESSING=0 \
PYTHONPATH=/home/david/simamba \
/usr/local/cuda/bin/nsys profile \
  --force-overwrite=true \
  --trace=cuda,nvtx,cublas,osrt \
  --sample=none \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop-shutdown \
  --output profiling/results/vllm_decode_nsys/simamba_cudagraph_fused_inproj_state_singleproc \
  python profiling/vllm_decode_once.py \
    --model /home/david/simamba/outputs/improved_simamba_10m_slimpajama500m_20260508_013540/vllm_export \
    --simamba-vllm-fork /tmp/hpmls26_vllm \
    --simamba-native-backend triton \
    --prompt-words 32 \
    --max-tokens 64 \
    --warmup 2 \
    --use-cudagraph
```

Export NSYS summary tables:

```bash
/usr/local/cuda/bin/nsys stats \
  --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum,osrt_sum \
  --format csv \
  --force-export=true \
  --output profiling/results/vllm_decode_nsys/simamba_cudagraph_fused_inproj_state_singleproc_stats \
  profiling/results/vllm_decode_nsys/simamba_cudagraph_fused_inproj_state_singleproc.nsys-rep
```

## Known Limitations

- The integration is test-focused and was validated for batch size 1.
- Tensor parallelism beyond TP=1 was not optimized.
- Prefix caching works for the smoke test, but cache-all mode is still
  experimental and uses a slower state-writing path.
- The `vllm serve` Simamba command requires this fork to be the active vLLM
  package. The measured results used the overlay helper in the Simamba repo.
- Simamba decode is now close to Mamba2, but Mamba2 is still faster on this VM:
  about 1.21x faster on the normal prompt and 1.15x faster on the repeated
  prefix prompt.

