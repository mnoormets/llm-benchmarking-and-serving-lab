# LLM Benchmarking and Serving Lab

Local streaming load generator for OpenAI-compatible servers (including vLLM and compatible HF serving) and Ollama native generate. No external/paid endpoints or model auto-downloads. Does not pretend an HF `pipeline` object is a production HTTP server or implement TensorRT-LLM itself.

```
python -m serving.benchmark --url http://127.0.0.1:8000/v1/chat/completions --protocol openai --model YOUR_MODEL --label vllm-warm-c1 --requests 20 --concurrency 1
python -m serving.benchmark --url http://127.0.0.1:11434/api/generate --protocol ollama --model YOUR_MODEL --label ollama-warm-c4 --requests 20 --concurrency 4
```

Sweep concurrency 1, 4, 16, then 100 only if earlier loads are safe. Separate cold-start from warmed runs and pre-run warmups. Use identical base model/revision, tokenizer, prompt set, output length, quantization/dtype, context limit, hardware and server version; record these alongside each run. Different backends/quantization are confounders, not engine-only speed comparisons. Report p50/p95 TTFT, successful requests/s, tokens/s and failures, not only fastest latency.

TTFT is arrival of first nonempty generated content chunk (network-visible approximation to first token). One chunk may contain many tokens: output counts come from final server usage or Ollama eval_count. Missing usage yields unknown token throughput. Closed-loop worker queuing is included in wall throughput but not worker-start TTFT. This does not measure open-loop arrival SLOs or exact inter-token latency. Streaming finish marker is required, HTTP errors remain failures.

PagedAttention manages KV-cache blocks; continuous batching schedules requests across decoding steps. A concurrency comparison alone cannot isolate the impact of PagedAttention. TensorRT-LLM is not simply converting all Python to C++; engine build/configuration and hardware support must be checked. Speculative decoding requires an appropriate draft model and quality-preserving verification, with acceptance rate and total speed measured.

No real vLLM/Ollama/TensorRT-LLM GPU performance results yet. Tests use a labelled local protocol fixture, not LLM inference. Free Colab single-GPU availability is not a multi-GPU allocation.

Quantization plan: fix an independent text corpus, baseline/config hashes and tokenizer; compare identical perplexity likelihood windows and extraction accuracy before/after AWQ/GPTQ. GGUF is a container format with quantization variants, not one calibration algorithm; pruning is distinct. AWQ/GPTQ conversion and lm-evaluation-harness execution are not claimed complete. Exact quality changes must be measured; do not promise a universal 0.5% loss.

Sources: [vLLM benchmarks](https://docs.vllm.ai/en/latest/benchmarking/), [Ollama generate](https://docs.ollama.com/api/generate).

## Quantization quality measurement

`python -m serving.perplexity --model-path LOCAL_SAFETENSORS_MODEL --tokenizer-path SAME_LOCAL_TOKENIZER --corpus evaluation.jsonl --device cuda` evaluates a fixed joined text stream with overlapping likelihood windows. Each causal target is counted once; corpus and token hashes allow before/after input comparisons. Use the same window, stride, tokenizer and independent corpus. Model files must already be local: this command does not buy compute or download models. Loader compatibility with each AWQ/GPTQ backend still needs verification; it does not convert or prune models.

Five CPU tests check scored token coverage and a known uniform-distribution perplexity of 10. These establish the evaluator contract, not an AWQ quality result or real model benchmark.

## Free Colab GPU execution

[Open the T4 notebook](https://colab.research.google.com/github/mnoormets/llm-benchmarking-and-serving-lab/blob/main/notebooks/colab_serving.ipynb). Use a **fresh free T4 runtime** so the QLoRA model is not still occupying VRAM. Run all; the code handles clone/install/server startup, model pull, warmup, two repeats at client concurrency 1/10/50/100, chart and ZIP download. Requires several GB of download and may take tens of minutes.

Uses Mistral-7B-Instruct-v0.3 Q4_K_M with recorded Ollama digest, not the earlier Qwen QLoRA adapter. Server decode parallelism is fixed at 4; higher client concurrency measures queueing/saturation, not 100 simultaneously decoding GPU sequences. TTFT includes server queueing and ends at first nonempty content chunk. Aggregate TPS is successful generated tokens divided by full wall runtime; per-request engine TPS uses Ollama eval_count / eval_duration and is a different metric. Output cap 32, context 2048, fixed prompt set/seed, cold-start receipt separate, three warmups, raw failures retained. Prompt-cache reuse is possible and the experiment is explicitly warm. No comparative vLLM/TensorRT speedup is claimed.

Prepared runner has no real GPU results until its exported receipts are inspected. Local tests use only a labelled protocol fixture. Sources: [Ollama Linux](https://docs.ollama.com/linux), [parallelism](https://docs.ollama.com/faq), [generate timing fields](https://docs.ollama.com/api/generate).

## Colab installer failure repair

A user-exported screenshot showed the official shell installer exiting with status 1 before model startup; its underlying stderr was not visible, so no exact cause is asserted. Colab runner now installs the official Linux tar.zst package under its own ignored runtime prefix, using Python zstandard with filtered tar extraction. It does not require systemd, sudo, or a system zstd executable, and does not reuse partially installed global binaries. Logs, archive SHA256 and install receipt are saved; an installation failure automatically exports a small diagnostic ZIP. Official package binaries still execute normally; source and observed SHA are recorded, not an independent publisher-signature verification. GPU execution of this new installer must be confirmed in Colab.
