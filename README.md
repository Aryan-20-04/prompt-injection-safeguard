<div align="center">

<img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
<img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
<img src="https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black"/>
<img src="https://img.shields.io/badge/Google_Colab-T4_GPU-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white"/>
<img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>

<br/><br/>

# 🛡️ Prompt Injection Benchmark

### A research-grade evaluation framework for prompt injection & jailbreak detection models

*Systematically benchmark encoder models, guard models, LLMs, and API models on adversarial prompt datasets — with reproducible runs, rich metrics, and a self-contained HTML dashboard.*

<br/>

![Dashboard Preview](docs/dashboard_preview.png)

</div>

---

## 📋 Table of Contents

- [What This Is](#-what-this-is)
- [Architecture](#-architecture)
- [Supported Models](#-supported-models)
- [Supported Datasets](#-supported-datasets)
- [Metrics](#-metrics)
- [Quickstart](#-quickstart)
- [Running on Google Colab](#-running-on-google-colab)
- [Configuration](#-configuration)
- [Output Format](#-output-format)
- [Adding a New Model](#-adding-a-new-model-30-seconds)
- [Adding a New Dataset Suite](#-adding-a-new-dataset-suite-yaml-only)
- [Results](#-results)
- [Reproducibility](#-reproducibility)

---

## 🔍 What This Is

Prompt injection and jailbreak attacks are among the most actively researched threats in deployed LLM systems. Dozens of detection models have been published, but there is no standard way to compare them head-to-head across consistent datasets, splits, and metrics.

This framework provides:

- **A unified adapter interface** — any model (encoder, LLM, or API) plugged in the same way
- **Multi-dataset benchmark suites** — run across 4 adversarial datasets in one command
- **Security-focused metrics** — not just F1, but Attack Detection Rate and False Positive Rate
- **Full reproducibility** — seeds, config hashes, dataset hashes, GPU info saved per run
- **A self-contained HTML dashboard** — one file, no server, shareable anywhere

---

## 🏗️ Architecture

```
prompt-injection-benchmark/
│
├── configs/
│   ├── benchmark.yaml          ← Run configuration (suite, models, eval mode)
│   ├── datasets.yaml           ← Custom dataset suite definitions
│   └── model_registry.yaml     ← Central model catalogue
│
├── core/
│   ├── runner.py               ← Main orchestrator
│   └── config_loader.py        ← Config validation
│
├── data/
│   ├── hf_loader.py            ← HuggingFace dataset loader with label remapping
│   └── suites.py               ← Built-in + YAML-extended suite registry
│
├── models/
│   ├── base_model.py           ← BaseModel ABC
│   ├── discovery.py            ← Auto-discovers all adapters in models/adapters/
│   └── adapters/
│       ├── hf_classifier.py    ← Shared mixin for all encoder models
│       ├── llm_base.py         ← Shared mixin for all generative LLMs
│       ├── distilbert.py       ← DistilBERT adapter
│       ├── roberta.py          ← RoBERTa adapter
│       ├── deberta.py          ← DeBERTa-v3 adapter
│       ├── bert.py             ← BERT adapter
│       ├── kiwi_guard.py       ← ProtectAI guard model
│       ├── qwen.py             ← Qwen2 LLM adapter
│       ├── llama.py            ← Llama 3 adapter
│       ├── mistral.py          ← Mistral adapter
│       ├── openai_guard.py     ← OpenAI GPT via API
│       └── anthropic_guard.py  ← Anthropic Claude via API
│
├── evaluation/
│   └── evaluator.py            ← Binary + multi-label evaluation engine
│
├── inference/
│   └── engine.py               ← Batched sequential + parallel inference
│
├── leaderboard/
│   └── generator.py            ← Ranked leaderboard with all metrics
│
├── reproducibility/
│   └── snapshot.py             ← Full environment + seed capture
│
└── visualization/
    ├── charts.py               ← 5 analysis charts (PNG)
    └── dashboard.py            ← Self-contained HTML dashboard
```

---

## 🤖 Supported Models

| ID | Model | Type | Size | Notes |
|---|---|---|---|---|
| `kiwi_guard` | protectai/deberta-v3-base-prompt-injection | Guard | 184M | Best performer — purpose-built |
| `distilbert` | distilbert-base-uncased | Encoder | 66M | Fast baseline |
| `roberta` | roberta-base | Encoder | 125M | Strong general classifier |
| `deberta` | microsoft/deberta-v3-base | Encoder | 184M | Best encoder backbone |
| `bert` | bert-base-uncased | Encoder | 110M | Classic baseline |
| `qwen` | Qwen/Qwen2-1.5B-Instruct | LLM | 1.5B | Smallest LLM option |
| `llama` | meta-llama/Meta-Llama-3-8B-Instruct | LLM | 8B | Needs 4-bit on T4 |
| `mistral` | mistralai/Mistral-7B-Instruct-v0.3 | LLM | 7B | Needs 4-bit on T4 |
| `openai_guard` | gpt-4o-mini | API | — | Requires `OPENAI_API_KEY` |
| `anthropic_guard` | claude-haiku-4-5 | API | — | Requires `ANTHROPIC_API_KEY` |

> **Note:** For encoder models, use fine-tuned checkpoints for meaningful results. Base pretrained weights (e.g. `roberta-base`) will predict a constant class. See the [Configuration](#-configuration) section for recommended fine-tuned models.

---

## 📦 Supported Datasets

| Suite | Datasets included | Samples |
|---|---|---|
| `prompt_injection` | Smooth-3/llm-prompt-injection-attacks | ~5500 |
| `prompt_injection_full` | Above + PromptBench + AdvBench + JailbreakBench | ~7000 |
| `smoke` | Smooth-3/llm-prompt-injection-attacks (50 samples) | 50 |

Add custom suites in `configs/datasets.yaml` — no Python changes needed.

---

## 📊 Metrics

Every run computes the following for each model × dataset combination:

| Metric | Description |
|---|---|
| `micro_f1` | F1 weighted by sample count — primary ranking metric |
| `macro_f1` | F1 averaged equally across classes |
| `precision` | Macro-averaged precision |
| `recall` | Macro-averaged recall |
| `attack_detection_rate` | TP / (TP + FN) — attacks correctly flagged ↑ |
| `false_positive_rate` | FP / (FP + TN) — benign samples wrongly flagged ↓ |
| `per_class_f1` | Per-label F1 (BENIGN, JAILBREAK, etc.) |
| `confusion_matrix` | Full confusion matrix |
| `latency_p50/p95/p99` | Per-sample inference latency percentiles |

> **ADR and FPR are the most operationally relevant metrics** for a deployed safety system. A model with 0.99 F1 but 0.30 FPR will flag 30% of legitimate user queries as attacks.

---

## ⚡ Quickstart

```bash
# Clone
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# Install
pip install transformers datasets scikit-learn matplotlib tqdm pyyaml accelerate

# Run smoke test (50 samples, ~2 min)
python scripts/run_benchmark.py --config configs/benchmark.yaml

# View dashboard
open runs/baseline_run/dashboard/dashboard.html
```

---

## 🚀 Running on Google Colab

```python
# Cell 1 — Clone & setup
import os, sys
!git clone https://github.com/your-username/your-repo-name.git /content/benchmark
os.chdir("/content/benchmark")
sys.path.insert(0, "/content/benchmark")

# Cell 2 — Install
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
!pip install -q transformers datasets scikit-learn matplotlib tqdm pyyaml accelerate

# Cell 3 — Run
import torch, gc
torch.cuda.empty_cache()
gc.collect()

from core.runner import BenchmarkRunner
results = BenchmarkRunner(eval_mode="binary").run_from_config_path(
    "/content/benchmark/configs/active_run.yaml"
)

# Cell 4 — Download dashboard
from google.colab import files
files.download("/content/runs/baseline_run/dashboard/dashboard.html")
```

**T4 GPU estimated runtimes:**

| Model | Time |
|---|---|
| DistilBERT | ~1 min |
| RoBERTa | ~2 min |
| DeBERTa | ~3 min |
| ProtectAI Guard | ~3 min |
| Mistral 7B (4-bit) | ~15 min |
| **Total (encoders only)** | **~10 min** |

---

## ⚙️ Configuration

`configs/benchmark.yaml` controls everything:

```yaml
run:
  output_dir: runs/baseline_run
  device: auto                  # auto | cpu | cuda | mps
  batch_size: 32
  seed: 42
  eval_mode: binary             # binary | multi_label

suite: prompt_injection         # prompt_injection | prompt_injection_full | smoke

labels:
  - BENIGN
  - JAILBREAK

models:
  - id: kiwi_guard
    model_name: protectai/deberta-v3-base-prompt-injection

  - id: distilbert
    model_name: fmops/distilbert-prompt-injection   # fine-tuned checkpoint

  # API models — set OPENAI_API_KEY / ANTHROPIC_API_KEY env vars
  # - id: openai_guard
  #   model_name: gpt-4o-mini
  # - id: anthropic_guard
  #   model_name: claude-haiku-4-5
```

**Evaluation modes:**

| Mode | Description |
|---|---|
| `binary` | All non-BENIGN labels collapse to JAILBREAK |
| `multi_label` | Full label set: JAILBREAK, INSTRUCTION_OVERRIDE, ROLE_HIJACK, DATA_EXFILTRATION |

---

## 📁 Output Format

```
runs/baseline_run/
├── resolved_config.yaml          ← Exact config used
├── run_metadata.json             ← GPU, seed, git hash, dataset hashes
├── metrics/
│   └── kiwi_guard__Smooth-3_....json   ← Full metrics per model × dataset
├── raw_predictions/
│   └── kiwi_guard__Smooth-3_....jsonl  ← Per-sample predictions
├── leaderboard/
│   └── leaderboard.json          ← Ranked table
├── charts/
│   ├── model_comparison.png
│   ├── latency_comparison.png
│   ├── macro_vs_micro.png
│   ├── per_attack_heatmap.png
│   └── fpr_adr.png
└── dashboard/
    └── dashboard.html            ← Self-contained report — open in any browser
```

---

## 🔌 Adding a New Model (30 seconds)

Create `models/adapters/my_model.py`:

```python
from models.model_registry import register
from models.base_model import ModelMetadata
from models.adapters.hf_classifier import HFClassifierMixin

@register("my_model")
class MyModelAdapter(HFClassifierMixin):
    metadata = ModelMetadata(
        name="My Model",
        model_type="encoder",
        hf_id="org/model-name",
    )
```

Add to `benchmark.yaml`:
```yaml
models:
  - id: my_model
    model_name: org/model-name
```

Discovery is automatic — no imports to update.

---

## 📝 Adding a New Dataset Suite (YAML only)

Add to `configs/datasets.yaml`:

```yaml
suites:
  my_suite:
    - hf_name: org/my-dataset
      split: test
      text_field: prompt
      label_field: label
      max_samples: 1000
      label_map:
        "0": BENIGN
        "1": JAILBREAK
```

Reference in `benchmark.yaml`:
```yaml
suite: my_suite
```

---

## 📈 Results

> Results from a single T4 GPU run on `Smooth-3/llm-prompt-injection-attacks` validation split (5500 samples).

| Rank | Model | Micro F1 | ADR ↑ | FPR ↓ | P50 (ms) |
|---|---|---|---|---|---|
| 1 | ProtectAI-DeBERTa-Guard | **0.974** | **0.981** | **0.018** | 3.2 |
| 2 | DeBERTa-v3 (fine-tuned) | 0.948 | 0.955 | 0.038 | 6.1 |
| 3 | RoBERTa (fine-tuned) | 0.920 | 0.930 | 0.062 | 4.7 |
| 4 | DistilBERT (fine-tuned) | 0.893 | 0.900 | 0.089 | 2.8 |

> **Key takeaway:** The purpose-built ProtectAI guard model significantly outperforms general-purpose encoders, even fine-tuned ones, while maintaining the lowest FPR and sub-4ms latency.

---

## 🔁 Reproducibility

Every run saves a `run_metadata.json` capturing:

```json
{
  "timestamp": "2025-03-15T10:22:05Z",
  "git_commit": "a3f91bc",
  "git_dirty": false,
  "torch": "2.3.0",
  "cuda_version": "12.1",
  "gpu": "Tesla T4",
  "gpu_memory_gb": 15.84,
  "seed": 42,
  "config_hash": "d4e7a91c3b02",
  "dataset_hashes": {
    "Smooth-3/llm-prompt-injection-attacks:validation": "8f3c21a09d44b17e"
  }
}
```

All random seeds (Python, NumPy, PyTorch, CUDA) are set at startup. `cudnn.deterministic = True` is enforced.

---

## 🛠️ Requirements

```
python >= 3.10
torch >= 2.0
transformers >= 4.40
datasets >= 2.18
scikit-learn >= 1.4
matplotlib >= 3.8
tqdm
pyyaml
accelerate

# Optional
openai       # for openai_guard
anthropic    # for anthropic_guard
bitsandbytes # for 4-bit LLM quantization on T4
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built for the research community. PRs welcome — especially new model adapters and dataset suites.

**[⭐ Star this repo](https://github.com/your-username/your-repo-name)** if you find it useful.

</div>