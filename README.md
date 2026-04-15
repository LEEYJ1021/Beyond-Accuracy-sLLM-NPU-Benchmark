# A System-Level Stability Methodology for Deploying Small Language Models on Embedded NPUs: Architectural Performance Characterization for Real-Time Industrial Edge AI

This repository provides the full experimental pipeline and analysis code for the paper:

> **A System-Level Stability Methodology for Deploying Small Language Models on Embedded NPUs: Architectural Performance Characterization for Real-Time Industrial Edge AI**

The study proposes a system-level embedded software design methodology that supports deployment decision-making by systematically quantifying **performance variability**, **tail-latency risk**, and **hardware-resource trade-offs** across nine sLLMs on the Rebellions ATOM™ NPU. Through 300 independent trials per model-task combination, the framework addresses the practical gap faced by engineers deploying AI systems in resource-constrained industrial environments — where conventional mean-latency benchmarks systematically underrepresent worst-case deployment risk.

**Keywords**: Embedded AI System Design; NPU Inference Stability; Real-Time Edge Systems; Embedded Software Design Methodology; Optimization Priority Score; Hardware-Software Co-Design.

---

## Key Findings

- **H1 (Model Size × Task Interaction):** A statistically significant interaction effect between model parameter count and task type on inference latency was confirmed (F(2, 2683) = 41.55, p < .001, η²p = 0.030). Latency scaling for Summarization is approximately 2.7× steeper than for NLU across the tested size range, demonstrating that single-task benchmarks are insufficient for multi-task resource provisioning.

- **H2 (Non-linear Context Length Scaling):** A natural cubic spline model (df=4) outperformed the linear baseline (ΔAIC = 8.49; in-sample ΔR² = 58.1%; LOOCV ΔR² = 48.3%), confirming that linear capacity planning models systematically misestimate latency at non-standard context length configurations on the ATOM™ NPU. The LOOCV-adjusted figure (48.3%) is reported as the conservative estimate; results should be interpreted as directional evidence given the small sample size (n = 9 model-aggregate data points).

- **H3 (Latency Variance Heterogeneity):** Levene's test confirmed significant heterogeneity in latency variance across NLU, QA, and Summarization tasks (W = 5.82, p = .003, Cohen's f = 1.58). NLU exhibited the highest instability (CV = 1.203) despite the lowest mean latency, motivating task-specific SLA thresholds in multi-task deployments.

- **H4 (Dominant Throughput Predictor):** Random Forest permutation importance analysis identified peak NPU memory utilization as the strongest predictive feature for token throughput (Importance = 0.458), outperforming model parameter count (0.353). Memory footprint reduction strategies — including quantization and KV cache management — are accordingly prioritized over parameter pruning alone. These findings reflect predictive associations, not established causal mechanisms.

- **H5 (Non-Normal Latency Distributions):** All latency distributions were significantly non-normal (Shapiro-Wilk, all p < .001). The **Optimization Priority Score (OPS)** composite index synthesizes CV, Tail Ratio (P95/P50), Skewness, and Kurtosis into a single deployment readiness ranking, enabling practitioners to identify models requiring stability intervention prior to production deployment.

---

## Optimization Priority Score (OPS)

The OPS is the primary methodological contribution of this study. It provides a composite, reproducible index for ranking models by their aggregate latency instability:

```
OPSᵢ = 0.35·CV*ᵢ + 0.35·TailRatio*ᵢ + 0.15·|Skew*ᵢ| + 0.15·Kurt*ᵢ
```

Where each metric is min-max normalized across the nine evaluated models (range: 0 to 1). A **higher OPS indicates greater instability** and higher priority for stability-focused optimization (e.g., quantization calibration, inference engine tuning, context length capping) before production deployment.

| Model | Mean (ms) | CV | P95/P50 | Skewness | Kurtosis | OPS | Classification |
|---|---|---|---|---|---|---|---|
| OPT-2.7B | 1462.5 | 0.154 | 1.42 | 0.45 | 1.99 | 0.089 | Stable |
| Qwen2.5-0.5B-Instruct | 611.2 | 0.177 | 1.33 | 1.08 | 3.41 | 0.093 | Stable |
| DeepSeek-R1-Distill-Qwen-1.5B | 1205.3 | 0.433 | 1.25 | 0.83 | 2.57 | 0.397 | Moderate |
| Qwen2.5-3B-Instruct | 1985.5 | 0.458 | 1.24 | 0.72 | 2.17 | 0.414 | Moderate |
| Llama-3.2-3B-Instruct | 1866.9 | 0.474 | 1.26 | 0.77 | 2.33 | 0.431 | Moderate |
| Qwen2.5-1.5B-Instruct | 1025.9 | 0.624 | 1.47 | 1.26 | 2.86 | 0.574 | Moderate |
| EXAONE-3.5-2.4B-Instruct ★ | 932.1 | 0.745 | 1.51 | 1.12 | 2.71 | 0.661 | Intervention required |
| gemma-2b-it ★ | 753.2 | 0.816 | 2.47 | 0.97 | 2.89 | 0.841 | Intervention required |
| Midm-2.0-Mini-Instruct ★ | 1191.3 | 0.827 | 2.90 | 1.08 | 2.95 | 0.907 | Intervention required |

★ OPS > 0.6: stability intervention required prior to industrial deployment.

> **Note:** Outlier Ratio (OR) is reported as a supplementary diagnostic in paper Table A2 but is excluded from OPS computation due to high collinearity with CV (Pearson r = 0.87). OPS weight sensitivity analysis (±0.10 perturbation per metric) confirmed that top-3 and bottom-3 model rankings are invariant to reasonable weighting assumptions (see paper Appendix C, Table C1).

---

## Evaluated Models

| Model | Parameters (B) | Context Length | Architecture | License |
|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5 | 32,768 | Decoder-only Transformer | DeepSeek Community |
| EXAONE-3.5-2.4B-Instruct | 2.4 | 32,768 | Hybrid Transformer (Encoder–Decoder) | EXAONE Open |
| gemma-2b-it | 2.0 | 8,192 | Decoder-only Transformer | Gemma (Google) |
| Llama-3.2-3B-Instruct | 3.2 | 128,000 | Transformer (GQA) | Llama 3.2 Community |
| Midm-2.0-Mini-Instruct | 2.0 | 32,768 | Decoder-only Transformer | Apache 2.0 |
| OPT-2.7B | 2.7 | 2,048 | Decoder-only Transformer | Meta (Fairseq) |
| Qwen2.5-0.5B-Instruct | 0.5 | 32,768 | Decoder-only Transformer | Apache 2.0 |
| Qwen2.5-1.5B-Instruct | 1.5 | 32,768 | Decoder-only Transformer | Apache 2.0 |
| Qwen2.5-3B-Instruct | 3.0 | 32,768 | Decoder-only Transformer | Apache 2.0 |

Model parameter counts range from 0.5B to 3.2B, targeting the computational range most applicable to current edge applications where balancing capability and resource constraints is a key deployment challenge.

---

## Evaluation Tasks and Experimental Design

Three NLP tasks spanning the computational complexity range of industrial edge AI applications were evaluated:

| Task | Dataset | Metric | Description |
|---|---|---|---|
| NLU | GLUE SST-2 (validation) | F1-Score | Sentiment classification |
| QA | SQuAD v2 (validation) | ROUGE-1 | Extractive question answering |
| Summarization | CNN/DailyMail 3.0.0 (validation) | ROUGE-1 | Abstractive summarization |

These three tasks collectively span the complexity range most relevant to industrial edge AI — from lightweight classification (NLU) to extractive generation (QA) to abstractive generation (Summarization) — enabling systematic characterization of how task complexity interacts with hardware-induced performance variability.

**Experimental design:** 100 samples per task × 3 random seeds = **300 independent trials per model-task combination** (N = 8,100 total observations across 9 models). Each prompt was executed in a separate, reset inference session to prevent cache state carry-over between trials. Trial order was randomized to prevent systematic bias from thermal throttling or sequential cache state effects.

Inference was standardized with `max_new_tokens=50`, `batch_size=1`, `temperature=0.7`, and `random_seed=42` to reflect single-user real-time edge deployment conditions. All models were compiled using `RBLNAutoModelForCausalLM` from `optimum-rebellions`, following manufacturer-recommended optimization practices.

---

## Repository Structure

```
.
├── data/
│   ├── Data_정제.xlsx           # Main aggregated data: per-model, per-task summary statistics
│   ├── Data_허깅페이스.xlsx       # Hugging Face model metadata (architecture, context length, license)
│   └── Data_시계열.xlsx         # Raw time-series latency data: one row per inference trial
├── src/
│   ├── 1_compile_models.py      # Compile Hugging Face models to ATOM™ NPU binaries
│   ├── 2_benchmark_inference.py # Run inference benchmarks and log per-trial metrics
│   └── 3_analyze_results.py     # Statistical analysis: H1–H5 hypothesis tests, OPS computation, figures
├── requirements.txt
└── README.md
```

**Data schema for `Data_시계열.xlsx`** (one row = one inference trial):

| Column | Description |
|---|---|
| `model_name` | Model identifier |
| `task` | NLU / QA / Summarization |
| `trial_id` | Trial index (1–300 per model-task combination) |
| `seed` | Random seed used (42, 43, 44) |
| `latency_ms` | End-to-end wall-clock latency (ms) |
| `tokens_generated` | Number of output tokens |
| `throughput_tps` | Tokens per second |
| `peak_npu_mem_gb` | Peak NPU memory during inference (GB) |
| `estimated_power_w` | Estimated NPU power draw (W) |

---

## Setup

### Prerequisites

- Ubuntu 22.04 LTS
- Python 3.9+
- Rebellions ATOM™ NPU with RBLN SDK installed ([official documentation](https://rebellions.ai))
- PyTorch 2.x
- `optimum-rebellions` library

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/LEEYJ1021/Beyond-Accuracy-sLLM-NPU-Benchmark.git
cd Beyond-Accuracy-sLLM-NPU-Benchmark

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install -r requirements.txt
```

> **Note:** `optimum[rbln]` requires the Rebellions RBLN SDK to be installed and licensed separately. Follow the [official Rebellions SDK setup guide](https://rebellions.ai) before proceeding.

---

## Usage: Experimental Workflow

### Step 1: Compile Models for the ATOM™ NPU

Convert Hugging Face pretrained models into NPU-optimized binaries.

```bash
python src/1_compile_models.py
```

- The `models_config` dictionary in the script defines the nine models evaluated in the paper. Add or remove entries to adjust the model set.
- Compiled binaries are saved to `compiled_models/<model_name>/` by default.
- The script includes automatic Out-Of-Memory (OOM) handling: if compilation fails, `max_seq_len` is incrementally reduced and compilation is retried. This mirrors the adaptive compilation strategy used in the paper.
- All models are compiled using `RBLNAutoModelForCausalLM` from `optimum-rebellions`, following manufacturer-recommended optimization practices.

### Step 2: Run Inference Benchmarks

Execute the full 300-trial benchmark for each model-task combination.

```bash
python src/2_benchmark_inference.py \
    --model_dir compiled_models/<model_name> \
    --task [nlu|qa|summarization] \
    --n_trials 300 \
    --seeds 42 43 44 \
    --output_file data/Data_시계열.xlsx
```

Key design choices reflected in the script:
- `batch_size=1` to simulate single-user real-time inference, reflecting the dominant edge deployment scenario where a dedicated device serves a single inference stream.
- Each trial runs in a separate, reset inference session to prevent cache state carry-over.
- Trial order is randomized across model-task combinations.
- Per-trial metrics logged: `latency_ms`, `tokens_generated`, `throughput_tps`, `peak_npu_mem_gb`, `estimated_power_w`.
- Results are automatically appended to `Data_시계열.xlsx` (one row per trial), eliminating manual data transcription.

> **Replication note:** The paper reports N = 300 independent trials per model-task combination (100 samples × 3 seeds). Running all 9 models × 3 tasks × 300 trials = 8,100 total observations is computationally intensive. Reduce `--n_trials` for exploratory runs; use the full configuration to reproduce paper results exactly.

### Step 3: Analyze Results and Reproduce Paper Figures

Run the full statistical analysis pipeline to reproduce all hypothesis tests, figures, and tables from the paper.

```bash
python src/3_analyze_results.py
```

The script sequentially executes:

| Module | Hypothesis | Output |
|---|---|---|
| Latency distribution overview | H5 | Figure 1: Comparative Latency Distribution (Box + Violin) |
| OPS computation (Eq. 11) | H5 | Figure 2: Optimization Priority & Distributional Cost Map; Table A2 |
| Two-way ANOVA (model size × task) | H1 | Figure 3: Interaction Effect Plot |
| Natural cubic spline regression + LOOCV | H2 | Figure 4: Spline Fit & Residual Diagnostics |
| Levene's test + post-hoc comparisons | H3 | Figure 5: Variance Heterogeneity; Table A3 |
| Random Forest permutation importance + Pareto analysis | H4 | Figure 6: Feature Importance & Pareto Frontier |
| Integrated deployment framework | — | Figure 7: Decision Support Framework |

All statistical tests use α = 0.05 with Benjamini-Hochberg (BH) FDR correction applied across the full hypothesis family. Effect sizes are reported alongside p-values throughout.

To save figures to disk, uncomment the `plt.savefig()` lines in the script or pass `--save_figures` if implemented.

---

## Reproducing the OPS Score (Equation 11)

The OPS is computed in `3_analyze_results.py` as follows:

```python
import pandas as pd
import numpy as np

# Load per-model distributional statistics (output of Step 3 or from Data_정제.xlsx)
df = pd.read_excel("data/Data_정제.xlsx", sheet_name="OPS")

# Min-max normalize each instability metric across the 9 models
def minmax(x):
    return (x - x.min()) / (x.max() - x.min())

df["CV_norm"]        = minmax(df["CV"])
df["TailRatio_norm"] = minmax(df["P95_P50_Ratio"])
df["Skew_norm"]      = minmax(df["Skewness"].abs())
df["Kurt_norm"]      = minmax(df["Kurtosis"])

# Compute OPS with weights: CV=0.35, TailRatio=0.35, Skew=0.15, Kurt=0.15
df["OPS"] = (
    0.35 * df["CV_norm"] +
    0.35 * df["TailRatio_norm"] +
    0.15 * df["Skew_norm"] +
    0.15 * df["Kurt_norm"]
)

print(df[["model_name", "OPS"]].sort_values("OPS"))
```

> **Note:** Outlier Ratio (OR) is reported as a supplementary diagnostic in paper Table A2 but is excluded from OPS computation due to high collinearity with CV (Pearson r = 0.87), which would introduce redundancy in the weighting scheme. Sensitivity analysis confirmed that OPS rankings are robust to ±0.10 perturbations in each weight, with no rank-order changes among the top-3 and bottom-3 models (Appendix C, Table C1).

---

## Hardware and Software Configuration

| Category | Component | Specification | Rationale |
|---|---|---|---|
| Accelerator | Rebellions ATOM™ NPU | Primary evaluation platform (~15W TDP) | Specialized architecture representative of industrial edge AI constraints; enables assessment of hardware-specific performance degradation patterns |
| CPU | 8-core CPU | System integration | Balanced configuration preventing CPU bottlenecks from masking NPU stability characteristics |
| Memory | 16GB LPDDR5 RAM | System RAM | Sufficient capacity to evaluate memory pressure effects on inference stability |
| OS | Ubuntu 22.04 LTS | Stable, reproducible environment | Controlled software environment for stability testing |
| Frameworks | PyTorch 2.x, RBLN SDK, optimum-rebellions | Manufacturer-recommended stack | Industry-standard stack ensuring practical relevance of deployment findings |

GPU (NVIDIA RTX 4090, 350W TDP) and CPU (Intel Core i9, 125W TDP) platforms were available during experimental setup but are excluded from primary evaluation. This study is specifically designed to support **post-procurement model selection** within a committed edge hardware budget — reflecting the decision-making constraints practitioners face once an NPU-based device is deployed in a manufacturing environment. Cross-platform reference efficiency figures are provided in paper Appendix B (Table B1) for deployment context.

The benchmarking methodology, statistical framework, and OPS scoring approach are designed to be **platform-agnostic** and transferable to other NPU architectures (e.g., Qualcomm AI 100, Hailo-8, Apple Neural Engine), with platform-specific recalibration of numerical thresholds required before operational use.

---

## Scope and Limitations

- **Platform specificity:** Absolute latency, throughput, and memory utilization values are specific to the ATOM™ NPU. Numerical results will differ on other NPU architectures due to differences in memory hierarchy, compiler optimization, and parallelism models.
- **Model scope:** Evaluated sLLMs span 0.5B–3.2B parameters on text-based NLP tasks. Vision, audio, and multimodal architectures may exhibit different stability profiles.
- **H2 sample size:** The spline regression analysis is conducted on n = 9 aggregate data points (one per model). In-sample R² (0.891) likely overestimates true explanatory power; LOOCV R² (0.743) provides the conservative estimate. Results should be interpreted as directional evidence of non-linearity.
- **H4 causal claims:** Random Forest analysis identifies predictive associations between system features and throughput, not causal mechanisms. Controlled ablation experiments would be required to validate causal claims.

---

## License

The source code in this repository is licensed under the **Apache 2.0 License**. See the `LICENSE` file for details.

Model weights used in this study are subject to their respective licenses:

| Model Family | License |
|---|---|
| Llama 3.2 | Llama 3.2 Community License |
| Gemma | Gemma Terms of Use (Google) |
| Qwen2.5 | Apache 2.0 |
| EXAONE | EXAONE Open License |
| DeepSeek | DeepSeek Community License |
| OPT | OPT License (Meta/Fairseq) |
| Midm | Apache 2.0 |

---

## Data Availability

All raw benchmark logs, analysis scripts, and environment configurations used in this study are publicly available at:

**https://github.com/LEEYJ1021/Beyond-Accuracy-sLLM-NPU-Benchmark**
