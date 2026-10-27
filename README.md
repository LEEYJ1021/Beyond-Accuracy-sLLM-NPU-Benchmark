# Beyond Accuracy: Execution-Time Characterization of sLLM Inference on an Embedded NPU

Data, code, and analysis pipeline for an empirical study of **nine small language models (sLLMs; 0.5B–3.2B parameters)** running on the **Rebellions ATOM™ NPU**, a ~15 W edge accelerator. The study goes beyond mean latency and accuracy: it decomposes execution time into its components, characterizes latency distributions, and links timing indicators to **simulated deadline-miss rates** under non-preemptive and limited-preemption EDF scheduling.

**Keywords:** Embedded NPU · Small language models · Execution-time characterization · Latency distribution · Deadline-miss simulation

> **Scope.** All numerical results are specific to **one NPU (ATOM™), its compiler (`optimum-rebellions`), and nine models**. The measurement and simulation procedures are reusable on other platforms by replacing the measured samples; numerical thresholds must be recalibrated.

---

## Table of Contents

1. [Summary of Findings](#summary-of-findings)
2. [Measurement Protocol](#measurement-protocol)
3. [Evaluated Models](#evaluated-models)
4. [Evaluation Tasks](#evaluation-tasks)
5. [Repository Structure](#repository-structure)
6. [Data Description](#data-description)
7. [Setup](#setup)
8. [Usage](#usage)
9. [Key Results in Detail](#key-results-in-detail)
10. [Optimization Priority Score (OPS)](#optimization-priority-score-ops)
11. [Design Rule for Practitioners](#design-rule-for-practitioners)
12. [Scope and Limitations](#scope-and-limitations)
13. [Notes on Earlier Versions of This Repository](#notes-on-earlier-versions-of-this-repository)
14. [License](#license)

---

## Summary of Findings

Evaluated on 2,696 valid trials (9 models × 3 tasks × 100 prompts; four OPT-2.7B trials excluded):

1. **Output length drives the tail.** Within each model, latency is nearly linear in the number of generated tokens (R² = 0.93–0.97 in seven of the eight models whose output length varies). Fixing output length at 50 tokens reduces the P95/P50 ratio from up to **2.90** (pooled) to **1.2–1.4**. Heavy tails of pooled latency therefore mainly reflect *mixtures of output-length regimes* (label-only vs. free-running outputs), not hardware jitter alone.
2. **Residual jitter is small.** After removing generated-token, input-token, and task effects, residual variation is 3.5–9.6 % of the mean (residual P99/P50 1.12–1.43). Under this protocol this is an *upper bound* on platform-level jitter.
3. **The informative indicator depends on how deadlines are set.** With deadlines proportional to mean latency, the coefficient of variation tracks miss rates (ρ = 0.95 at 1.5× mean). With absolute deadlines (1–3 s), mean latency tracks them (ρ = 0.77–0.85). A one-sided Chebyshev–Cantelli argument explains this analytically.
4. **Scheduling matters more than the NPU.** In EDF simulations driven by measured latencies, non-preemptive execution yields miss rates of up to **26 %** at U ≤ 0.8; token-quantum preemption largely removes them, and **capping generated tokens** bounds the worst case without changing the scheduler.
5. **Task-specific budgets are warranted.** A uniform pooled-P95 budget is exceeded by 12–16 % of summarization requests while over-provisioning label-only NLU by 22.6–27.2×.
6. **Input-length dependence.** At fixed output length, summarization latency is close to linear in input length (0.13–0.57 ms per input token; R² = 0.60–0.82). No evidence of non-linear input scaling was found within models.
7. **Memory and throughput.** Peak NPU memory is strongly associated with throughput (Spearman ρ = −0.77), but it is collinear with parameter count (ρ = 0.92), so the two cannot be separated; this is an association, not a causal or predictive claim.
8. **OPS is a coarse descriptive screen.** It places Qwen2.5-0.5B at the stable end and Midm-2.0 / gemma-2b-it at the unstable end, but the ranking is weight-sensitive and performs no better than CV alone against simulated miss rates.

---

## Measurement Protocol

| Item | Setting |
|---|---|
| Accelerator | Rebellions ATOM™ NPU (~15 W envelope), default clock and power settings |
| Host | 8-core CPU, 16 GB LPDDR5, Ubuntu 22.04 LTS |
| Software | PyTorch 2.x, RBLN SDK, `optimum-rebellions` (`RBLNAutoModelForCausalLM`) |
| Batch size | 1 (single inference stream) |
| Generation | `max_new_tokens = 50`, temperature 0.7, random seed 42 |
| Prompts | 100 per task, sampled from the validation split with fixed shuffle seed 42; **every model receives the same 100 prompts per task in the same fixed order** |
| Repetitions | **Each prompt executed once** (no repeated runs per prompt) |
| Trials | 100 per model–task pair, 300 per model, **2,696 valid trials** in total |
| Excluded | 4 OPT-2.7B summarization trials returned an error (inputs of 2,056–2,171 tokens exceed the 2,048-token limit) |
| Timing metric | Wall-clock time from tokenized request submission to final token (prefill + decode; not instrumented separately) |
| Not controlled/logged | Temperature and ambient conditions |
| Peak NPU memory | A single per-model value from the deployment record (not sampled at runtime; cannot distinguish runtime use from compile-time reservation) |
| Power / KV cache | Per-model estimates; not used in any analysis |

Prompting is zero-shot with a system message defining role and output constraints, identical across models. Per-model compilation settings (maximum sequence length, parallel size, number of blocks) are recorded in the dataset and were not harmonized across models.

---

## Evaluated Models

| Model | Params (B) | Context length (dataset value) | Architecture | License |
|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5 | 32,768 | Decoder-only Transformer | DeepSeek Community |
| EXAONE-3.5-2.4B-Instruct | 2.4 | 32,768 | Decoder-only Transformer | EXAONE Open |
| gemma-2b-it | 2.0 | 8,192 | Decoder-only Transformer | Gemma (Google) |
| Llama-3.2-3B-Instruct | 3.2 | 128,000 | Transformer (GQA) | Llama 3.2 Community |
| Midm-2.0-Mini-Instruct | 2.0 | 4,096 | Decoder-only Transformer | Apache 2.0 |
| OPT-2.7B | 2.7 | 2,048 | Decoder-only Transformer | Meta (Fairseq) |
| Qwen2.5-0.5B-Instruct | 0.5 | 32,768 | Decoder-only Transformer | Apache 2.0 |
| Qwen2.5-1.5B-Instruct | 1.5 | 32,768 | Decoder-only Transformer | Apache 2.0 |
| Qwen2.5-3B-Instruct | 3.0 | 128,000 | Decoder-only Transformer | Apache 2.0 |

> Context-length values are those recorded in the public dataset and may reflect the **compiled maximum sequence length** rather than the vendor-stated model-card value.

---

## Evaluation Tasks

Three tasks span a complexity hierarchy relevant to timing-budget provisioning, analogous to mixed-criticality workloads:

| Task | Dataset | Output behavior | Quality metric (context only) |
|---|---|---|---|
| NLU | GLUE SST-2 (validation) | Bounded, often label-only | F1 |
| QA | SQuAD v2 (validation) | Short extractive generation | ROUGE-1 |
| Summarization | CNN/DailyMail 3.0.0 (validation) | Long auto-regressive decoding (≈50 tokens, token limit) | ROUGE-1 |

Task output quality is reported for context only and is not analysed further.

---

## Repository Structure

```
.
├── data/                              # Original dataset release (Excel)
│   ├── Data_정제.xlsx                 # Aggregated per-model / per-task summary statistics
│   ├── Data_허깅페이스.xlsx             # Model metadata (architecture, context length, license)
│   └── Data_시계열.xlsx               # Raw per-prompt latency logs (one row per trial)
│
├── src/                               # Original pipeline (legacy; see "Notes on Earlier Versions")
│   ├── 1_compile_models.py            # Compile Hugging Face models to ATOM™ NPU binaries
│   ├── 2_benchmark_inference.py       # Run inference and log per-trial metrics
│   └── 3_analyze_results.py           # Original statistical analysis and figures
│
├── revision_R1/                       # Current analysis pipeline (results reported in the paper)
│   ├── data/                          # CSV versions of the datasets + derived input features
│   │   ├── Data_시계열.csv             # Raw per-prompt latency logs
│   │   ├── Data_정제.csv               # Aggregated statistics
│   │   ├── Data_허깅페이스.csv          # Model metadata
│   │   ├── input_chars_only.csv       # Derived: input character counts
│   │   └── input_features.csv         # Derived: per-prompt input-token counts (model tokenizers)
│   ├── run_all.py                     # Main entry point: statistics, tables, figures
│   ├── prep_data.py                   # Data loading and preprocessing
│   ├── input_features.py              # Input-length features using each model's own tokenizer
│   ├── within_model_analysis.py       # Within-model latency decomposition (Eq. 7), input dependence
│   ├── followup_analysis.py           # Follow-up analyses (indicators vs. deadline misses, bounds)
│   ├── followup2.py                   # Additional follow-up analyses
│   ├── additional_analysis.py         # Supplementary analyses
│   ├── rel_deadline.py                # Relative vs. absolute deadline analysis
│   ├── cpu_extra.py                   # Supplementary computations
│   ├── figlib.py                      # Shared plotting helpers
│   ├── make_figs.py                   # Figure generation
│   ├── make_figs2.py                  # Figure generation (additional figures)
│   └── patch_figs.py / patch_figs2.py / patch_figs3.py   # Figure post-processing patches
│
├── requirements.txt
├── LICENSE                            # Apache-2.0
└── README.md
```

---

## Data Description

**`Data_시계열` (raw logs; one row = one prompt execution).** Key fields:

| Column | Description |
|---|---|
| `model_name` | Model identifier |
| `task` | NLU / QA / Summarization |
| `latency_ms` | End-to-end wall-clock latency (ms) |
| `tokens_generated` | Number of output tokens |
| `throughput_tps` | Tokens per second |
| `peak_npu_mem_gb` | Per-model peak NPU memory from the deployment record (GB) |
| `estimated_power_w` | Per-model power estimate (W); not used in analyses |

Additional compilation fields (maximum sequence length, parallel size, number of blocks, compile times) are recorded in the dataset. Verify exact column names against the file headers (`head -1 revision_R1/data/Data_시계열.csv`).

**`input_features.csv`** holds per-prompt input-token counts computed with each model's own tokenizer on CPU and aligned to prompts by sample index.

---

## Setup

**Prerequisites**

- Ubuntu 22.04 LTS, Python 3.9+
- For *measurement* (re-running inference): Rebellions ATOM™ NPU with the RBLN SDK and `optimum-rebellions` (licensed separately; follow the vendor setup guide)
- For *analysis only*: no NPU required; the CSV/Excel logs are sufficient

```bash
git clone https://github.com/LEEYJ1021/Beyond-Accuracy-sLLM-NPU-Benchmark.git
cd Beyond-Accuracy-sLLM-NPU-Benchmark

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### A. Reproduce the analysis (no NPU needed)

```bash
cd revision_R1
python prep_data.py            # preprocess raw logs
python input_features.py       # (optional) regenerate input-token features; requires tokenizers
python run_all.py              # statistics, simulations, tables, figures
```

Scripts expect the CSV files under `revision_R1/data/` and are meant to be run from inside `revision_R1/`. Outputs (tables, figures) are written to the output folders defined in the scripts.

### B. Re-collect measurements (requires ATOM™ NPU)

```bash
python src/1_compile_models.py          # compile models; OOM handling reduces max_seq_len and retries
python src/2_benchmark_inference.py \
    --model_dir compiled_models/<model_name> \
    --task [nlu|qa|summarization] \
    --output_file data/Data_시계열.xlsx
```

Use the protocol in [Measurement Protocol](#measurement-protocol): batch size 1, `max_new_tokens=50`, temperature 0.7, seed 42, the same fixed prompts and order for every model, one execution per prompt.

---

## Key Results in Detail

### 1. Latency decomposition

Latency of prompt *j* for model *m* on task *k*:

```
T_mkj = α_mk + b_m · N_mkj + c_m · L_mkj + ε_mkj
```

where *N* is generated tokens, *L* input tokens, *b_m* the marginal per-output-token cost, and *c_m* a regression proxy for per-input-token (prefill) cost.

| Model | Marginal cost *b_m* (ms/token) |
|---|---|
| Qwen2.5-0.5B | 13.5 |
| Qwen2.5-1.5B | 29.1 |
| gemma-2b-it | 30.2 |
| EXAONE-3.5-2.4B | 30.3 |
| DeepSeek-R1-Distill-1.5B | 30.4 |
| Midm-2.0 | 48.0 |
| Llama-3.2-3B | 48.0 |
| Qwen2.5-3B | 49.1 |
| OPT-2.7B | undefined (always 50 tokens) |

Per-token cost differs ≈3.6× across models and is nearly constant within a model. Midm-2.0 (2.0B) has a per-token cost close to the 3B models, so parameter count alone does not determine it. Generated tokens, input tokens, and task jointly explain 98–99 % of latency variance in seven of nine models (median Shapley shares: 0.52 generated tokens, 0.17 input tokens, 0.30 task).

### 2. Latency distribution statistics (2,696 trials)

| Model | n | Mean (ms) | CV | Skew | Excess kurt. | P95/P50 | OPS |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | 300 | 611.2 | 0.177 | −0.04 | 4.09 | 1.33 | 0.186 |
| DeepSeek-R1-Distill-Qwen-1.5B | 300 | 1,205.3 | 0.433 | −1.02 | −0.31 | 1.25 | 0.270 |
| Llama-3.2-3B-Instruct | 300 | 1,866.9 | 0.474 | −1.00 | −0.42 | 1.26 | 0.287 |
| Qwen2.5-1.5B-Instruct | 300 | 1,025.9 | 0.624 | −0.70 | −1.28 | 1.21 | 0.306 |
| OPT-2.7B | 296 | 1,462.5 | 0.154 | 1.76 | 3.00 | 1.42 | 0.315 |
| Qwen2.5-3B-Instruct | 300 | 1,996.3 | 0.459 | −1.27 | 0.22 | 1.25 | 0.318 |
| EXAONE-3.5-2.4B-Instruct | 300 | 932.1 | 0.745 | −0.11 | −1.49 | 1.51 | 0.375 |
| gemma-2b-it ★ | 300 | 753.2 | 0.816 | 0.29 | −1.31 | 2.47 | 0.631 |
| Midm-2.0-Mini-Instruct ★ | 300 | 1,207.8 | 0.830 | 0.34 | −1.30 | 2.90 | 0.731 |

★ OPS > 0.6 is a descriptive cut only. Normality is rejected for every model (Shapiro–Wilk p < .001), which is expected at this sample size. The informative contrast is between output-length regimes: pooled distributions show a short plateau (label-only outputs) plus an upper branch (free-running outputs); 50-token trials show a single narrow mode. OPT-2.7B has very low CV only because every output runs to the 50-token limit, so a dispersion index must be read together with latency level.

### 3. Indicators vs. simulated deadline-miss rates (Spearman ρ, n = 9)

| Indicator | 1.5× mean | 2× mean | 3× mean | 1 s | 2 s | 3 s |
|---|---|---|---|---|---|---|
| Mean latency | −0.12 | −0.38 | 0.14 | 0.85 | 0.84 | 0.77 |
| CV | **0.95** | **0.82** | 0.55 | −0.47 | 0.18 | 0.17 |
| P95/P50 | 0.48 | 0.82 | 0.55 | −0.47 | −0.02 | −0.14 |
| OPS | 0.73 | 0.82 | 0.55 | −0.12 | 0.43 | 0.21 |

|ρ| ≥ 0.68 is required for p < .05 with n = 9; Benjamini–Hochberg adjustment across 28 tests. Cells at 2× and 3× rest on few non-zero miss rates (six of nine models have zero misses at 2×), so the table should be read as a pattern across deadline definitions, not as cell-wise evidence. In a utilization-matched design (U = 0.7), miss-rate correlations are ρ = 0.92 (CV), 0.70 (OPS), and −0.17 (mean latency).

### 4. Periodic-task EDF simulation (U ≤ 0.8)

A discrete-event simulation of one NPU executing three periodic tasks (NLU, QA, summarization), deadlines equal to periods, random release phases, execution times drawn with replacement from measured samples. Period sets: P1 = 1/4/8 s, P2 = 2/8/16 s, P3 = 3/12/24 s. Policies: non-preemptive EDF and limited-preemption EDF (quantum Q = 1, 4, 16 × per-token cost; resume overhead 0, 20, 50 ms); 600 s per configuration, 20 replications.

| Period set | Model | Non-preemptive miss | Preemptive miss | Blocking ratio B |
|---|---|---|---|---|
| P1 | Qwen2.5-1.5B | 26.3 % | 11.9–13.4 % | 2.01 |
| P1 | Midm-2.0 | 20.1 % | 0.0 % | 3.57 |
| P1 | EXAONE-2.4B | 11.8 % | 0.0 % | 2.23 |
| P1 | Qwen2.5-0.5B | 7.6 % | ≤ 0.2 % | 0.87 |
| P1 | gemma-2b-it | 5.1 % | ≤ 0.2 % | 1.99 |
| P3 | Llama-3.2-3B | 12.1 % | 0.0–1.1 % | 1.03 |
| P3 | Qwen2.5-3B | 12.0 % | 0.0–1.0 % | 1.15 |

(P2 and the remaining P3 rows, mostly < 5 %, are in the paper's Table 6.) The mechanism is **non-preemptive blocking**: a ~1.5–2.5 s summarization job that has started cannot be interrupted, so NLU jobs with 1 s deadlines miss. The blocking ratio `B = P99(summarization latency) / p_min` is a necessary-condition indicator (non-preemptive misses occur in all ten model–period rows with B > 1), not a predictor of magnitude. Results are insensitive to the quantum and to resume overheads of 0–50 ms. Qwen2.5-1.5B needs an output cap or longer deadline, not a scheduling change, because ~14 % of its NLU jobs exceed 1 s by themselves.

> The simulation assumes token-granularity preemption is available at the stated overhead, which idealizes the RBLN runtime. It provides statistical evidence under stated assumptions, **not a schedulability proof**.

### 5. Latency bounds (hold-out validation)

| Bound | Exceedance on held-out half |
|---|---|
| Empirical maximum | 1.9 % |
| Empirical maximum × 1.1 | 0.5 % |
| P99 | 2.5 % |
| 1.5 × P50 | 5.1 % |
| Model-based (`a_task + 50·b`, residual margin, ×1.1) | 0.3 % (but over-provisions label-only NLU by 17–23×) |

Because latency is near-linear in generated tokens, **capping `max_new_tokens`** bounds the worst case via `a + b·cap` without changing the scheduler.

### 6. Other tests

- **H1 (size × task interaction).** F(2, 2690) = 38.05, p < .001, η²p = 0.028; the summarization-to-NLU slope ratio is 1.76 (bootstrap 95 % CI 1.54–2.09). With model-clustered standard errors the interaction is not significant, so it is an observation-level result for these nine models.
- **H2 (input dependence).** Linear in input length within model; spline, piecewise-linear, and staircase forms are not preferred. Per-token cost shows no detectable association with dataset context length (ρ = 0.26, p = .54; low power).
- **H3 (task budgets).** Levene W = 6.04, p = .002, Cohen's f = 0.067 (small). The case for task-specific budgets rests on provisioning consequences, not on the variance test.
- **H4 (memory–throughput).** ρ = −0.77 (p = .016); −0.74 without Llama-3.2-3B, −0.67 without Qwen2.5-0.5B; partial ρ = −0.78 controlling for parameters (depends on throughput pooling). Random-forest leave-one-model-out R² is low (0.13), so it is not used as a predictive model.

---

## Optimization Priority Score (OPS)

OPS is a **descriptive** composite of four min–max normalized distribution statistics (normalized across the nine evaluated models):

```
OPS_i = 0.35·CV*_i + 0.35·TailRatio*_i + 0.15·|Skew*_i| + 0.15·Kurt*_i
```

It is **not derived from schedulability theory**. Known properties and limits:

- Values are relative to the nine models evaluated and are **not comparable across studies**.
- It uses |skewness|, so the sign is discarded; bimodal mixtures produce negative excess kurtosis and small |skewness|, so the moment components can rate a bimodal model as *more* stable.
- The ranking is **weight-sensitive**: under 2,000 random Dirichlet(1,1,1,1) weight vectors, Midm-2.0 is least stable with probability 0.56 and OPT-2.7B with 0.36; with equal weights EXAONE moves to rank 1. Only the extremes under baseline weights are stable (hierarchical-bootstrap rank intervals: Qwen2.5-0.5B 1–1; EXAONE 7–7; gemma 8–8; Midm 9–9).
- It performs no better than CV alone against simulated miss rates.
- A companion **OPS-core = 0.5·CV\* + 0.5·(P95/P50)\*** uses only the two components that track deadline misses (Spearman ρ = 0.82 with baseline OPS; preserves the extremes).

Use OPS as a coarse first screen, then confirm with simulation on the target task set and deadlines.

```python
import pandas as pd

df = pd.read_csv("revision_R1/data/<per-model-statistics>.csv")  # per-model CV, P95/P50, skew, kurtosis

def minmax(x):
    return (x - x.min()) / (x.max() - x.min())

df["OPS"] = (0.35 * minmax(df["CV"])
           + 0.35 * minmax(df["P95_P50_Ratio"])
           + 0.15 * minmax(df["Skewness"].abs())
           + 0.15 * minmax(df["Kurtosis"]))
print(df[["model_name", "OPS"]].sort_values("OPS"))
```

(Outlier ratio is reported as a diagnostic but excluded from OPS.) Recompute statistics from the raw logs with `run_all.py` rather than from any older aggregated sheet.

---

## Design Rule for Practitioners

1. **Set deadlines per task**, not per model.
2. **Choose an output cap** `N_cap = ⌊(D − α_mk − c_m·L_max − δ_q) / b_m⌋` from the per-token cost, task intercept, and input-length term (e.g., N_cap = 42 for Llama-3.2-3B, 34 for Midm-2.0, 35 for Qwen2.5-3B at a 3 s summarization deadline).
3. **Choose a preemption quantum** Q as a small multiple of *b_m* (results are insensitive to Q between 1 and 16 tokens), if the runtime supports token-boundary preemption.
4. **Select the indicator by deadline definition:** CV and P95/P50 for deadlines proportional to mean latency; mean latency for absolute deadlines.
5. **Validate the bound by hold-out** on prompts representative of the deployment distribution, and re-measure whenever the model, prompt template, or decoding settings change.

---

## Scope and Limitations

- **Platform.** All absolute numbers are ATOM™-specific. We expect the decomposition into output length and per-token cost, the dependence of indicator usefulness on the deadline definition, and the benefit of preemption and output caps to transfer; latencies, per-token costs, and the memory–throughput association require recalibration.
- **Measurement design.** Each prompt was run once with temperature-0.7 sampling; one prompt template; per-model compilation settings were not harmonized; temperature was not logged. Prefill and decode were not instrumented separately; input effects are estimated from natural variation of input length across datasets (observational). Output-length regimes are properties of the model–prompt–decoding configuration, not of the NPU alone.
- **Memory and power.** Peak memory is a deployment-record value that may reflect compile-time allocation; power is estimated.
- **Simulation.** Independent periodic jobs on one NPU, EDF only, three period sets, 100 prompts per task, U ≤ 0.8, idealized token-granularity preemption with resume overhead up to 50 ms; real KV-cache save/restore costs may be larger. Residuals are serially correlated in 14 of 27 series (a block-bootstrap sensitivity check addresses the independence assumption).
- **Model-level inference.** With nine models, model-level results are exploratory (|ρ| ≈ 0.68 needed for p < .05).
- **Bounds** are validated by hold-out within the same prompt sample.
- **Coverage.** Text-only NLP tasks on 0.5B–3.2B models; safety and security dimensions are not studied.

**Future work:** other NPUs (e.g., Qualcomm AI 100, Hailo-8); controlled input-length sweeps with TTFT/TPOT instrumentation and repeated runs per prompt; real token-boundary preemption in the runtime; probabilistic response-time analysis and extreme-value pWCET; memory-allocation ablations at fixed parameter count.

---

## Notes on Earlier Versions of This Repository

Earlier versions of this README and of the scripts in `src/` described the data collection as 300 independent trials per model–task pair (100 samples × 3 seeds) with randomized trial order and thermal management, and reported a different set of statistics and OPS values. **This did not accurately describe the collected data.** The current protocol is the one in [Measurement Protocol](#measurement-protocol), and all statistics, OPS values, and figures in `revision_R1/` were re-derived from the 2,696 trials in the raw logs. Where numbers in older files (`src/3_analyze_results.py` outputs, aggregated Excel sheets) differ from this README, **this README and `revision_R1/` take precedence.** The nine-point cross-model spline analysis of context length (LOOCV R² = 0.743 vs. 0.501 linear) is retained only as an exploratory analysis; it conflates architecture, attention type, and parameter count with context length and supports no inference about context scaling.

---

## License

The source code in this repository is licensed under the **Apache 2.0 License** (see `LICENSE`). Model weights are subject to their own licenses:

| Model Family | License |
|---|---|
| Llama 3.2 | Llama 3.2 Community License |
| Gemma | Gemma Terms of Use (Google) |
| Qwen2.5 | Apache 2.0 |
| EXAONE | EXAONE Open License |
| DeepSeek | DeepSeek Community License |
| OPT | OPT License (Meta/Fairseq) |
| Midm | Apache 2.0 |
