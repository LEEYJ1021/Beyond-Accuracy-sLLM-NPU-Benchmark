# Beyond Accuracy: A Multi-Dimensional Benchmarking of sLLMs on a Specialized NPU

This repository contains the official source code and analysis scripts for the research paper, **"Beyond Accuracy: A Multi-Dimensional Benchmarking of Small-to-Medium Language Models on a Specialized NPU Reveals Critical Speed-Stability Trade-offs."**

## Abstract

This study establishes a comprehensive benchmarking framework to evaluate small-to-medium Language Models (sLLMs) on a specialized Neural Processing Unit (NPU), moving beyond conventional accuracy metrics. By testing nine sLLMs on the Rebellions ATOM™ NPU across 300 independent trials per model-task combination, we empirically validated five core hypotheses. Our analysis confirmed a statistically significant interaction between model size and task complexity on inference latency (F(2, 2683) = 41.55, p < .001, partial η² = 0.030). We also revealed a strongly non-linear relationship between context length and latency, where a natural cubic spline model achieved 58.1% greater explanatory power (ΔR²) over a linear model. A Random Forest analysis identified peak NPU memory utilization as the dominant driver of token throughput (Permutation Importance = 0.458), underscoring the criticality of memory optimization. We demonstrated significant heterogeneity in latency variance across tasks (Levene's W = 5.82, p = .003), with NLU tasks exhibiting the highest instability. Distributional analysis confirmed non-normal latency profiles with heavy tails, leading to a novel Optimization Priority Score for guiding deployment. These results expose a critical trade-off between speed and stability, providing a data-driven foundation for effective model selection on specialized hardware.

**Keywords**: On-Device AI; Small Language Models (sLLMs); Neural Processing Unit (NPU) Benchmarking; Inference Latency Distribution; Performance Stability; Pareto Efficiency

---

## Repository Structure

The repository is organized to separate data, source code, and results.

```
.
├── data/
│   ├── Data_정제.xlsx           # Main aggregated data for analysis
│   ├── Data_허깅페이스.xlsx       # Hugging Face metadata for models
│   └── Data_시계열.xlsx         # Raw time-series latency data from benchmarks
├── results/
│   └── (Generated plots and tables will be saved here)
├── src/
│   ├── 1_compile_models.py      # Script to compile Hugging Face models for the NPU
│   ├── 2_benchmark_inference.py # Script to run inference benchmarks and gather metrics
│   └── 3_analyze_results.py     # Script to perform statistical analysis and generate paper figures
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## Setup

Follow these steps to set up the environment for running the benchmarks and analysis.

1.  **Clone the Repository**
    ```bash
    git clone <repository-url>
    cd <repository-name>
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    Install all required Python packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `optimum[rbln]` requires the Rebellions SDK to be properly installed and configured in your environment. Please follow the official Rebellions documentation for this step.*

4.  **Prepare Data**
    Place the three required Excel files (`Data_정제.xlsx`, `Data_허깅페이스.xlsx`, `Data_시계열.xlsx`) into the `data/` directory. The analysis script is configured to read from this location.

---

## Usage: Experimental Workflow

The project is divided into three main stages: model compilation, benchmarking, and data analysis.

### Step 1: Compile Models for NPU

The first step is to convert the pre-trained models from Hugging Face into a format optimized for the Rebellions ATOM™ NPU.

-   **Modify `src/1_compile_models.py`**: Review the `models_config` dictionary in the script and ensure the models you wish to test are included and correctly configured.
-   **Run the script**:
    ```bash
    python src/1_compile_models.py
    ```
-   **Output**: This will create a directory for each compiled model containing the NPU-ready binaries. The script includes robust Out-Of-Memory (OOM) handling, automatically adjusting parameters like `max_seq_len` and retrying compilation.

### Step 2: Run Inference Benchmarks

After compiling, run inference benchmarks for each model on the desired tasks (NLU, QA, Summarization).

-   **Configure `src/2_benchmark_inference.py`**:
    -   Set the `MODEL_DIR` variable to the path of a compiled model from Step 1.
    -   Uncomment the appropriate `RBLN...ForCausalLM` class corresponding to the model architecture.
-   **Run the benchmark for each model-task combination**:
    ```bash
    python src/2_benchmark_inference.py
    ```
-   **Collect Data**: The script will print detailed performance metrics to the console for each run. **This output must be manually collected and organized** into the `Data_시계열.xlsx` file. Each row in the spreadsheet should represent a single inference trial.

### Step 3: Analyze Results and Reproduce Paper Findings

Once the raw data is collected, run the main analysis script to process the data, perform statistical tests, and generate the figures and tables presented in the paper.

-   **Ensure data files are in the `data/` directory.**
-   **Run the analysis script**:
    ```bash
    python src/3_analyze_results.py
    ```
-   **Output**: The script will print detailed statistical outputs to the console and display plots for each analysis step (e.g., Latency Distribution, ANOVA, Pareto Frontier, Feature Importance). The generated plots can be saved by uncommenting `plt.savefig()` lines within the script.

---

## License

The code in this repository is licensed under the Apache 2.0 License. See the `LICENSE` file for more details. Model weights used in this study are subject to their own licenses (e.g., Llama 3.2 Community License, Gemma License).
