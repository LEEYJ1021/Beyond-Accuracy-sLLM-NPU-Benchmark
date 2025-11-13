# Enhancing Software Dependability: A Reliability-Centric Benchmarking Framework for Small Language Models on Specialized NPUs in Edge AI Applications

This repository contains the official source code and analysis scripts for the research paper, **"Beyond Accuracy: A Multi-Dimensional Benchmarking of Small-to-Medium Language Models on a Specialized NPU Reveals Critical Speed-Stability Trade-offs."**

## Abstract

This research introduces a comprehensive benchmarking framework to evaluate the software dependability of small language models (sLLMs) deployed on specialized NPUs for Edge AI applications. Moving beyond mere accuracy, the study rigorously analyzes the trade-offs between computational efficiency and operational reliability. Key findings reveal that NPU memory usage is the dominant predictor of system reliability, and that performance degradation follows strongly non-linear patterns. The work also uncovers significant heterogeneity in performance variance and non-normal latency profiles, providing data-driven methodologies and practical solutions for the design of dependable, mission-critical AI systems.

**Keywords**: Software Dependability, Edge AI Reliability, NPU Benchmarking, Mission-Critical Systems, Performance Stability, AI Safety.

---

## Repository Structure

The repository is organized to separate data, source code, and results.

```
.
├── data/
│   ├── Data_정제.xlsx           # Main aggregated data for analysis
│   ├── Data_허깅페이스.xlsx       # Hugging Face metadata for models
│   └── Data_시계열.xlsx         # Raw time-series latency data from benchmarks
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
