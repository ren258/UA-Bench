# UA-Bench Evaluation Code

This repository contains the **evaluation code for UA-Bench**, a benchmark designed to assess large language models’ ability to distinguish **data uncertainty** and **model uncertainty** in question answering.

The code is used to evaluate the datasets provided under the `data/` directory and to produce both per-instance outputs and aggregated evaluation metrics.

---

## Directory Structure

```
UA-Bench/
├── code/                    # Evaluation and analysis code
│   ├── evaluate.py           # Main evaluation entry
│   ├── io_utils.py           # File I/O utilities
│   ├── prompts.py            # Prompt templates used in evaluation
│   ├── parsing.py            # Output parsing logic
│   ├── judge.py              # LLM-as-a-judge prompts and parsing
│   ├── metrics.py            # Evaluation metrics
│   ├── parallel_api_call.py  # Parallel API invocation utilities
│   ├── gpt_api.py            # OpenAI / network API interface
│   ├── vllm_api.py           # Local vLLM inference interface
│   ├── start_vllm.sh         # Helper script to launch vLLM server
│   └── summarize_attr_results.py  # Result aggregation script
│
├── data/                    # UA-Bench datasets
│   ├── gaia.json
│   ├── gsm8k-mip.json
│   ├── math-mip.json
│   ├── musique_1000.json
│   ├── olympiadbench-math.json
│   └── selfaware.json
```

---

## Overview

* This folder provides the **official evaluation pipeline** for UA-Bench.
* The evaluation is performed on the datasets located in the `data/` directory.
* The main evaluation entry point is **`evaluate.py`**, which orchestrates prompt construction, model inference, output parsing, judging, and metric computation.

---

## Code Components

### `evaluate.py`

* The **main entry point** for evaluation.
* Loads datasets, builds prompts, performs model inference, parses outputs, invokes judging (if enabled), and computes evaluation metrics.
* Produces per-instance result files for further analysis.

---

### `io_utils.py`

* Provides utility functions for:

  * Loading and saving JSON / JSONL datasets
  * Managing intermediate and final output files
* Centralizes all file I/O logic to ensure consistent data handling.

---

### `prompts.py`

* Contains the **three prompt variants** evaluated in the paper:

  * **Direct Answer**
  * **Abstention-Only**
  * **Uncertainty Attribution**
* Each prompt explicitly specifies the expected output format and uncertainty decision rules.

---

### `parsing.py`

* Implements robust parsing of model outputs.
* Extracts:

  * Final answers
  * Abstention tokens
  * Uncertainty attribution labels
* Designed to handle malformed outputs and multiple boxed answers conservatively.

---

### `judge.py`

* Implements the **LLM-as-a-judge** component.
* Defines:

  * Judging prompts
  * Parsing logic for judge model outputs
* Used to determine answer correctness when string matching is insufficient.

---

### `metrics.py`

* Implements all evaluation metrics reported in the paper, including:

  * Answer accuracy
  * Data-uncertainty recall
  * Model-uncertainty statistics
  * Derived metrics for uncertainty attribution performance
* Metrics are computed from parsed predictions and ground-truth annotations.

---

### `parallel_api_call.py`

* Provides utilities for **parallel model inference**.
* Supports both:

  * **Local inference via vLLM** (`vllm_api.py`)
  * **Network-based APIs** (`gpt_api.py`)
* Enables efficient large-scale evaluation with configurable concurrency.

> **Note:**
> For network-based API calls (e.g., OpenAI-compatible APIs), the corresponding `API_KEY` must be set via environment variables.

---

### `summarize_attr_results.py`

* Aggregates multiple per-run or per-dataset output files.
* Produces **final summarized evaluation results** used for tables and analysis in the paper.

---

## Typical Evaluation Workflow

1. Run `evaluate.py` on one or more datasets in `data/`
2. Obtain per-instance prediction and judging result files
3. Use `summarize_attr_results.py` to aggregate results across datasets and models

---

## Notes

* This codebase is intended **solely for evaluation** and does not include model training.
* All datasets follow the UA-Bench unified JSON schema with explicit uncertainty annotations.
* The evaluation pipeline is model-agnostic and supports both open-source and closed-source LLMs.
