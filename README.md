# UnLeakedTestBench

**Benchmarking LLMs for Unit Test Generation from Real-World Functions**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

UnLeakedTestBench is a rigorous benchmark for evaluating Large Language Models (LLMs) on function-level unit test generation. It addresses critical limitations in existing benchmarks by providing:

- **3,909 real-world Python functions** with high cyclomatic complexity (≥10)
- **Decontaminated dataset** that mitigates test case leakage from LLM training data
- **Paired benchmark design** with PreLeakedTestBench for controlled contamination analysis

Our evaluation of 12 state-of-the-art LLMs reveals that UnLeakedTestBench presents significantly greater challenges than existing benchmarks, with models achieving only:

- **41.32%** accuracy (vs. 91.79% on TestEval)
- **45.10%** statement coverage (vs. 92.18% on TestEval)
- **30.22%** branch coverage (vs. 82.04% on TestEval)
- **40.21%** mutation score (vs. 49.69% on TestEval)

## Key Features

### 1. Real-World Relevance

- Functions sourced from **The Stack v2**, a large corpus of permissively licensed code
- Function-level tasks that reflect actual software development practices
- Self-contained functions with minimal external dependencies

### 2. Controlled Complexity

- All functions have **cyclomatic complexity ≥ 10** (avg. 14.87)
- Filters out trivial "toy" examples that inflate performance metrics
- Ensures meaningful evaluation of LLM reasoning capabilities

### 3. Data Decontamination

- Rigorous filtering to exclude functions with publicly available test cases
- **UnLeakedTestBench**: 3,909 decontaminated functions for measuring true generalization
- **PreLeakedTestBench**: 18,169 functions (including leaked) for contamination impact analysis

### 4. Comprehensive Evaluation Metrics

- **Pass@k**: Test generation accuracy
- **LCov@k / BCov@k**: Statement and branch coverage
- **Mut@k**: Fault detection via mutation testing

## Dataset Statistics

| Benchmark | # Functions | Avg. Cyclomatic Complexity | Data Status |
|-----------|------------|---------------------------|-------------|
| UnLeakedTestBench | 3,909 | 14.87 | Decontaminated |
| PreLeakedTestBench | 18,169 | 14.52 | Mixed (includes leaked) |
| TestEval | 210 | - | Not decontaminated |

## Installation

### Environment Setup

```bash
# Create conda environment
conda create -n unleaked python=3.12
conda activate unleaked

# Install required packages
pip install cosmic-ray pytest pytest-cov
```

### Additional Dependencies

For running the full evaluation pipeline, you may need:

```bash
pip install openai anthropic transformers torch
```

## Usage

### Task Definition

UnLeakedTestBench uses a **K-query iterative test generation task**:

1. **Round 1**: LLM receives the function under test (FUT) and generates 1 test case
2. **Round i (2 ≤ i ≤ K)**: LLM receives FUT + all previously generated tests, generates 1 new distinct test

This iterative process encourages test diversity and comprehensive coverage.

## Evaluation Metrics

### 1. Test Generation Accuracy (Pass@k)

Proportion of correct, executable test cases:

```text
Pass@k = (Σ Correct_Tests_i) / (N × K)
```

where `N` is the number of functions and `K` is the number of queries.

### 2. Code Coverage

- **LCov@k**: Percentage of executable lines covered by first k generated tests
- **BCov@k**: Percentage of execution branches covered by first k generated tests
- **ΔLCov@k / ΔBCov@k**: Incremental coverage improvement

### 3. Mutation Score (Mut@k)

Fault detection capability using Cosmic-Ray mutation testing:

```text
Mut@k = (Σ Killed_Mutants_i) / N
```

Higher mutation scores indicate better bug-finding ability.

## Experimental Results

### Performance Comparison (Average across 12 LLMs)

| Metric | UnLeakedTestBench | PreLeakedTestBench | TestEval |
|--------|-------------------|-------------------|----------|
| Pass@1 | 41.32% | 47.07% | 91.79% |
| LCov@k | 45.10% | 55.13% | 92.18% |
| BCov@k | 30.22% | 40.07% | 82.04% |
| Mut@k | 40.21% | 50.80% | 49.69% |

### Key Findings

1. **UnLeakedTestBench is significantly more challenging** than existing benchmarks
2. **Strong correlation with code generation ability** (ρ = 0.79, p = 0.002)
3. **Data contamination inflates performance**, particularly for branch coverage
4. **Cyclomatic complexity is a valid difficulty indicator** for test generation

## Evaluated Models

We evaluated 12 state-of-the-art LLMs:

| Family | Models |
|--------|--------|
| CodeLlama | CodeLlama-7b-Instruct-hf |
| Seed-Coder | Seed-Coder-8B-Instruct |
| DeepSeekCoder | deepseek-coder-{1.3b, 6.7b, 33b}-instruct |
| Gemma-3 | gemma-3-{4b, 12b, 27b}-it |
| Qwen2.5-Coder | Qwen2.5-Coder-{7B, 14B, 32B}-Instruct |
| Microsoft Phi-4 | Phi-4-mini-instruct |


## Data Preservation Notice

**Important**: To preserve benchmark integrity and prevent test case leakage into future LLM training sets, we do **not** release the ground-truth tests. Instead, we provide:

- Complete function source code for evaluation
- Comprehensive evaluation results for all benchmarked models
- Analysis scripts for comparing new models against our findings

This approach allows researchers to evaluate new models while maintaining the benchmark's scientific validity.


## Contributing

We welcome contributions to improve UnLeakedTestBench! Areas of interest:

- Extending to other programming languages
- Adding new evaluation metrics
- Improving decontamination techniques
- Evaluating additional LLMs

Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **The Stack v2** for providing the source code corpus
- **Cosmic-Ray** for mutation testing framework
- All contributors and researchers who helped validate the benchmark

## Contact

For questions, issues, or collaborations:

- **Dong Huang**: <dhuang@nus.edu.sg>
- **Mingzhe Du** (Corresponding Author): <mingzhe@nus.edu.sg>



**UnLeakedTestBench** - Setting a higher bar for LLM test generation evaluation
