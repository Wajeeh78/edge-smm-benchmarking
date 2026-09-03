# Edge Benchmarking: State Space Models vs. Transformers on Apple Silicon

An empirical investigation analyzing the performance trade-offs between State Space Models (Mamba-130M) and traditional Transformers (GPT-2 124M) when executed on consumer edge hardware featuring unified memory architectures (Apple Silicon M4 via Metal Performance Shaders).

## Key Empirical Findings

* **Short-Context Throughput:** Due to mature matrix multiplication (GEMM) hardware optimizations, standard Transformers achieve higher generation velocity in low-context regimes (68.35 tokens/sec vs. 32.00 tokens/sec).
* **Long-Context Scaling & Stability:** Traditional Transformers encounter rigid architectural limits (crashing at 1024 tokens due to absolute positional embedding bounds), whereas State Space Models exhibit stable, linear scaling up to 2,000+ tokens on unified memory.

## Project Structure

* `run_mamba.py` - Initializes and validates the 130M Mamba architecture on Apple MPS.
* `benchmark.py` - Measures baseline token generation throughput.
* `benchmark_transformer.py` - Evaluates equivalent GPT-2 baseline performance.
* `benchmark_scaling.py` - Stress-tests context length expansion from 100 to 2,000 tokens.

## Reproducing the Results

1. Clone the repository and configure a local virtual environment.
2. Install dependencies:
   ```bash
   pip install torch torchvision torchaudio transformers datasets pandas
