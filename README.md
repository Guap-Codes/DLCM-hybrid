# 🧠 Dynamic Large Concept Models (DLCM)

**A high-performance, hybrid PyTorch/Rust implementation of Dynamic Large Concept Models.**

This repository contains a mathematically rigorous implementation of the DLCM architecture, featuring a fully differentiable PyTorch training pipeline with **Decoupled µP** (Maximal Update Parameterization), and a blazing-fast, hardware-accelerated **Rust/CUDA inference engine** built on `candle-core`.

## ✨ Key Features

- **Dynamic Semantic Segmentation**: Continuous boundary probabilities with temperature-sharpened Bernoulli sampling, allowing the model to autonomously group tokens into semantic concepts.
- **Heterogeneous Architecture**: Separate `d_token` (token-level) and `d_concept` (concept-level) dimensions, bridged by dynamic mean-pooling and concept replication.
- **Decoupled µP**: Custom initialization and learning rate scaling rules to ensure stable training across varying widths and depths, preventing gradient explosion in the cross-attention bridges.
- **Rust/CUDA Inference Engine**: A standalone CLI generation tool that memory-maps `safetensors` weights and executes the forward pass entirely on the GPU via `candle-core`, bypassing Python's GIL.
- **Modern Transformer Primitives**: Implements **SwiGLU** activations, **QK-Norm** for attention stabilization, **RMSNorm**, and custom cross-attention routing without self-attention in the decoder.

---

## 🏗️ Architecture Overview

The DLCM architecture processes text through four distinct stages:
1. **Token-Level Encoder**: Processes raw token embeddings into contextualized hidden states.
2. **Dynamic Segmenter**: Computes cosine similarity between adjacent tokens to predict semantic boundaries, then uses `scatter_add` to pool token states into discrete "Concepts".
3. **Concept-Level Model**: A deep Transformer that reasons over the compressed sequence of Concepts.
4. **Token-Level Decoder**: Uses Cross-Attention to project the high-level Concept representations back down to the token sequence for next-token prediction.

---

## 📂 Project Structure

```text
dlcm-hybrid/
├── python/dlcm/            # PyTorch models, Lightning modules, and µP optimizer
├── rust/
│   ├── dlcm-core/          # Core CUDA-accelerated ops (segmentation, replication)
│   ├── dlcm-inference/     # Standalone Rust inference engine (candle-core)
│   └── dlcm-python/        # PyO3 bindings for Python interop
├── tests/python/           # Unit tests, shape validation, and proxy training script
├── scripts/                # Orchestration scripts (export & generate)
├── run_colab_master.sh     # End-to-end automated pipeline for Google Colab (T4 GPU)
├── Cargo.toml              # Rust workspace configuration
└── pyproject.toml          # Python build configuration (maturin)
```

---

## 🚀 Getting Started

### Prerequisites
- **Python 3.10+** (with `pip` or `uv`)
- **Rust Toolchain** (`rustc`, `cargo`)
- **NVIDIA GPU & CUDA Toolkit** (Required for hardware acceleration)
- **System Dependencies**: `build-essential`, `pkg-config`, `libssl-dev`, `patchelf`

### 1. Python Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install maturin pytorch-lightning safetensors tokenizers datasets transformers
```

### 2. Compile the Rust Extensions
The Python training loop relies on Rust-compiled kernels for dynamic segmentation and concept replication.
```bash
# Ensure the 'cuda' feature is enabled in your Cargo.toml files
maturin develop --release
```

### 3. Run Local Tests
Verify the mathematical correctness of the architecture, gradient routing, and µP scaling.
```bash
PYTHONPATH=./python python tests/python/test_nano_pipeline.py
PYTHONPATH=./python python tests/python/test_mup.py
```

---

## 🏋️ Training & Inference

### Training the 87M Proxy Model
The repository includes a training script that trains the 87M parameter proxy model on the `HuggingFaceFW/fineweb-edu` dataset, utilizing streaming data and Automatic Mixed Precision (AMP).

```bash
PYTHONPATH=./python python tests/python/train_proxy.py
```
*This script will automatically export the trained weights to `dlcm_proxy_trained.safetensors` upon completion.*

> **💡 Note on Text Generation Quality & Training Duration:** 
> The default training script (`train_proxy.py`) runs for **1,000 steps** (approx. 8 minutes). This is sufficient to verify the core mathematical objective (achieving the theoretical ~4.0 compression ratio), but *not* enough for the model to learn human grammar, resulting in repetitive or artifact-heavy text. 
> 
> To achieve coherent, natural language generation, increase `max_steps` in `train_proxy.py` to **10,000+ steps** (approx. 1.5 - 2 hours on a T4 GPU). This is the "click point" where the cross-attention layers stabilize and the model begins generating actual sentences. For long runs, consider adding PyTorch Lightning `ModelCheckpoint` callbacks to prevent data loss on ephemeral cloud instances.

### Running the Rust Inference Engine
Once trained, you can bypass PyTorch entirely and generate text using the optimized Rust binary.

```bash
# 1. Compile the inference engine
cargo build --release --manifest-path rust/dlcm-inference/Cargo.toml

# 2. Generate text
./target/release/dlcm-inference \
    --weights dlcm_proxy_trained.safetensors \
    --prompt "The concept of dynamic compression in neural networks is" \
    --max-tokens 50 \
    --temperature 0.8
```

---

## ☁️ Automated Colab Pipeline

For users without a local NVIDIA GPU, this repository includes a fully automated bash script (`run_colab_master.sh`) that leverages the Google Colab CLI. 

It provisions a T4 GPU, installs all dependencies, patches CUDA 12.x compiler quirks mid-air, trains the model, and runs the Rust inference engine in a single command.

```bash
# Ensure your .env file contains your HF_TOKEN
echo "HF_TOKEN=your_token_here" > .env

# Package the project
tar -czf dlcm-payload.tar.gz python/ tests/ scripts/ rust/ Cargo.toml pyproject.toml

# Execute the pipeline
./run_colab_master.sh
```

---

## 🛠️ Engineering Highlights

This implementation required solving several complex engineering challenges at the intersection of PyTorch, Rust, and CUDA:

- **CUDA 12.x Compatibility Patching**: Dynamically patched `candle-kernels` headers during CI/CD to resolve `__hmax_nan` redefinition errors on modern NVIDIA toolchains.
- **Strict Memory Layouts**: Engineered custom tensor routing in Rust to ensure matrices are strictly `contiguous` before passing them to CUDA `matmul` and `scatter_add` kernels.
- **Weight Transposition Alignment**: Bridged the gap between PyTorch's `[out, in]` weight storage and Candle's expectations via on-the-fly transposition in the custom `dlcm-core` ops.
- **Zero-Copy Weight Loading**: Utilized `safetensors` memory-mapping in Rust to load the 120M parameter model onto the GPU in milliseconds.

---

## 🗺️ Roadmap & Future Work

This repository represents a fully functional baseline. Future iterations will focus on pushing the inference engine to production-grade speeds and scaling the architecture:

- [ ] **Rust KV-Caching**: Implement a `KvCache` struct in the Rust inference engine to store past Keys and Values. This will eliminate redundant $O(N)$ recomputation during autoregressive decoding, boosting generation speed from ~5 tok/s to 50+ tok/s.
- [ ] **FlashAttention Varlen Integration**: Replace the standard scaled dot-product attention with `candle-flash-attn` to natively support the variable-length concept segments generated by the dynamic segmenter, drastically reducing VRAM usage.
- [ ] **Advanced Decoding Strategies**: Implement Repetition Penalty, Min-P, and Mirostat sampling in the Rust sampler to further improve text diversity and eliminate early-training looping artifacts.
- [ ] **Multi-GPU / Tensor Parallelism**: Scale the Rust inference engine and PyTorch training pipeline to support distributed execution across multiple GPUs for larger model variants (e.g., 1B+ parameters).
- [ ] **Persistent Checkpointing & Resumption**: Integrate robust checkpointing into the training pipeline to allow safe pausing and resuming of multi-day training runs on ephemeral cloud VMs (like Colab).


---

## 🙏 Acknowledgements

This repository is an independent, high-performance implementation of the **Dynamic Large Concept Models (DLCM)** architecture. We extend our gratitude to the original authors for their groundbreaking research on dynamic semantic segmentation and decoupled heterogeneous routing. 

If you use this codebase in your research, please cite the original paper:
```bibtex
@article{qu2026dynamic,
  title={Dynamic Large Concept Models: Latent Reasoning in an Adaptive Semantic Space},
  author={Qu, Xingwei and Wang, Shaowen and Huang, Zihao and Zhang, Ge and Hua, Kai and Yin, Fan and Zhu, Rui-Jie and Zhou, Jundong and Min, Qiyang and Wang, Zihao and Li, Yizhi and Zhang, Tianyu and Xing, He and Zhang, Zheng and Song, Yuxuan and Zheng, Tianyu and Zeng, Zhiyuan and Lin, Chenghua and Huang, Wenhao},
  journal={arXiv preprint arXiv:2512.24617},
  year={2026}
}
```

Special thanks to the Hugging Face team for `candle-core` and `safetensors`, which made the Rust/CUDA inference engine possible.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Built with ❤️ using PyTorch, Rust, and Candle.*
