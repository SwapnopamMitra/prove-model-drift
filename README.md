# prove-model-drift

Don't trust my binary. Don't trust my tests. Run this 40-line PyTorch script on your own machines.

## What is this?

Standard machine-learning training is not bit-exact across different hardware. Even if you fix the random seed, the initial weights, and the dataset, the underlying floating-point math drifts between CPU architectures, GPUs, and library builds.

This repository contains a minimal PyTorch script that trains a small neural network and prints the SHA-256 hash of the final model weights.

## How to run it

You need Python 3 and PyTorch.

```bash
git clone https://github.com/SwapnopamMitra/prove-model-drift.git
cd prove-model-drift
python3 prove_model_drift.py
```

## The Results (Tested October 2026)

Same script, same seed (42), four different environments.

**The final loss was identical on all four. The model weights were completely different.**

| Environment | Device | PyTorch | Final Loss | Weights SHA-256 |
|---|---|---|---|---|
| Colab CPU | Intel Xeon @ 2.20GHz | 2.11.0+cpu | 0.69410753 | `67ed1ee5...` |
| Colab GPU | NVIDIA Tesla T4 | 2.11.0+cu130 | 0.69410753 | `0ae59421...` |
| TPU VM (host CPU) | AMD EPYC 7B13 | 2.14.1+cpu | 0.69410753 | `ef84b798...` |
| Python Runner | Intel Xeon @ 2.80GHz | 2.8.0 | 0.69410753 | `098cbd36...` |

### Full hashes

```text
Colab CPU      67ed1ee52938f57d0c6d563eb84057e8a89aa164caf2186a0e64152dc23692a2
Colab GPU      0ae59421ab7821874cc8971673cc92743ade3d8b3244620893d17caf90a829d3
TPU VM (host)  ef84b798c8020f0100c2864bb9956b783ea1da566b66d64ddc92a4dcc2e6aa2d
Python Runner  098cbd36d74178015a5ced2560e0a296b6d57fabbb33a7f3b2be2cabd6c868a4
```

### Raw run metadata

<details>
<summary>Colab CPU</summary>

```json
{
  "seed": 42,
  "device": "cpu",
  "pytorch": "2.11.0+cpu",
  "python": "3.13.18",
  "os": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "cpu": { "cpu_name": "Intel(R) Xeon(R) CPU @ 2.20GHz" },
  "gpu": null,
  "cuda_version": null,
  "cpu_threads_used_by_torch": 2,
  "final_loss": 0.69410753,
  "weights_sha256": "67ed1ee52938f57d0c6d563eb84057e8a89aa164caf2186a0e64152dc23692a2"
}
```
</details>

<details>
<summary>Colab GPU</summary>

```json
{
  "seed": 42,
  "device": "cuda",
  "pytorch": "2.11.0+cu130",
  "python": "3.13.16",
  "os": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "cpu": { "cpu_name": "Intel(R) Xeon(R) CPU @ 2.00GHz" },
  "gpu": "Tesla T4",
  "cuda_version": "13.0",
  "cpu_threads_used_by_torch": 1,
  "final_loss": 0.69410753,
  "weights_sha256": "0ae59421ab7821874cc8971673cc92743ade3d8b3244620893d17caf90a829d3"
}
```
</details>

<details>
<summary>TPU VM (host CPU)</summary>

```text
Successfully renamed PrivateUse1 backend to 'tpu'. Device: device(type='tpu')
Registered Python module for 'tpu'.
Device type: tpu, Device index: default
Initializing TPU distributed runtime
```

```json
{
  "seed": 42,
  "device": "cpu",
  "pytorch": "2.14.1+cpu",
  "python": "3.13.18",
  "os": "Linux-6.6.159+-x86_64-with-glibc2.39",
  "cpu": { "cpu_name": "AMD EPYC 7B13" },
  "gpu": null,
  "cuda_version": null,
  "cpu_threads_used_by_torch": 12,
  "final_loss": 0.69410753,
  "weights_sha256": "ef84b798c8020f0100c2864bb9956b783ea1da566b66d64ddc92a4dcc2e6aa2d"
}
```
</details>

<details>
<summary>Python Runner</summary>

```json
{
  "seed": 42,
  "device": "cpu",
  "pytorch": "2.8.0",
  "python": "3.13.7",
  "os": "Linux-6.8.0-1070-gcp-x86_64-with-glibc2.40",
  "cpu": { "cpu_name": "Intel(R) Xeon(R) CPU @ 2.80GHz" },
  "gpu": null,
  "cuda_version": null,
  "cpu_threads_used_by_torch": 2,
  "final_loss": 0.69410753,
  "weights_sha256": "098cbd36d74178015a5ced2560e0a296b6d57fabbb33a7f3b2be2cabd6c868a4"
}
```
</details>

## Why this matters

If you are training an ML model for clinical use, finance, or compliance, you need to prove the model running today is mathematically identical to the model you validated last month. A matching loss is not proof. A matching hash is proof.

> **Note:** `torch.use_deterministic_algorithms(True)` only guarantees run-to-run repeatability on the same machine. It does not fix cross-hardware drift.

## The Fix

I built **CAGM**, a fixed-point ML classifier that removes floating-point math from the training loop and emits a SHA-256 proof of the entire training trajectory.

I ran CAGM across the same four environments. Every run produced the exact same proof.

Four environments. Two CPU environments, an RTX 3050, and a Tesla T4. Same dataset. Same accuracy. Same training proof. Same model SHA-256.

| Environment | Accuracy | Training proof | Model SHA-256 |
|---|---|---|---|
| Colab CPU | 0.514460 | Match | Match |
| TPU VM (host CPU) | 0.514460 | Match | Match |
| Local RTX 3050 6GB | 0.514460 | Match | Match |
| Colab Tesla T4 | 0.514460 | Match | Match |

**Dataset SHA-256:**

```text
737649857cd126fb09cd3cd9d0f63b39353f09eb73fc6c6642597ff3f35d3f96
```

**Model SHA-256 (`cagm_model1.onnx`), identical in all four environments:**

```text
f32522987216045155d0be61753f6f9d7218ee61aea427b4ea08e2faf14efe5e
```

If you want to see how it works, look at [samayuktam.com](https://samayuktam.com).
