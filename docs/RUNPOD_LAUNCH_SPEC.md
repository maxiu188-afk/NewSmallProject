# RunPod launch specification

## Selected resources

| Setting | Selection | Reason |
|---|---|---|
| GPU | One NVIDIA A40, 48 GB | The A40 is compute capability `sm_86`, which is an unmodified upstream target; 48 GB leaves room for the first LLaMA-2 smoke, calibration, and controlled fake-quant matrix. |
| Container image | `nvidia/cuda:12.8.2-devel-ubuntu22.04` | The `devel` image supplies the CUDA headers and `nvcc` needed by the upstream `CUDAExtension`. |
| Python | 3.11 | Matches the modern local API smoke. |
| First modern PyTorch pair | `torch==2.11.0` from the `cu128` index | The PyTorch wheel and CUDA toolkit use the same 12.8 minor version. |
| First modern Transformers version | `5.14.1` | The portable F5 pipeline passed local API smoke at this pinned version. |
| Container disk | 50 GB | Holds the operating system, CUDA development image, compiler products, and temporary package caches. It is disposable. |
| Volume disk | 150 GB, mounted at `/workspace` | Holds the repository, model snapshots, datasets, experiment definitions, raw logs, and results across Pod restarts. |

The Pod host driver is not installed by the container. Before any package
installation, confirm that `nvidia-smi` reports an A40 and a driver that can
run CUDA 12.8; reject a mismatched host rather than modifying the source to fit
it.

## Layout on the persistent volume

```text
/workspace/NewSmallProject/       tracked repository checkout
/workspace/hf-cache/              model and tokenizer snapshots
/workspace/datasets/              downloaded calibration and evaluation data
/workspace/results/               ignored raw logs, manifests, and metrics
```

Set `HF_HOME` and `HF_HUB_CACHE` to `/workspace/hf-cache`. Keep the project
working tree separate from generated model/data artifacts, and do not copy a
macOS virtual environment onto the Pod.

## Launch sequence

1. Create the Pod with the selected GPU, image, container disk, and volume disk.
2. Clone the fixed Git revision into `/workspace/NewSmallProject` and obtain the
   unchanged upstream `QuaRot/` checkout beside it.
3. Run `scripts/runpod_preflight.py` before installing Python packages or
   modifying source. Save the resulting JSON under `results/runpod-preflight/`.
4. Create an independent Python 3.11 virtual environment. Install the selected
   `cu128` PyTorch wheel, then record `torch.__version__` and
   `torch.version.cuda` alongside `nvcc --version`.
5. Run the reference build and numerical gates in `RUNPOD_SMOKE_RUNBOOK.md`.
   Do not begin PPL, calibration, or performance work unless those gates pass.

## Boundary

This specification selects a compatible starting point; it is not a completed
RunPod preflight. The actual GPU, driver, toolkit, package versions, source
revisions, and commands must be captured from the created Pod as run evidence.
