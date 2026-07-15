# CUDA and PyTorch compatibility policy

> This policy applies to every future server-side QuaRot script. It is retained
> while the project is local-only so that a later CUDA run does not repeat known
> environment mismatches.

## Why this is a separate policy

There are three different version facts, and they must not be conflated:

| Fact | Inspect with | Meaning |
|---|---|---|
| NVIDIA driver capability | `nvidia-smi` | Maximum CUDA generation the installed driver advertises; not necessarily the compiler version |
| CUDA toolkit/compiler | `nvcc --version` | Toolkit used to compile a custom CUDA extension |
| PyTorch CUDA build | `python -c "import torch; print(torch.version.cuda)"` | CUDA runtime ABI selected by the installed PyTorch wheel |

The upstream QuaRot `setup.py` builds a `CUDAExtension`. For that path, the
toolkit selected by `nvcc` and the PyTorch wheel must be treated as a matched
build pair: same CUDA major version is the minimum expectation; the same minor
version is the default policy. A successful `import torch` alone is not enough
evidence that the extension will compile.

## Approved future environment pairs

Use separate virtual environments instead of one “universal” CUDA environment.
The first candidate pairs are:

| Server `nvcc` | PyTorch wheel family | Intended use |
|---|---|---|
| CUDA 12.8 | `torch==2.11.0` from PyTorch `cu128` index | extension build and runtime on that host |
| CUDA 13.0 | `torch==2.11.0` from PyTorch `cu130` index | extension build and runtime on that host |

For ordinary PyTorch execution that does **not** build a custom extension, a
`cu128` wheel can often run on a CUDA-13-capable driver through NVIDIA backward
compatibility. This convenience must not be used to justify compiling QuaRot
with CUDA 13 `nvcc` against a `cu128` wheel.

The upstream reference pins `torch==2.2.1`, which predates these wheel families.
It may remain useful for source behaviour comparison, but it is not the default
choice for a CUDA 12.8/13.0 extension build without a recorded compatibility
check.

## Mandatory checks in future server scripts

Before any package installation, CUDA build, or benchmark, a server script must:

1. Save `nvidia-smi`, `nvcc --version`, Python version, and GPU compute
   capability to the run directory.
2. Select an explicit environment label, for example `cu128` or `cu130`; never
   infer it solely from the CUDA value printed by `nvidia-smi`.
3. After installation, save `torch.__version__` and `torch.version.cuda`.
4. Fail before extension compilation if `nvcc` and `torch.version.cuda` do not
   satisfy the selected matched-pair policy.
5. Save the original build log before proposing an architecture, CUDA, or source
   patch.

Future scripts should expose the selected CUDA label as an explicit parameter
and write it to the experiment manifest. They must not hard-code a single wheel
or silently substitute a CPU, MPS, or different CUDA build.

## Sources

- [PyTorch previous versions and wheel indexes](https://pytorch.org/get-started/previous-versions/)
- [NVIDIA CUDA minor-version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
- [PyTorch CUDA extension requirements](https://docs.pytorch.org/docs/stable/cpp_extension.html)
