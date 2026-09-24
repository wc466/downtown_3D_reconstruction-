"""Check the machine before running experiments.

Reports Python/PyTorch/CUDA, the GPU and its memory, RAM, free disk, whether the pinned
MASt3R code imports, and the download size of each checkpoint on Hugging Face.
Downloads nothing. Exits non-zero if something required is missing.

    python scripts/check_env.py
"""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerialrecon.model_zoo import MODELS  # noqa: E402
from aerialrecon.paths import REPO_ROOT, add_mast3r_to_path  # noqa: E402

GIB = 2 ** 30
problems = []


def row(label, value):
    print(f"  {label:<26} {value}")


def main():
    import torch

    print("Software")
    row("python", sys.version.split()[0])
    row("torch", f"{torch.__version__} (built for CUDA {torch.version.cuda})")

    print("GPU")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        row("device", f"{props.name} (compute {props.major}.{props.minor})")
        row("memory", f"{props.total_memory / GIB:.1f} GiB")
        x = torch.randn(1024, 1024, device="cuda")
        row("test matmul", f"ok ({float((x @ x).abs().mean()):.1f})")
    else:
        row("device", "NONE - torch.cuda.is_available() is False")
        problems.append("No CUDA GPU visible to PyTorch (run `nvidia-smi` inside WSL to check the driver)")

    print("Memory and disk")
    if Path("/proc/meminfo").exists():
        kib = int(Path("/proc/meminfo").read_text().split("MemTotal:")[1].split()[0])
        row("RAM visible to this OS", f"{kib / 2**20:.1f} GiB")
        if kib / 2**20 < 8:
            problems.append("Less than 8 GiB RAM visible; raise the WSL memory limit in %UserProfile%\\.wslconfig")
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    for label, path in (("free disk (repo)", REPO_ROOT), ("free disk (HF cache)", hf_home)):
        while not path.exists():
            path = path.parent
        row(label, f"{shutil.disk_usage(path).free / GIB:.1f} GiB")

    print("MASt3R code")
    try:
        add_mast3r_to_path()
        from mast3r.model import AsymmetricMASt3R  # noqa: F401
        from dust3r.inference import inference  # noqa: F401
        row("import mast3r / dust3r", "ok")
    except Exception as e:  # report and continue with the other checks
        row("import mast3r / dust3r", f"FAILED: {e!r}")
        problems.append("MASt3R code does not import (did you run `git submodule update --init`?)")

    print("Checkpoints on Hugging Face (not downloaded)")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        for alias, repo_id in MODELS.items():
            try:
                info = api.model_info(repo_id, files_metadata=True)
                size = sum(s.size or 0 for s in info.siblings)
                row(alias, f"{repo_id}: {size / GIB:.2f} GiB")
            except Exception as e:
                row(alias, f"{repo_id}: could not query ({type(e).__name__})")
                problems.append(f"Cannot reach Hugging Face model {repo_id}")
    except ImportError:
        problems.append("huggingface_hub is not installed")

    print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
