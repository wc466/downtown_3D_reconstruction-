"""Per-stage wall-clock time and peak GPU memory, plus a record of the run environment."""
import platform
import subprocess
import time
from contextlib import contextmanager

import torch

from aerialrecon.paths import AMD_ROOT, REPO_ROOT

MIB = 2 ** 20


class Profiler:
    """Records wall-clock seconds and peak CUDA memory (MiB) for each named stage.

    Peak memory is reset at the start of every stage, so each number is the peak
    *during* that stage, including memory already held (e.g. model weights).
    """

    def __init__(self, device):
        self.cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        self.stages = {}

    @contextmanager
    def stage(self, name):
        if self.cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        yield
        if self.cuda:
            torch.cuda.synchronize()
        rec = {"seconds": round(time.perf_counter() - t0, 3)}
        if self.cuda:
            rec["peak_allocated_mib"] = round(torch.cuda.max_memory_allocated() / MIB, 1)
            rec["peak_reserved_mib"] = round(torch.cuda.max_memory_reserved() / MIB, 1)
        self.stages[name] = rec


def _git_commit(path):
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment_info(device):
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "repo_commit": _git_commit(REPO_ROOT),
        "aerial_megadepth_commit": _git_commit(AMD_ROOT),
    }
    if str(device).startswith("cuda") and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.device(device))
        info["gpu"] = props.name
        info["gpu_total_mib"] = round(props.total_memory / MIB)
    try:  # "cuRoPE2D" if the optional CUDA kernel was compiled, else the slower PyTorch "RoPE2D"
        import models.pos_embed  # croco's package, importable once dust3r has been imported
        info["rope_impl"] = models.pos_embed.RoPE2D.__name__
    except ImportError:
        info["rope_impl"] = None
    return info
