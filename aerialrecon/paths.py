"""Repository paths and import setup for the pinned MASt3R / DUSt3R code."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AMD_ROOT = REPO_ROOT / "third_party" / "aerial-megadepth"  # git submodule, pinned commit
MAST3R_ROOT = AMD_ROOT / "mast3r"  # vendored MASt3R (includes dust3r/ and croco/)
AMD_ASSETS = AMD_ROOT / "assets"  # example images shipped with AerialMegaDepth
RESULTS = REPO_ROOT / "results"


def add_mast3r_to_path():
    """Make `mast3r` and `dust3r` importable from the AerialMegaDepth submodule."""
    if not (MAST3R_ROOT / "mast3r" / "model.py").is_file():
        raise ImportError(
            f"MASt3R code not found under {MAST3R_ROOT}.\n"
            "Run: git submodule update --init third_party/aerial-megadepth")
    if str(MAST3R_ROOT) not in sys.path:
        sys.path.insert(0, str(MAST3R_ROOT))
    import mast3r.utils.path_to_dust3r  # noqa: F401  (puts dust3r on sys.path)
