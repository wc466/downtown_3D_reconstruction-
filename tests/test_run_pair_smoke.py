"""End-to-end pipeline test with a tiny random-weight MASt3R on CPU (no checkpoint download).

This checks plumbing only (imports, shapes, outputs). The numbers it produces are meaningless.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_pair  # noqa: E402

from aerialrecon.io import read_ply  # noqa: E402


def test_run_pair_random_tiny(tmp_path):
    # The default pair mixes a portrait and a landscape image, which exercises
    # dust3r's per-pair (unbatched) output path.
    metrics = run_pair.main(["--model", "random-tiny", "--device", "cpu", "--out", str(tmp_path)])

    for name in ("matches.png", "pair.ply", "metrics.json"):
        assert (tmp_path / name).is_file(), name
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["n_matches"] == metrics["n_matches"] >= 0
    (h0, w0), (h1, w1) = saved["network_input_hw"]
    assert h0 == 512 and w0 < 512 and w1 == 512 and h1 < 512  # portrait, then landscape
    assert all(d % 16 == 0 for d in (h0, w0, h1, w1))  # dust3r crops to multiples of the patch size
    assert set(saved["timing_and_memory"]) == {"load_model", "inference", "matching"}
    # Random weights can give degenerate pointmaps; the script must record that, not crash.
    assert all(f["focal_px"] is None or f["focal_px"] > 0 for f in saved["focal_estimates"])
    xyz, rgb = read_ply(tmp_path / "pair.ply")
    assert len(xyz) == sum(saved["points_kept_per_view"])
