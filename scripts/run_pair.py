"""Run MASt3R on one image pair; save matches, a point cloud and run metrics.

Milestone 1 uses this to check that the environment works. The default pair is the
one used by AerialMegaDepth's demo_mast3r_nongradio.py, and the matching settings
(subsample=4, pixel_tol=3, match_conf=0.3, border=3) are copied from that demo.

Examples:
    python scripts/run_pair.py --model vanilla
    python scripts/run_pair.py --model aerial
    python scripts/run_pair.py --model vanilla --images a.jpg b.jpg
    python scripts/run_pair.py --model random-tiny --device cpu   # pipeline test only
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerialrecon.io import save_json, write_ply  # noqa: E402
from aerialrecon.model_zoo import load_mast3r, resolve  # noqa: E402
from aerialrecon.paths import AMD_ASSETS, RESULTS, add_mast3r_to_path  # noqa: E402
from aerialrecon.profiling import Profiler, environment_info  # noqa: E402
from aerialrecon.viz import plot_matches  # noqa: E402

add_mast3r_to_path()
from dust3r.inference import inference  # noqa: E402
from dust3r.post_process import estimate_focal_knowing_depth  # noqa: E402
from dust3r.utils.image import load_images  # noqa: E402
from mast3r.fast_nn import extract_correspondences_nonsym  # noqa: E402

DEFAULT_PAIR = [
    AMD_ASSETS / "t04_v13_s00_r01_VaryingAltitudes_WACV_test_A10" / "00572.jpg",
    AMD_ASSETS / "t04_v13_s00_r01_VaryingAltitudes_WACV_test_A10" / "00563.jpg",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="vanilla",
                   help="'vanilla', 'aerial', a Hugging Face id, a local .pth, or 'random-tiny' (test only)")
    p.add_argument("--images", nargs=2, type=Path, default=DEFAULT_PAIR)
    p.add_argument("--size", type=int, default=512, help="long side of the network input")
    p.add_argument("--device", default=None, help="default: cuda if available, else cpu")
    p.add_argument("--subsample", type=int, default=4)
    p.add_argument("--pixel-tol", type=int, default=3)
    p.add_argument("--match-conf", type=float, default=0.3, help="min descriptor confidence of a match")
    p.add_argument("--pc-conf", type=float, default=1.5, help="min point confidence kept in the .ply")
    p.add_argument("--out", type=Path, default=None, help="default: results/pairs/<pair>/<model>")
    return p.parse_args(argv)


def to_rgb(view):
    """Network input tensor (1, 3, H, W) in [-1, 1] -> (H, W, 3) float in [0, 1]."""
    return (view["img"][0].permute(1, 2, 0).cpu().numpy() * 0.5 + 0.5).clip(0, 1)


def estimate_focal(pts3d):
    """Focal length (px) of a view from its pointmap in its own camera frame, principal point at center."""
    H, W = pts3d.shape[:2]
    pp = torch.tensor([W / 2, H / 2], dtype=pts3d.dtype)
    focal = float(estimate_focal_knowing_depth(pts3d[None], pp, focal_mode="weiszfeld")[0])
    if not (math.isfinite(focal) and focal > 0):  # degenerate pointmap: record it, don't crash
        return {"focal_px": None, "hfov_deg": None}
    return {"focal_px": round(focal, 2), "hfov_deg": round(math.degrees(2 * math.atan(W / 2 / focal)), 2)}


def main(argv=None):
    args = parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    images = [p.resolve() for p in args.images]
    model_tag = args.model if args.model in ("vanilla", "aerial", "random-tiny") else Path(args.model).stem
    pair_tag = f"{images[0].parent.name}__{images[0].stem}__{images[1].stem}"
    out = args.out or RESULTS / "pairs" / pair_tag / model_tag
    out.mkdir(parents=True, exist_ok=True)

    prof = Profiler(device)
    with prof.stage("load_model"):
        model = load_mast3r(args.model, device)
    views = load_images([str(p) for p in images], size=args.size, verbose=False)

    # Both directions, as MASt3R's own pipelines do: (1, 2) gives matches and view 2 in
    # view 1's frame; (2, 1) gives view 2's pointmap in its own frame, for its focal length.
    with prof.stage("inference"):
        output = inference([(views[0], views[1]), (views[1], views[0])], model, device,
                           batch_size=1, verbose=False)
    pred1, pred2 = output["pred1"], output["pred2"]  # index 0: pair (1, 2); index 1: pair (2, 1)

    with prof.stage("matching"):
        xy0, xy1, conf = extract_correspondences_nonsym(
            pred1["desc"][0], pred2["desc"][0], pred1["desc_conf"][0], pred2["desc_conf"][0],
            device=device, subsample=args.subsample, pixel_tol=args.pixel_tol)
    keep = (conf >= args.match_conf).cpu().numpy()
    xy0, xy1 = xy0.cpu().numpy()[keep], xy1.cpu().numpy()[keep]
    n_before_border = len(xy0)
    border = 3  # ignore matches this close to the image edge (as in the upstream demo)
    ok = np.ones(len(xy0), dtype=bool)
    for xy, v in ((xy0, views[0]), (xy1, views[1])):
        H, W = v["true_shape"][0]
        ok &= (xy[:, 0] >= border) & (xy[:, 0] < W - border) & (xy[:, 1] >= border) & (xy[:, 1] < H - border)
    xy0, xy1 = xy0[ok], xy1[ok]

    rgb0, rgb1 = to_rgb(views[0]), to_rgb(views[1])
    plot_matches(rgb0, rgb1, xy0, xy1, out / "matches.png",
                 title=f"{model_tag}: {len(xy0)} matches (conf >= {args.match_conf})")

    # Point cloud of both views in view 1's frame, from the (1, 2) pass.
    pts = [pred1["pts3d"][0], pred2["pts3d_in_other_view"][0]]
    confs = [pred1["conf"][0], pred2["conf"][0]]
    xyz, rgb = [], []
    for p, c, im in zip(pts, confs, (rgb0, rgb1)):
        m = (c >= args.pc_conf).numpy()
        xyz.append(p.numpy()[m])
        rgb.append(im[m])
    write_ply(out / "pair.ply", np.concatenate(xyz), np.concatenate(rgb))

    metrics = {
        "model": args.model,
        "model_id": resolve(args.model),
        "images": [str(p.relative_to(Path.cwd())) if p.is_relative_to(Path.cwd()) else str(p) for p in images],
        "network_input_hw": [[int(x) for x in v["true_shape"][0]] for v in views],
        "settings": {k: getattr(args, k) for k in ("size", "subsample", "pixel_tol", "match_conf", "pc_conf")},
        "n_matches": int(len(xy0)),
        "n_matches_before_border_filter": int(n_before_border),
        "points_kept_per_view": [int(len(x)) for x in xyz],
        "conf_median_per_view": [round(float(c.median()), 3) for c in confs],
        "focal_estimates": [estimate_focal(pred1["pts3d"][0]), estimate_focal(pred1["pts3d"][1])],
        "timing_and_memory": prof.stages,
        "environment": environment_info(device),
    }
    save_json(out / "metrics.json", metrics)
    print(f"{model_tag}: {len(xy0)} matches; outputs in {out}")
    return metrics


if __name__ == "__main__":
    main()
