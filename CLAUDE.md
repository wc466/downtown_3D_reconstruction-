# CLAUDE.md

Context for Claude Code sessions in this repo. Milestone 1's code was written in a
cloud container with no GPU; GPU runs happen on the owner's desktop.

## Project

Does adding aerial imagery (oblique aerial views, and true nadir orthophotos) to
ground-level photos improve 3D reconstruction with feed-forward geometry models?
The framing is scene reconstruction and completeness, e.g. recovering roofs that
ground photos can't see. Do not frame or build it as camera localization or
navigation. Output: a public GitHub repo plus figures for a LinkedIn post. Budget:
about 10–20 hours, so prefer a finished narrow result over a broad unfinished one.

- Models: vanilla MASt3R (core baseline); MASt3R fine-tuned on AerialMegaDepth (core
  comparison); VGGT only if time allows.
- E1: ground only. E2: ground + oblique aerial, using the example scenes shipped with
  AerialMegaDepth. E3 (stretch, the most interesting): ground + nadir orthophoto, i.e.
  the owner's phone photos of one downtown building (`data/my_scene/ground/`) plus a
  public-domain orthophoto of the block (USGS NAIP or county orthoimagery). Orthophotos
  are orthographic but the models assume pinhole cameras, so test the tile as-is and as
  a small crop (which approximates a long-focal-length pinhole view) and report both.
  For the resolution gap, try a few crop sizes and report how sensitive results are.
- Metrics: registration (does each aerial image register with the ground set; for
  pairs, the fraction localized); completeness (roof coverage, and % of the footprint
  covered if feasible); accuracy only where ground truth exists; runtime and peak GPU
  memory per run.
- Deliverables: README, environment setup, one script per experiment, `results/`;
  side-by-side point clouds per condition, rotating point-cloud GIFs, one summary
  table. The README "Observations" section stays a TODO: the owner writes it.

## Rules from the owner

- Public, license-clean data only. Record each dataset's license and citation in the
  README before it appears in a figure.
- No proprietary code or data from any employer.
- Never fabricate results, numbers, or citations. A failed run is a valid finding:
  report it. Every reference in the README must be a real paper you have verified.
- Ask before downloading anything larger than ~2 GB.
- Work in milestones and stop for the owner's review after each: (1) environment +
  vanilla MASt3R on one shipped example; (2) E1 + E2 with figures; (3) E3 (nadir);
  (4) README + GIFs.
- When editing code, show the diff and keep changes minimal; don't remove existing
  comments.
- Give shell commands in copy-paste-friendly code blocks.

## Where things stand

- Milestone 1 code is written and passes CPU tests with a tiny random-weight model
  (`python -m pytest -q`). It has **not** yet run with real checkpoints or on a GPU.
- Next: on the desktop, follow the README's "Setup" and "Milestone 1" sections. Show
  the owner the output of `scripts/check_env.py` and the `metrics.json` and
  `matches.png` from `results/pairs/.../vanilla/`, then stop for review.
- GPU machine: RTX 3070 (8 GB VRAM), Windows + WSL2 Ubuntu. Load one model at a time.

## Known issues and open items

- Licenses not yet verified (the pages were unreachable from the cloud container): the
  ULTRRA (`t03_*`, `t04_*`) and Accenture-NVS1 (`siteACC*`) example images, and the
  `kvuong2711/checkpoint-aerial-mast3r` model card. Verify them before any of these
  appear in a published figure, and update the README table.
- `dust3r.utils.image.load_images` center-crops square images to 4:3 unless
  `square_ok=True`, so make E3 orthophoto crops 4:3 up front.
- torch is pinned to 2.5.1 because MASt3R's local `.pth` loader calls `torch.load`
  without `weights_only=False`, and 2.6 changed that default.
- The shipped examples have no ground-truth poses. For E2, "registered" needs a proxy
  that is fixed before looking at results (e.g. at least N RANSAC-verified inliers plus
  a visual check), and the README must call it a proxy.
- Ground-only and ground+aerial reconstructions come out in different frames and
  scales: align them on the shared ground cameras before comparing coverage.
- Strip location EXIF from the owner's photos before committing them.
