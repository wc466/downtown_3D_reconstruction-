# Do aerial and nadir views improve feed-forward 3D reconstruction?

Ground-level photos of a building show its facades but not its roof. This project tests
whether adding aerial imagery (oblique aerial views, and true nadir orthophotos) to a set of
ground photos gives more complete 3D reconstructions with feed-forward geometry models
(MASt3R, and MASt3R fine-tuned on AerialMegaDepth). The question is about scene
reconstruction and completeness, e.g. whether roofs that ground photos cannot see get filled in.

> **Status:** work in progress. Nothing below is a result yet.

| Milestone | State |
|---|---|
| 1. Environment + vanilla MASt3R on one shipped example | Code ready and CPU smoke-tested; waiting on the first GPU run |
| 2. E1 (ground only) + E2 (ground + oblique aerial), with figures | Not started |
| 3. E3 (ground + nadir orthophoto) | Not started |
| 4. README + GIFs | Not started |

## Setup (Windows + WSL2 with an NVIDIA GPU)

Developed for an RTX 3070 (8 GB) under WSL2 Ubuntu. Plain Linux works the same from step 2.

1. **Windows:** install a current NVIDIA driver. WSL2 uses the Windows driver, so don't
   install an NVIDIA driver inside Ubuntu.
2. **In the WSL Ubuntu terminal**, check that the GPU is visible:
   ```bash
   nvidia-smi   # should list the GPU and show "CUDA Version: 12.4" or higher
   ```
3. Get the code. The AerialMegaDepth code, which includes MASt3R and DUSt3R, is a git
   submodule pinned to commit `cdf478f`:
   ```bash
   git clone https://github.com/wc466/downtown_3D_reconstruction-.git
   cd downtown_3D_reconstruction-
   git checkout claude/laughing-hypatia-1drqbn   # until this branch is merged
   git submodule update --init third_party/aerial-megadepth
   ```
4. Create the Python environment. This downloads about 3 GB, mostly PyTorch with its
   bundled CUDA libraries:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   source $HOME/.local/bin/env
   uv venv --python 3.11 .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```
5. Check the machine and run the tests. The tests use a tiny random-weight model on CPU,
   take about a minute, and download nothing:
   ```bash
   python scripts/check_env.py
   python -m pytest -q
   ```
   `check_env.py` also prints the download size of each checkpoint.

Notes:
- The message `Warning, cannot find cuda-compiled version of RoPE2D, using a slow pytorch
  version instead` is expected. The optional CUDA kernel only affects speed. Each run
  records which implementation it used (`rope_impl` in `metrics.json`), so runtimes are
  only compared between runs that used the same one.
- If a run is killed while loading a model, WSL may have too little RAM. Create
  `%UserProfile%\.wslconfig` on Windows containing `[wsl2]` and, on the next line,
  `memory=12GB`, then run `wsl --shutdown` from PowerShell and reopen Ubuntu.

## Milestone 1: vanilla MASt3R on one shipped example

```bash
python scripts/run_pair.py --model vanilla
```

The first run downloads the checkpoint into `~/.cache/huggingface`. The default image
pair and matching settings are the ones in AerialMegaDepth's
`mast3r/demo_mast3r_nongradio.py`. Outputs go to
`results/pairs/<pair>/vanilla/`:

- `matches.png`: the two network inputs with a random subset of matches drawn as lines
- `pair.ply`: both views' points in view 1's frame, confidence ≥ 1.5 (not committed to git)
- `metrics.json`: match count, per-view confidence, estimated focal length and field of
  view, wall-clock time and peak GPU memory per stage, and software/GPU versions and commits

To run the fine-tuned model on the same pair, use `--model aerial`.
See `python scripts/run_pair.py --help` for other options.

## Models

| Alias | Checkpoint | License |
|---|---|---|
| `vanilla` | [`naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric`](https://huggingface.co/naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric) | CC BY-NC-SA 4.0, plus the licenses of its training datasets ([`CHECKPOINTS_NOTICE`](https://github.com/kvuong2711/aerial-megadepth/blob/cdf478ff4fdaf888fb91c940bc48d55095e2cdbd/mast3r/CHECKPOINTS_NOTICE)). The upstream README calls the Map-free dataset license "very restrictive". |
| `aerial` | [`kvuong2711/checkpoint-aerial-mast3r`](https://huggingface.co/kvuong2711/checkpoint-aerial-mast3r) | **Not yet verified.** Check the model card. |

## Data and licenses

Every dataset must have a verified license and citation here before it appears in a figure.

| Data | Used for | License | Status |
|---|---|---|---|
| AerialMegaDepth example images from the [ULTRRA challenge](https://sites.google.com/view/ultrra-wacv-2025) (`t03_*`, `t04_*` folders; IARPA WRIVA data, distributed via IEEE DataPort) | M1, E2 | Unknown | **Not yet verified.** The IEEE DataPort terms page couldn't be reached from the build machine. |
| AerialMegaDepth example images from [Accenture-NVS1](https://arxiv.org/abs/2503.18711) (`siteACC*` folders) | E1, E2 | Unknown | **Not yet verified** (same reason) |
| AerialMegaDepth example images `WGLBS/maidan` | Not planned | Unknown (source unidentified) | Not used |
| Nadir orthophoto (USGS NAIP or county orthoimagery) | E3 | To be decided | Not chosen yet |
| My own phone photos (`data/my_scene/ground/`) | E3 | Mine | Location EXIF must be stripped before committing |

Code: the MASt3R and DUSt3R code in the submodule is CC BY-NC-SA 4.0 (non-commercial;
`mast3r/LICENSE`). The AerialMegaDepth repository has no top-level LICENSE file at the
pinned commit.

## Observations

TODO (author)

- TODO
- TODO
- TODO

## References

```bibtex
@inproceedings{vuong2025aerialmegadepth,
  title={AerialMegaDepth: Learning Aerial-Ground Reconstruction and View Synthesis},
  author={Vuong, Khiem and Ghosh, Anurag and Ramanan, Deva and Narasimhan, Srinivasa and Tulsiani, Shubham},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025},
}

@misc{mast3r_arxiv24,
  title={Grounding Image Matching in 3D with MASt3R},
  author={Vincent Leroy and Yohann Cabon and Jerome Revaud},
  year={2024},
  eprint={2406.09756},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@inproceedings{dust3r_cvpr24,
  title={DUSt3R: Geometric 3D Vision Made Easy},
  author={Shuzhe Wang and Vincent Leroy and Yohann Cabon and Boris Chidlovskii and Jerome Revaud},
  booktitle = {CVPR},
  year = {2024}
}
```

These BibTeX entries are copied from the authors' own READMEs at the pinned commit. Each
dataset's citation will be added once its license has been verified.

## Repository layout

```
aerialrecon/       shared helpers: paths, model loading, profiling, I/O, plotting
scripts/           one script per step (check_env.py, run_pair.py, ...)
tests/             CPU tests; the smoke test runs the whole pipeline with random weights
third_party/       aerial-megadepth submodule (MASt3R + DUSt3R + CroCo), pinned
data/my_scene/     my own photos for E3 (not committed until location EXIF is stripped)
results/           figures and metrics.json files (point clouds are not committed)
```

## License

TODO: license for this repository's own code (dependencies keep their own licenses).
