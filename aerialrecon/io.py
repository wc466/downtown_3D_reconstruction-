"""Small file-format helpers."""
import json
from pathlib import Path

import numpy as np


def write_ply(path, xyz, rgb=None):
    """Write a binary little-endian PLY point cloud. xyz: (N, 3) float; rgb: (N, 3) in [0, 1] or uint8."""
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    fields = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
    if rgb is not None:
        rgb = np.asarray(rgb).reshape(-1, 3)
        if rgb.dtype != np.uint8:
            rgb = np.clip(np.round(rgb * 255), 0, 255).astype(np.uint8)
        fields += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    data = np.empty(len(xyz), dtype=fields)
    data["x"], data["y"], data["z"] = xyz.T
    if rgb is not None:
        data["red"], data["green"], data["blue"] = rgb.T
    ply_types = {"<f4": "float", "u1": "uchar"}
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(xyz)}"]
    header += [f"property {ply_types[t]} {name}" for name, t in fields]
    header += ["end_header"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())


def read_ply(path):
    """Read a PLY written by `write_ply`. Returns (xyz, rgb or None)."""
    with open(path, "rb") as f:
        n, props = 0, []
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                _, t, name = line.split()
                props.append((name, {"float": "<f4", "uchar": "u1"}[t]))
        data = np.frombuffer(f.read(), dtype=props, count=n)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1)
    rgb = np.stack([data["red"], data["green"], data["blue"]], axis=1) if "red" in data.dtype.names else None
    return xyz, rgb


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
