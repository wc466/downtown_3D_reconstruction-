import numpy as np

from aerialrecon.io import read_ply, write_ply


def test_ply_roundtrip_with_float_colors(tmp_path):
    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(100, 3)).astype(np.float32)
    rgb = rng.uniform(size=(100, 3))
    write_ply(tmp_path / "a.ply", xyz, rgb)
    xyz2, rgb2 = read_ply(tmp_path / "a.ply")
    np.testing.assert_array_equal(xyz, xyz2)
    np.testing.assert_array_equal(np.round(rgb * 255).astype(np.uint8), rgb2)


def test_ply_without_colors(tmp_path):
    xyz = np.arange(12, dtype=np.float32).reshape(4, 3)
    write_ply(tmp_path / "b.ply", xyz)
    xyz2, rgb2 = read_ply(tmp_path / "b.ply")
    np.testing.assert_array_equal(xyz, xyz2)
    assert rgb2 is None
