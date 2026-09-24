"""Model loading tests that need no network access."""
import torch

from aerialrecon.model_zoo import RANDOM_TINY, load_mast3r, resolve


def test_aliases():
    assert resolve("vanilla") == "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    assert resolve("aerial") == "kvuong2711/checkpoint-aerial-mast3r"
    assert resolve("some/other-id") == "some/other-id"


def test_hub_format_roundtrip(tmp_path):
    """Save in Hugging Face format and reload through the same from_pretrained code path that
    downloads the real checkpoints (minus the network), to catch huggingface_hub incompatibilities."""
    model = load_mast3r(RANDOM_TINY, "cpu")
    model.save_pretrained(tmp_path)
    reloaded = load_mast3r(str(tmp_path), "cpu")
    assert type(reloaded) is type(model)
    for (k, a), (k2, b) in zip(model.state_dict().items(), reloaded.state_dict().items()):
        assert k == k2
        torch.testing.assert_close(a, b)
