"""Model aliases and loading."""
import torch

from aerialrecon.paths import add_mast3r_to_path

MODELS = {
    # MASt3R released by NAVER (weights: CC BY-NC-SA 4.0, see third_party/.../mast3r/CHECKPOINTS_NOTICE).
    "vanilla": "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric",
    # MASt3R fine-tuned on AerialMegaDepth (Vuong et al., CVPR 2025).
    "aerial": "kvuong2711/checkpoint-aerial-mast3r",
}

# A tiny, randomly initialised MASt3R with the same heads and outputs as the real model.
# Only for testing the pipeline without downloading checkpoints: its outputs are meaningless.
RANDOM_TINY = "random-tiny"
_RANDOM_TINY_ARCH = dict(
    pos_embed='RoPE100', patch_embed_cls='PatchEmbedDust3R', img_size=(512, 512),
    head_type='catmlp+dpt', output_mode='pts3d+desc24',
    depth_mode=('exp', -float('inf'), float('inf')), conf_mode=('exp', 1, float('inf')),
    enc_embed_dim=128, enc_depth=2, enc_num_heads=4,
    dec_embed_dim=128, dec_depth=10, dec_num_heads=4,  # the MASt3R head requires dec_depth > 9
    two_confs=True, desc_conf_mode=('exp', 0, float('inf')), landscape_only=False)


def resolve(name):
    """Map an alias ('vanilla', 'aerial') to a Hugging Face id; pass other ids/paths through."""
    return MODELS.get(name, name)


def load_mast3r(name, device):
    """Load MASt3R from an alias, a Hugging Face id, a local .pth file, or 'random-tiny'."""
    add_mast3r_to_path()
    from mast3r.model import AsymmetricMASt3R
    if name == RANDOM_TINY:
        torch.manual_seed(0)
        model = AsymmetricMASt3R(**_RANDOM_TINY_ARCH)
    else:
        model = AsymmetricMASt3R.from_pretrained(resolve(name))
    return model.to(device).eval()
